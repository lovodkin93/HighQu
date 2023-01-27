import os
import sys
import torch
import json
import time
import logging
import random
import argparse
import numpy as np
import itertools
from typing import List
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import get_linear_schedule_with_warmup

from arguments import get_args
from policy import Policy
from data_pool import DataPool
from reward import Reward
from utils.utils import ensure_dir, ceil_div, reduce_mean, reduce_sum, is_t5_model_def, get_jsonl_line_data, custom_clean_special_tokens, add_control_code
from utils.constants import ROUGE1, ROUGE2, ROUGEL, ROUGELSUM, all_rouges
from rouge_score import rouge_scorer
from preprocess_utils.preprocessor import get_special_tokens_constants
from transformers.trainer_utils import get_last_checkpoint
import wandb
from accelerate import Accelerator, DistributedType
import pdb
import shutil
from copy import deepcopy
import pickle

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


full_sample_dataloader = None

class PromptDataset(Dataset):
    def __init__(self, path, model_name):
        self.model_name = model_name
        with open(path, 'r') as f1:
            self.inputs = [json.loads(s.strip())["input"].strip() for s in f1.readlines()]
        self.path = path


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {'input': self.inputs[idx], 'jsonl_id': str(idx)}


class PromptCollator(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.max_source_length = args.max_source_length if hasattr(args, 'max_source_length') else None
        self.add_global_attention = args.add_global_attention
        self.add_global_attention_on_highlights = args.add_global_attention_on_highlights
        self.input_prefix = f"{args.source_prefix.strip()} " if args.source_prefix is not None else "" # relevant for t5 models

    def __call__(self, sequences):
        input_text = [self.input_prefix + sequence['input'] for sequence in sequences]
        jsonl_ids = [sequence['jsonl_id'] for sequence in sequences] # index of input texts in the input dataset

        encodings_dict = self.tokenizer(input_text, max_length= self.max_source_length, padding=True, truncation=True) 
        input_ids = torch.as_tensor(encodings_dict['input_ids'])
        attention_mask = torch.as_tensor(encodings_dict['attention_mask'])
        
        global_attention_mask = []
        if self.add_global_attention:
            for input_ids_instance in tqdm(encodings_dict['input_ids']):
                curr_global_attention_mask = [0 for _ in range(len(input_ids_instance))]
                curr_global_attention_mask[0] = 1

                ids_with_global_attention = self.tokenizer.additional_special_tokens_ids

                if self.add_global_attention_on_highlights:
                    for input_id_idx, input_id in enumerate(input_ids_instance):
                        # Put attention on highlight tokens
                        if input_id in ids_with_global_attention: 
                            curr_global_attention_mask[input_id_idx] = 1

                global_attention_mask.append(curr_global_attention_mask)
        global_attention_mask = torch.as_tensor(global_attention_mask)
        return input_ids, attention_mask, global_attention_mask, jsonl_ids



class SequenceDataset(Dataset):
    def __init__(self, data_pool, model_name: str):
        self.model_name = model_name
        self.inputs, self.jsonl_ids, self.generated_reductions, self.cat_tokens = data_pool.get_data()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {'input': self.inputs[idx],
                'jsonl_id': self.jsonl_ids[idx],
                'generated_reduction': self.generated_reductions[idx],
                'cat_tokens': self.cat_tokens[idx]
                }



class SequenceCollator(object):
    def __init__(self, tokenizer, max_source_length, max_target_length, add_global_attention, add_global_attention_on_highlights):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.add_global_attention = add_global_attention
        self.add_global_attention_on_highlights = add_global_attention_on_highlights

    def __call__(self, sequences):
        input_texts = [sequence['input'] for sequence in sequences]
        jsonl_ids = [sequence['jsonl_id'] for sequence in sequences]
        generated_reductions = [sequence['generated_reduction'] for sequence in sequences]
        cat_ids = [self.tokenizer.convert_tokens_to_ids(sequence['cat_tokens']) for sequence in sequences]


        input_texts_encodings_dict = self.tokenizer(input_texts, max_length= self.max_source_length, padding=True, truncation=True) 
        input_texts_input_ids = torch.as_tensor(input_texts_encodings_dict['input_ids'])
        input_texts_attention_mask = torch.as_tensor(input_texts_encodings_dict['attention_mask'])
        
        input_texts_input_ids = torch.cat([input_texts_input_ids.new(cat_ids)[:, None], input_texts_input_ids], dim=1)
        input_texts_attention_mask = torch.cat([input_texts_attention_mask.new([1] * len(input_texts_attention_mask))[:, None], input_texts_attention_mask], dim=1)

        with self.tokenizer.as_target_tokenizer():        
            generated_reductions_encodings_dict = self.tokenizer(generated_reductions, max_length=self.max_target_length, padding=True, truncation=True) 
        generated_reductions_input_ids = torch.as_tensor(generated_reductions_encodings_dict['input_ids'])
        generated_reductions_attention_mask = torch.as_tensor(generated_reductions_encodings_dict['attention_mask'])

        global_attention_mask = []
        if self.add_global_attention:
            for input_ids_instance in tqdm(input_texts_encodings_dict['input_ids']):
                curr_global_attention_mask = [0 for _ in range(len(input_ids_instance))]
                curr_global_attention_mask[0] = 1

                ids_with_global_attention = self.tokenizer.additional_special_tokens_ids

                if self.add_global_attention_on_highlights:
                    for input_id_idx, input_id in enumerate(input_ids_instance):
                        # Put attention on highlight tokens
                        if input_id in ids_with_global_attention: 
                            curr_global_attention_mask[input_id_idx] = 1

                global_attention_mask.append(curr_global_attention_mask)
        global_attention_mask = torch.as_tensor(global_attention_mask)
        global_attention_mask = torch.cat([global_attention_mask.new([1] * len(global_attention_mask))[:, None], global_attention_mask], dim=1)
        return input_texts_input_ids, input_texts_attention_mask, global_attention_mask, jsonl_ids, generated_reductions_input_ids, generated_reductions_attention_mask



class FixedController:
    def __init__(self, coef):
        self.value = coef

    def update(self, current, n_steps, lower_bound):
        pass


class AdaptiveController:
    def __init__(self, init_coef, target, horizon):
        self.value = init_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps, lower_bound):
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        if lower_bound:
            mult = 1 + proportional_error * n_steps / self.horizon
        else:
            mult = 1 - proportional_error * n_steps / self.horizon
        self.value *= mult


class ConditionTrainer:
    def __init__(self,
                 params: argparse.Namespace,
                 policy,
                 ref_policy,
                 data_pool,
                 score_model,
                 tree_tokens: List[str],
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: LambdaLR,
                 accelerator):

        self.params = params
        self.policy = policy
        self.ref_policy = ref_policy
        self.data_pool = data_pool
        self.score_model = score_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.report_to_wandb = params.report_to_wandb
        self.wandb_record_step = params.wandb_record_step
        self.writer = SummaryWriter()
        self.accelerator = accelerator

        if self.params.adaptive_kl:
            self.kl_ctl = AdaptiveController(self.params.kl_coef, self.params.target_kl, self.params.horizon)
        else:
            self.kl_ctl = FixedController(self.params.kl_coef)
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

        if self.params.adaptive_entropy:
            self.entropy_ctl = AdaptiveController(self.params.entropy_coef, self.params.target_entropy,
                                                  self.params.horizon)
        else:
            self.entropy_ctl = FixedController(self.params.entropy_coef)

        self.tree_tokens = tree_tokens
        self.best_cat = self.tree_tokens[0]
        self.best_cat_id = self.policy.tokenizer.convert_tokens_to_ids(self.best_cat)

        self.sample_dataloader, self.sampler = None, None
        self.seq_collator = SequenceCollator(tokenizer=policy.tokenizer, max_source_length=params.max_source_length, max_target_length=params.max_target_length, add_global_attention=params.add_global_attention, add_global_attention_on_highlights=params.add_global_attention_on_highlights)


    def sample(self, step):
        if step % self.params.sample_interval != 0:
            return
        log.info(f"[step {step}] Sampling ...")

        input_docs, input_jsonl_ids, output_reductions = [], [], []
        for i, batch in enumerate(tqdm(self.train_dataloader, total=len(self.train_dataloader), desc='Sampling from current policy')):
            input_ids, attention_mask, global_attention_mask, jsonl_ids = batch

            if step == 0:
                rollouts = self.ref_policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p, global_attention_mask=global_attention_mask)
                input_doc, output_reduction = rollouts['input_doc/text'], rollouts['output_reduction/text']
            else:
                input_ids, attention_mask, global_attention_mask = add_control_code(input_ids, attention_mask, global_attention_mask, self.best_cat_id)
                rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p, global_attention_mask=global_attention_mask)
                output_reduction = rollouts['output_reduction/text']
                input_doc = [self.policy.tokenizer.decode(p, clean_up_tokenization_spaces=True) for p in rollouts['input_doc/input_ids'][:, 1:]]
                # remove all the special tokens, except the "highlight_start" and "highlight_end" ones (which appeared in the original text)
                input_doc = custom_clean_special_tokens(input_doc=input_doc, model_name=self.policy.model_name, model_type=self.policy.model_type, special_tokens=self.policy.tokenizer.special_tokens_map.values())            

            input_docs.extend(input_doc)
            input_jsonl_ids.extend(jsonl_ids)
            output_reductions.extend(output_reduction)

        scores = self.score_model.get_reward(input_docs, input_jsonl_ids, output_reductions, f'step{step}', self.params.dataset_train, True)["final_reward"]

        process_ind = "0" if self.accelerator.device.index is None else self.accelerator.device.index # when not multi-GPU then self.accelerator.device.index is None
        with open(os.path.join(self.params.save_dir, f"data_pool_machine_{process_ind}.json"), 'w') as f:
                f.write(json.dumps({"input_docs":input_docs, "input_jsonl_ids":input_jsonl_ids, "output_reductions":output_reductions, "scores":[float(scr) for scr in scores]}, indent=None)) # in scores need to convert cast to float because when score=0 or score=1 then it is of type int64 which is not JSON serializable


        self.accelerator.wait_for_everyone()   
        input_docs_full, input_jsonl_ids_full, output_reductions_full, scores_full = [], [], [], []
        for proc_i in range(self.accelerator.num_processes):
            with open(os.path.join(self.params.save_dir, f"data_pool_machine_{proc_i}.json")) as f1:
                curr_data = json.load(f1)
                input_docs_full.extend([elem.strip() for elem in curr_data["input_docs"]])
                input_jsonl_ids_full.extend([elem for elem in curr_data["input_jsonl_ids"]])
                output_reductions_full.extend([elem.strip() for elem in curr_data["output_reductions"]])
                scores_full.extend([elem for elem in curr_data["scores"]])
        self.accelerator.wait_for_everyone() 
        if self.accelerator.is_main_process:
            [os.remove(os.path.join(self.params.save_dir, f"data_pool_machine_{ind}.json")) for ind in range(self.accelerator.num_processes) if os.path.exists(os.path.join(self.params.save_dir, f"data_pool_machine_{ind}.json"))]

        self.data_pool.add(inputs=input_docs_full, jsonl_ids=input_jsonl_ids_full, generated_reductions=output_reductions_full, scores=scores_full) 
        sample_dataset = SequenceDataset(data_pool=self.data_pool, model_name = self.params.init_model) 
        self.sample_dataloader = DataLoader(sample_dataset, batch_size=self.params.batch_size,
                                            shuffle=True, collate_fn=self.seq_collator)
        self.sample_dataloader = self.accelerator.prepare(self.sample_dataloader)
        self.sampler = iter(self.sample_dataloader)

    def step(self, step_num):
        step_started_at = time.time()
        self.sample(step=step_num)
        try:
            batch = next(self.sampler)
            assert len(batch[0]) == self.params.batch_size, 'insufficient batch'
        except (StopIteration, AssertionError):
            self.sampler = iter(self.sample_dataloader)
            batch = next(self.sampler)


        
        self.optimizer.zero_grad()
        ppo_loss, stats = self.loss(step_num, *batch)
        self.ref_policy.accelerator.backward(ppo_loss)
        if self.params.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.params.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        for metric in ['kl', 'entropy']:
            if step_num % self.wandb_record_step == 0 and self.report_to_wandb and self.accelerator.is_main_process:
                self.accelerator.print(f"[step {step_num}] saving to wandb the {metric} objective.")
                metric_wandb_name = "kl_divergence" if metric=="kl" else "entropy"
                self.accelerator.log({f"Objective/{metric_wandb_name}": stats[f'objective/{metric}']}, step=step_num)
        for metric in ['lm', 'kl', 'entropy', 'total']:
            if step_num % self.wandb_record_step == 0 and self.report_to_wandb and self.accelerator.is_main_process:
                if metric == "lm":
                    metric_wandb_name = "language_model"
                elif metric == "kl":
                    metric_wandb_name = "kl_divergence"
                else:
                    metric_wandb_name = metric
                self.accelerator.print(f"[step {step_num}] saving to wandb loss-{metric_wandb_name}.")
                self.accelerator.log({f"Train/loss-{metric_wandb_name}": stats[f'loss/{metric}']}, step=step_num)

        
        if step_num % self.wandb_record_step == 0 and self.report_to_wandb and self.accelerator.is_main_process:
            self.accelerator.print(f"[step {step_num}] saving to wandb the Params.")
            self.accelerator.log({f"Params/lr": self.optimizer.param_groups[0]['lr'], 
                      f'Params/kl_diveregnce_coef':self.kl_ctl.value, 
                      f'Params/entropy_coef': self.entropy_ctl.value}, step=step_num)


        self.kl_ctl.update(stats['objective/kl'], self.params.batch_size, True)
        self.entropy_ctl.update(stats['objective/entropy'], self.params.batch_size, False)

        step_time = time.time() - step_started_at
        eps_per_second = float(self.params.batch_size) / step_time
        self.accelerator.print(f"[step {step_num}] step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")
        eval_results = self.eval(step=step_num)
        
        scores_to_save = None
        if not eval_results is None:
            precision_eval, recall_eval, combined_reward_eval, \
                highlights_rouge1, highlights_rouge2, highlights_rougeL, highlights_rougeLsum, \
                unhighlights_rouge1, unhighlights_rouge2, unhighlights_rougeL, unhighlights_rougeLsum = eval_results
            scores_to_save = {"precision_eval":precision_eval, "recall_eval":recall_eval, "combined_reward_eval":combined_reward_eval,
                              "highlights_Rouge1":highlights_rouge1, "highlights_Rouge2":highlights_rouge2, "highlights_RougeL":highlights_rougeL, "highlights_RougeLsum":highlights_rougeLsum,
                              "unhighlights_Rouge1":unhighlights_rouge1, "unhighlights_Rouge2":unhighlights_rouge2, "unhighlights_RougeL":unhighlights_rougeL, "unhighlights_RougeLsum":unhighlights_rougeLsum}
            scores_to_save.update(stats)
        self.save(step=step_num, scores_to_save=scores_to_save)




    def loss(self, step, input_texts_input_ids, input_texts_attention_mask, input_texts_global_attention_mask, jsonl_ids, generated_reductions_input_ids, generated_reductions_attention_mask):
        outputs = self.policy.forward_pass(input_texts_input_ids=input_texts_input_ids, input_texts_attention_mask=input_texts_attention_mask, input_texts_global_attention_mask=input_texts_global_attention_mask, generated_reductions_input_ids=generated_reductions_input_ids, generated_reductions_attention_mask=generated_reductions_attention_mask)
        lm_loss, logprobs, entropy, logits, masks, output_text = outputs['generated_reduction/lm_loss'], outputs['generated_reduction/log_prob'], \
                                            outputs['generated_reduction/entropy'], outputs['generated_reduction/logits'], \
                                            outputs['generated_reduction/masks'], outputs['generated_reduction/output_text']

        logits = outputs['generated_reduction/logits'][:, :, :-len(self.tree_tokens)]
        masks = masks.to(self.policy.device)

        

        with torch.no_grad():
            ref_outputs = self.ref_policy.forward_pass(input_texts_input_ids[:, 1:], input_texts_attention_mask[:, 1:], input_texts_global_attention_mask[:, 1:],
                                                    generated_reductions_input_ids, generated_reductions_attention_mask)
            ref_logprobs, ref_logits, ref_output_text = ref_outputs['generated_reduction/log_prob'], ref_outputs['generated_reduction/logits'], ref_outputs['generated_reduction/output_text']

        kl = torch.sum(self.kl_loss(F.log_softmax(ref_logits, dim=-1), F.softmax(logits, dim=-1)), dim=-1)
        loss = reduce_mean(lm_loss + self.kl_ctl.value * kl - self.entropy_ctl.value * entropy, masks)

        data = {'logprobs': logprobs, 'ref_logprobs': ref_logprobs, 'masks': masks,
                'logits': logits, 'ref_logits': ref_logits,
                'lm_loss': reduce_mean(lm_loss, masks), 'kl_loss': reduce_mean(kl, masks),
                'entropy': reduce_mean(entropy, masks), 'total_loss': loss}
        stats = self.record_step_stats(data)
        return loss, stats

    def record_step_stats(self, data):
        masks = data['masks']
        kl = torch.sum(self.kl_loss(F.log_softmax(data['ref_logits'], dim=-1), F.softmax(data['logits'], dim=-1)), dim=-1)
        mean_kl = torch.mean(reduce_sum(kl, masks, axis=1))
        mean_entropy = torch.mean(reduce_sum(-data['logprobs'], masks, axis=1))
        stats = {
            'objective/kl': mean_kl.item(),
            'objective/entropy': mean_entropy.item(),
        }
        stats.update({
            'loss/total': data['total_loss'].item(),
            'loss/kl': data['kl_loss'].item(),
            'loss/lm': data['lm_loss'].item(),
            'loss/entropy': data['entropy'].item(),
        })

        return stats




    def print_samples(self, queries, responses, lm_loss, logprobs, ref_logprobs, masks, step):
        if step % self.params.log_interval != 0:
            return
            # Log samples
        for i in range(min(3, len(queries))):
            sample_kl = torch.sum((logprobs[i] - ref_logprobs[i]) * masks[i]).item()
            print(queries[i] + responses[i])
            print(f"  lm_loss = {lm_loss[i].item():+.2f}")
            print(f"  kl = {sample_kl:+.2f}")
            print(f"  total = {lm_loss[i].item() + self.params.kl_coef * sample_kl:+.2f}")

    def get_ckpt_score(self, ckpt_path):
        with open(os.path.join(ckpt_path, "scores.json")) as f:
            ckpt_score = json.load(f)[self.params.metric_for_best_model] 
        return float(ckpt_score)

    def save(self, step, scores_to_save):
        if step % self.params.save_interval != 0:
            return
        self.accelerator.save_state(f'{self.params.model_dir}/ckp_{step}')
        if self.accelerator.is_main_process:
            with open(os.path.join(f'{self.params.model_dir}/ckp_{step}', 'scores.json'), 'w') as f:
                json.dump({key:str(value) for key,value in scores_to_save.items()}, f, indent=0)
            with open(os.path.join(f'{self.params.model_dir}/ckp_{step}', 'data_pool.pkl'), 'wb') as outp:
                pickle.dump(deepcopy(self.data_pool), outp, pickle.HIGHEST_PROTOCOL)

            # keep only save_total_limit ckpts (the current one plus the save_total_limit-1 best ones from the previous ckpts)
            if not self.params.save_total_limit == "all":
                ckpt_scores = [(ckpt , self.get_ckpt_score(f'{self.params.model_dir}/{ckpt}')) for ckpt in os.listdir(self.params.model_dir) if ckpt != f"ckp_{step}"]
                ckpt_scores.sort(key=lambda a: a[1])
                ckpt_to_delete = [elem[0] for elem in ckpt_scores[:-(int(self.params.save_total_limit)-1)]]
                [shutil.rmtree(f'{self.params.model_dir}/{ckpt_d}', ignore_errors=True) for ckpt_d in ckpt_to_delete]
            self.accelerator.print(f"[step {step}] model checkpoint saved to {self.params.model_dir}/ckp_{step}")

    def eval(self, step):
        if step % self.params.eval_interval != 0:
            return
        self.accelerator.print(f"[step {step}] evaluating ...")

        generated_outputs, precision_scores, recall_scores, final_rewards = [], [], [], []

        highlights_rouge1_list, highlights_rouge2_list, highlights_rougeL_list, highlights_rougeLsum_list, unhighlights_rouge1_list, unhighlights_rouge2_list, unhighlights_rougeL_list, unhighlights_rougeLsum_list = [], [], [], [], [], [], [], []
    
        highlights_rouge1_precision_list, highlights_rouge2_precision_list, highlights_rougeL_precision_list, highlights_rougeLsum_precision_list, unhighlights_rouge1_precision_list, unhighlights_rouge2_precision_list, unhighlights_rougeL_precision_list, unhighlights_rougeLsum_precision_list = [], [], [], [], [], [], [], []
        highlights_rouge1_recall_list, highlights_rouge2_recall_list, highlights_rougeL_recall_list, highlights_rougeLsum_recall_list, unhighlights_rouge1_recall_list, unhighlights_rouge2_recall_list, unhighlights_rougeL_recall_list, unhighlights_rougeLsum_recall_list = [], [], [], [], [], [], [], []




        input_docs, jsonl_ids_list, output_reductions = [], [], []

        for i, (input_ids, attention_mask, global_attention_mask, jsonl_ids) in enumerate(tqdm(self.val_dataloader)):
            with torch.no_grad():
                input_ids, attention_mask, global_attention_mask = add_control_code(input_ids, attention_mask, global_attention_mask, self.best_cat_id)
                rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p, global_attention_mask=global_attention_mask)
                
                # the `1:` is so as to ignore the tree_token 
                forward_inputs = {'input_texts_input_ids': rollouts['input_doc/input_ids'][:, 1:],
                                  'input_texts_attention_mask': rollouts['input_doc/mask'][:, 1:],
                                  'input_texts_global_attention_mask': rollouts['input_doc/global_mask'][:, 1:],
                                  'generated_reductions_input_ids': rollouts['output_reduction/input_ids'],
                                  'generated_reductions_attention_mask': rollouts['output_reduction/mask']}
                ref_logprobs = self.ref_policy.forward_pass(**forward_inputs)['generated_reduction/log_prob']

                input_doc = [self.policy.tokenizer.decode(p, clean_up_tokenization_spaces=True) for p in rollouts['input_doc/input_ids'][:, 1:]] # the 1: is so as to ignore the tree_token 
                
                # remove all the special tokens, except the "highlight_start" and "highlight_end" ones (which appeared in the original text)
                is_t5_model = is_t5_model_def(self.policy.model_name, self.policy.model_type)
                special_tokens_constants = get_special_tokens_constants(is_t5_model)
                tokens_to_keep = [value for key,value in special_tokens_constants.items() if key in ['highlight_start', 'highlight_end']]
                all_special_tokens = sum([[value]  if type(value)==str else value for value in self.policy.tokenizer.special_tokens_map.values()], [])
                tokens_to_remove = [tkn for tkn in all_special_tokens if not tkn in tokens_to_keep]
                for tkn in tokens_to_remove:
                    input_doc = [elem.replace(tkn, "") for elem in input_doc]
                
                
                
                output_reduction = rollouts['output_reduction/text']
                score = self.score_model.get_reward(input_docs=input_doc, jsonl_ids=jsonl_ids, output_reductions=output_reduction, epoch=f'step{step}_eval{i}', jsonl_path=self.params.dataset_val, is_sampling=False)
                precision_score, recall_score, final_reward = score["precision_score"], score["recall_score"], score["final_reward"]
                precision_score = self.accelerator.gather_for_metrics((torch.Tensor(precision_score).to(self.accelerator.device)))
                recall_score = self.accelerator.gather_for_metrics((torch.Tensor(recall_score).to(self.accelerator.device)))
                final_reward = self.accelerator.gather_for_metrics((torch.Tensor(final_reward).to(self.accelerator.device)))
                
                # collect for wandb
                highlights_rouge1 = self.accelerator.gather_for_metrics((torch.Tensor(score['highlights_rouge1']).to(self.accelerator.device)))
                highlights_rouge2 = self.accelerator.gather_for_metrics((torch.Tensor(score['highlights_rouge2']).to(self.accelerator.device)))
                highlights_rougeL = self.accelerator.gather_for_metrics((torch.Tensor(score['highlights_rougeL']).to(self.accelerator.device)))
                highlights_rougeLsum = self.accelerator.gather_for_metrics((torch.Tensor(score['highlights_rougeLsum']).to(self.accelerator.device)))
                unhighlights_rouge1 = self.accelerator.gather_for_metrics((torch.Tensor(score['unhighlights_rouge1']).to(self.accelerator.device)))
                unhighlights_rouge2 = self.accelerator.gather_for_metrics((torch.Tensor(score['unhighlights_rouge2']).to(self.accelerator.device)))
                unhighlights_rougeL = self.accelerator.gather_for_metrics((torch.Tensor(score['unhighlights_rougeL']).to(self.accelerator.device)))
                unhighlights_rougeLsum = self.accelerator.gather_for_metrics((torch.Tensor(score['unhighlights_rougeLsum']).to(self.accelerator.device)))


                highlights_rouge1_precision = self.accelerator.gather_for_metrics((torch.Tensor(score['highlights_rouge1_precision']).to(self.accelerator.device)))
                highlights_rouge2_precision = self.accelerator.gather_for_metrics((torch.Tensor(score['highlights_rouge2_precision']).to(self.accelerator.device)))
                highlights_rougeL_precision = self.accelerator.gather_for_metrics((torch.Tensor(score['highlights_rougeL_precision']).to(self.accelerator.device)))
                highlights_rougeLsum_precision = self.accelerator.gather_for_metrics((torch.Tensor(score['highlights_rougeLsum_precision']).to(self.accelerator.device)))
                unhighlights_rouge1_precision = self.accelerator.gather_for_metrics((torch.Tensor(score['unhighlights_rouge1_precision']).to(self.accelerator.device)))
                unhighlights_rouge2_precision = self.accelerator.gather_for_metrics((torch.Tensor(score['unhighlights_rouge2_precision']).to(self.accelerator.device)))
                unhighlights_rougeL_precision = self.accelerator.gather_for_metrics((torch.Tensor(score['unhighlights_rougeL_precision']).to(self.accelerator.device)))
                unhighlights_rougeLsum_precision = self.accelerator.gather_for_metrics((torch.Tensor(score['unhighlights_rougeLsum_precision']).to(self.accelerator.device)))

            
                highlights_rouge1_recall = self.accelerator.gather_for_metrics((torch.Tensor(score['highlights_rouge1_recall']).to(self.accelerator.device)))
                highlights_rouge2_recall = self.accelerator.gather_for_metrics((torch.Tensor(score['highlights_rouge2_recall']).to(self.accelerator.device)))
                highlights_rougeL_recall = self.accelerator.gather_for_metrics((torch.Tensor(score['highlights_rougeL_recall']).to(self.accelerator.device)))
                highlights_rougeLsum_recall = self.accelerator.gather_for_metrics((torch.Tensor(score['highlights_rougeLsum_recall']).to(self.accelerator.device)))
                unhighlights_rouge1_recall = self.accelerator.gather_for_metrics((torch.Tensor(score['unhighlights_rouge1_recall']).to(self.accelerator.device)))
                unhighlights_rouge2_recall = self.accelerator.gather_for_metrics((torch.Tensor(score['unhighlights_rouge2_recall']).to(self.accelerator.device)))
                unhighlights_rougeL_recall = self.accelerator.gather_for_metrics((torch.Tensor(score['unhighlights_rougeL_recall']).to(self.accelerator.device)))
                unhighlights_rougeLsum_recall = self.accelerator.gather_for_metrics((torch.Tensor(score['unhighlights_rougeLsum_recall']).to(self.accelerator.device)))
                
                
                generated_outputs.extend(rollouts['output_reduction/text'])
                precision_scores.extend(precision_score)
                recall_scores.extend(recall_score)
                final_rewards.extend(final_reward)


                highlights_rouge1_list.extend(highlights_rouge1) 
                highlights_rouge2_list.extend(highlights_rouge2) 
                highlights_rougeL_list.extend(highlights_rougeL)  
                highlights_rougeLsum_list.extend(highlights_rougeLsum)  
                unhighlights_rouge1_list.extend(unhighlights_rouge1) 
                unhighlights_rouge2_list.extend(unhighlights_rouge2) 
                unhighlights_rougeL_list.extend(unhighlights_rougeL) 
                unhighlights_rougeLsum_list.extend(unhighlights_rougeLsum)

                highlights_rouge1_precision_list.extend(highlights_rouge1_precision)
                highlights_rouge2_precision_list.extend(highlights_rouge2_precision) 
                highlights_rougeL_precision_list.extend(highlights_rougeL_precision)
                highlights_rougeLsum_precision_list.extend(highlights_rougeLsum_precision)
                unhighlights_rouge1_precision_list.extend(unhighlights_rouge1_precision)
                unhighlights_rouge2_precision_list.extend(unhighlights_rouge2_precision) 
                unhighlights_rougeL_precision_list.extend(unhighlights_rougeL_precision) 
                unhighlights_rougeLsum_precision_list.extend(unhighlights_rougeLsum_precision)

                highlights_rouge1_recall_list.extend(highlights_rouge1_recall)
                highlights_rouge2_recall_list.extend(highlights_rouge2_recall) 
                highlights_rougeL_recall_list.extend(highlights_rougeL_recall)
                highlights_rougeLsum_recall_list.extend(highlights_rougeLsum_recall)
                unhighlights_rouge1_recall_list.extend(unhighlights_rouge1_recall)
                unhighlights_rouge2_recall_list.extend(unhighlights_rouge2_recall) 
                unhighlights_rougeL_recall_list.extend(unhighlights_rougeL_recall) 
                unhighlights_rougeLsum_recall_list.extend(unhighlights_rougeLsum_recall)




        precision_score_avg = np.mean([elem.cpu() for elem in precision_scores])
        recall_score_avg = np.mean([elem.cpu() for elem in recall_scores])
        final_reward_avg = np.mean([elem.cpu() for elem in final_rewards]) 



        highlights_rouge1_avg = np.mean([elem.cpu() for elem in highlights_rouge1_list])
        highlights_rouge2_avg = np.mean([elem.cpu() for elem in highlights_rouge2_list])
        highlights_rougeL_avg = np.mean([elem.cpu() for elem in highlights_rougeL_list])
        highlights_rougeLsum_avg = np.mean([elem.cpu() for elem in highlights_rougeLsum_list])
        unhighlights_rouge1_avg = np.mean([elem.cpu() for elem in unhighlights_rouge1_list])
        unhighlights_rouge2_avg = np.mean([elem.cpu() for elem in unhighlights_rouge2_list])
        unhighlights_rougeL_avg = np.mean([elem.cpu() for elem in unhighlights_rougeL_list])
        unhighlights_rougeLsum_avg = np.mean([elem.cpu() for elem in unhighlights_rougeLsum_list])


        highlights_rouge1_precision_avg = np.mean([elem.cpu() for elem in highlights_rouge1_precision_list])
        highlights_rouge2_precision_avg = np.mean([elem.cpu() for elem in highlights_rouge2_precision_list])
        highlights_rougeL_precision_avg = np.mean([elem.cpu() for elem in highlights_rougeL_precision_list])
        highlights_rougeLsum_precision_avg = np.mean([elem.cpu() for elem in highlights_rougeLsum_precision_list])
        unhighlights_rouge1_precision_avg = np.mean([elem.cpu() for elem in unhighlights_rouge1_precision_list])
        unhighlights_rouge2_precision_avg = np.mean([elem.cpu() for elem in unhighlights_rouge2_precision_list])
        unhighlights_rougeL_precision_avg = np.mean([elem.cpu() for elem in unhighlights_rougeL_precision_list])
        unhighlights_rougeLsum_precision_avg = np.mean([elem.cpu() for elem in unhighlights_rougeLsum_precision_list])

        highlights_rouge1_recall_avg = np.mean([elem.cpu() for elem in highlights_rouge1_recall_list])
        highlights_rouge2_recall_avg = np.mean([elem.cpu() for elem in highlights_rouge2_recall_list])
        highlights_rougeL_recall_avg = np.mean([elem.cpu() for elem in highlights_rougeL_recall_list])
        highlights_rougeLsum_recall_avg = np.mean([elem.cpu() for elem in highlights_rougeLsum_recall_list])
        unhighlights_rouge1_recall_avg = np.mean([elem.cpu() for elem in unhighlights_rouge1_recall_list])
        unhighlights_rouge2_recall_avg = np.mean([elem.cpu() for elem in unhighlights_rouge2_recall_list])
        unhighlights_rougeL_recall_avg = np.mean([elem.cpu() for elem in unhighlights_rougeL_recall_list])
        unhighlights_rougeLsum_recall_avg = np.mean([elem.cpu() for elem in unhighlights_rougeLsum_recall_list])



        if self.accelerator.is_main_process:
            self.accelerator.print(f"  precision (to highlights) = {round(precision_score_avg, 2)}")
            self.accelerator.print(f"  recall (of highlights) = {round(recall_score_avg, 2)}")
            self.accelerator.print(f"  reward = {round(final_reward_avg, 2)}")

            self.accelerator.print(f"  Rouge1 (highlights) = {round(highlights_rouge1_avg, 2)}")
            self.accelerator.print(f"  Rouge2 (highlights) = {round(highlights_rouge2_avg, 2)}")
            self.accelerator.print(f"  RougeL (highlights) = {round(highlights_rougeL_avg, 2)}")
            self.accelerator.print(f"  RougeLsum (highlights) = {round(highlights_rougeLsum_avg, 2)}")
            self.accelerator.print(f"  Rouge1 (unhighlights) = {round(unhighlights_rouge1_avg, 2)}")
            self.accelerator.print(f"  Rouge2 (unhighlights) = {round(unhighlights_rouge2_avg, 2)}")
            self.accelerator.print(f"  RougeL (unhighlights) = {round(unhighlights_rougeL_avg, 2)}")
            self.accelerator.print(f"  RougeLsum (unhighlights) = {round(unhighlights_rougeLsum_avg, 2)}")


            self.accelerator.print(f"  Rouge1 (precision) (highlights) = {round(highlights_rouge1_precision_avg, 2)}")
            self.accelerator.print(f"  Rouge2 (precision) (highlights) = {round(highlights_rouge2_precision_avg, 2)}")
            self.accelerator.print(f"  RougeL (precision) (highlights) = {round(highlights_rougeL_precision_avg, 2)}")
            self.accelerator.print(f"  RougeLsum (precision) (highlights) = {round(highlights_rougeLsum_precision_avg, 2)}")
            self.accelerator.print(f"  Rouge1 (precision) (unhighlights) = {round(unhighlights_rouge1_precision_avg, 2)}")
            self.accelerator.print(f"  Rouge2 (precision) (unhighlights) = {round(unhighlights_rouge2_precision_avg, 2)}")
            self.accelerator.print(f"  RougeL (precision) (unhighlights) = {round(unhighlights_rougeL_precision_avg, 2)}")
            self.accelerator.print(f"  RougeLsum (precision) (unhighlights) = {round(unhighlights_rougeLsum_precision_avg, 2)}")

            self.accelerator.print(f"  Rouge1 (recall) (highlights) = {round(highlights_rouge1_recall_avg, 2)}")
            self.accelerator.print(f"  Rouge2 (recall) (highlights) = {round(highlights_rouge2_recall_avg, 2)}")
            self.accelerator.print(f"  RougeL (recall) (highlights) = {round(highlights_rougeL_recall_avg, 2)}")
            self.accelerator.print(f"  RougeLsum (recall) (highlights) = {round(highlights_rougeLsum_recall_avg, 2)}")
            self.accelerator.print(f"  Rouge1 (recall) (unhighlights) = {round(unhighlights_rouge1_recall_avg, 2)}")
            self.accelerator.print(f"  Rouge2 (recall) (unhighlights) = {round(unhighlights_rouge2_recall_avg, 2)}")
            self.accelerator.print(f"  RougeL (recall) (unhighlights) = {round(unhighlights_rougeL_recall_avg, 2)}")
            self.accelerator.print(f"  RougeLsum (recall) (unhighlights) = {round(unhighlights_rougeLsum_recall_avg, 2)}")

            if self.report_to_wandb:
                self.accelerator.print(f"[step {step}] saving to wandb the Eval results.")
                self.accelerator.log({"Eval/precision_to_highlights": precision_score_avg, 
                            f'Eval/recall_of_highlights': recall_score_avg, 
                            f'Eval/reward': final_reward_avg,
                            f'Eval/highlights_Rouge1': highlights_rouge1_avg,
                            f'Eval/highlights_Rouge2': highlights_rouge2_avg,
                            f'Eval/highlights_RougeL': highlights_rougeL_avg,
                            f'Eval/highlights_RougeLsum': highlights_rougeLsum_avg,
                            f'Eval/unhighlights_Rouge1': unhighlights_rouge1_avg,
                            f'Eval/unhighlights_Rouge2': unhighlights_rouge2_avg,
                            f'Eval/unhighlights_RougeL': unhighlights_rougeL_avg,
                            f'Eval/unhighlights_RougeLsum': unhighlights_rougeLsum_avg,
                            f'Eval/highlights_Rouge1_precision': highlights_rouge1_precision_avg,
                            f'Eval/highlights_Rouge2_precision': highlights_rouge2_precision_avg,
                            f'Eval/highlights_RougeL_precision': highlights_rougeL_precision_avg,
                            f'Eval/highlights_RougeLsum_precision': highlights_rougeLsum_precision_avg,
                            f'Eval/unhighlights_Rouge1_precision': unhighlights_rouge1_precision_avg,
                            f'Eval/unhighlights_Rouge2_precision': unhighlights_rouge2_precision_avg,
                            f'Eval/unhighlights_RougeL_precision': unhighlights_rougeL_precision_avg,
                            f'Eval/unhighlights_RougeLsum_precision': unhighlights_rougeLsum_precision_avg,
                            f'Eval/highlights_Rouge1_recall': highlights_rouge1_recall_avg,
                            f'Eval/highlights_Rouge2_recall': highlights_rouge2_recall_avg,
                            f'Eval/highlights_RougeL_recall': highlights_rougeL_recall_avg,
                            f'Eval/highlights_RougeLsum_recall': highlights_rougeLsum_recall_avg,
                            f'Eval/unhighlights_Rouge1_recall': unhighlights_rouge1_recall_avg,
                            f'Eval/unhighlights_Rouge2_recall': unhighlights_rouge2_recall_avg,
                            f'Eval/unhighlights_RougeL_recall': unhighlights_rougeL_recall_avg,
                            f'Eval/unhighlights_RougeLsum_recall': unhighlights_rougeLsum_recall_avg}, step=step)

        return precision_score_avg, recall_score_avg, final_reward_avg, highlights_rouge1_avg, highlights_rouge2_avg, highlights_rougeL_avg, highlights_rougeLsum_avg, unhighlights_rouge1_avg, unhighlights_rouge2_avg, unhighlights_rougeL_avg, unhighlights_rougeLsum_avg


     

 
def main():
    args = get_args()

    # if continuing train from most recent checkpoint due to crash - then load the args from the previous run
    resume_from_checkpoint = args.resume_from_checkpoint
    if not resume_from_checkpoint is None and list(os.scandir(os.path.join(args.resume_from_checkpoint, "model"))) != [] and args.load_prev_args:
        with open(os.path.join(resume_from_checkpoint, "args.json")) as f1:
            old_args = json.load(f1)
            old_args["resume_from_checkpoint"] = resume_from_checkpoint
            # count number of times we resumed run for wandb naming
            if "resume_count" in old_args.keys():
                old_args["resume_count"] = old_args["resume_count"] + 1
            else:
                old_args["resume_count"] = 1
            old_args = argparse.Namespace(**old_args)
        args = old_args

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    num_gpus = torch.cuda.device_count()
    log.info(f'Detect {num_gpus} GPUS')

    kwargs = {"num_processes":num_gpus}
    accelerator = Accelerator(cpu=False, mixed_precision='fp16', log_with="wandb")
    device = accelerator.device

    time = datetime.now()
    date_time = time.strftime("%m-%d-%Y_%H:%M:%S")

    if args.report_to_wandb:
        wandb_run_name = args.wandb_run_name if not args.wandb_run_name is None else f"controlled_text_reduction_RL/{date_time}"
        if "/" in wandb_run_name:
            [project_name, run_name] = wandb_run_name.split("/")
        else:
            project_name = "controlled_text_reduction_RL"
            run_name = wandb_run_name
        resume_wandb_suffix = f"_resumed_{str(args.resume_count)}" if hasattr(args, "resume_count") else ""
        accelerator.init_trackers(project_name, config={}, init_kwargs={"wandb":{"name":f"{run_name}{resume_wandb_suffix}"}})
    if not args.resume_from_checkpoint is None:
        if list(os.scandir(os.path.join(args.resume_from_checkpoint, "model"))) == []:
            args.save_dir = os.path.join(args.output_dir, date_time) 
        else:
            args.save_dir = args.resume_from_checkpoint
    else:
        args.save_dir = os.path.join(args.output_dir, date_time)


    if not hasattr(args, "model_dir"): # in the case previous args weren't loaded
        args.reward_dir = os.path.join(args.save_dir, 'reward')
        args.model_dir = os.path.join(args.save_dir, 'model')
        args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
    for d in [args.output_dir, args.save_dir, args.reward_dir, args.model_dir, args.tensorboard_dir]:
            ensure_dir(d)
    log.info(f'Write to output directory: {args.save_dir}')

    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    tree_tokens = [' _TREE_TOKEN_{}'.format(str(idx).zfill(5)) for idx in range(args.n_extra_tokens)] + \
                  [' _TREE_TOKEN_ZERO_COMMENTS'] # tokens of the "categories" --> i.e., the quantization quantiles

    last_checkpoint=None

    # making sure that when model is saved, there are train and eval losses and rewards
    if args.save_interval % args.eval_interval != 0 or args.save_interval % args.wandb_record_step != 0:
        raise AttributeError(
                f"Make sure eval_interval and wandb_record_step are a round multiple of save_interval. Currently save_interval={args.save_interval}, eval_interval={args.eval_interval} and wandb_record_step={args.wandb_record_step}"
            )


    accelerator.print(f'Initializing models ...')
    ref_policy = Policy(model_name=args.init_model, temperature=args.temperature, device=device, args=args, logger=log, last_checkpoint=None, accelerator=accelerator)
    policy = Policy(model_name=args.ref_model, temperature=args.temperature, device=device,
                        reward_cond=True, tree_tokens=tree_tokens, args=args, logger=log, last_checkpoint=last_checkpoint, accelerator=accelerator)
    rouge_metric_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    reward = Reward(save_path=args.reward_dir, batch_size=args.batch_size, R_P_rewards_ratio=args.R_P_rewards_ratio, R_P_rewards_alternate=args.R_P_rewards_alternate, R_P_rewards_alternate_ratio=args.R_P_rewards_alternate_ratio, reward_type=args.reward_type, rouge_metric_scorer=rouge_metric_scorer, sample_interval=args.sample_interval, P_reward_type = args.P_reward_type, R_reward_type = args.R_reward_type)
    data_pool = DataPool(tree_tokens=tree_tokens, n_extra_tokens=args.n_extra_tokens, model_name=args.init_model, data_path = args.dataset_train)
    accelerator.print(f'Initialization done!')

    is_t5_model = is_t5_model_def(policy.model_name, policy.model_type)
    if args.source_prefix is None and is_t5_model:
        log.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    prompt_collator = PromptCollator(tokenizer=policy.tokenizer, args=args)
    train_dataset = PromptDataset(path=args.dataset_train, model_name=args.init_model)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=prompt_collator)
    accelerator.print(f'Load train set with {len(train_dataset)} examples')

    val_dataset = PromptDataset(path=args.dataset_val, model_name=args.init_model)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=prompt_collator)
    accelerator.print(f'Load val set with {len(val_dataset)} examples')

    # set up optimizer and scheduler
    optimizer = Adam(policy.model.parameters(), lr=args.lr, eps=args.eps)
    args.total_steps = ceil_div(args.total_episodes, args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.total_steps)

    scheduler, optimizer, train_dataloader, val_dataloader = accelerator.prepare(scheduler, optimizer, train_dataloader, val_dataloader)

    resume_step = -1
    if args.resume_from_checkpoint is not None:
        if [elem for elem in os.scandir(args.resume_from_checkpoint) if elem.name=="pytorch_model.bin"]:
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = args.resume_from_checkpoint
            with open(os.path.join(f'{path}', 'data_pool.pkl'), 'rb') as inp:
                data_pool = pickle.load(inp)
        elif list(os.scandir(os.path.join(args.resume_from_checkpoint, "model"))) != []:
            # Get the most recent checkpoint
            dirs = [os.path.join(args.resume_from_checkpoint, "model", f.name) for f in os.scandir(os.path.join(args.resume_from_checkpoint, "model")) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            accelerator.print(f"Resumed from checkpoint: {path}")
            accelerator.load_state(path)
            with open(os.path.join(f'{path}', 'data_pool.pkl'), 'rb') as inp:
                data_pool = pickle.load(inp)
        else:
            accelerator.print(f"No checkpoint was found in path {args.model_dir}. Training from scratch init_model `{args.init_model}`")

        # Extract `step_{i}`
        resume_step = int(os.path.basename(path).split("_")[1])




    
    
    
    trainer = ConditionTrainer(params=args, policy=policy, ref_policy=ref_policy, data_pool=data_pool,
                               score_model=reward, tree_tokens=tree_tokens,
                               train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                               optimizer=optimizer, scheduler=scheduler, accelerator=accelerator)

    if args.resume_from_checkpoint and resume_step > -1:
        sample_dataset = SequenceDataset(data_pool=data_pool, model_name = args.init_model) 
        trainer.sample_dataloader = DataLoader(sample_dataset, batch_size=args.batch_size,
                                            shuffle=True, collate_fn=trainer.seq_collator)
        trainer.sample_dataloader = accelerator.prepare(trainer.sample_dataloader)
        trainer.sampler = iter(trainer.sample_dataloader)

    for step_num in range(args.total_steps):
        if step_num <= resume_step:
            continue
        try:
            trainer.step(step_num)
        except RuntimeError:
            torch.cuda.empty_cache()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            continue
        # except:
        #     accelerator.end_training()
        #     raise
        # accelerator.end_training()


if __name__ == "__main__":
    main()
