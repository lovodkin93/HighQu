import torch
from pathlib import Path
import os
import numpy as np
import logging
import json

from statistics import mean
import pandas as pd
from tqdm import tqdm
from policy import Policy
from reward import Reward
from data_pool import DataPool
from main import PromptCollator, PromptDataset
from torch.utils.data import DataLoader
from main import PromptDataset, PromptCollator, ConditionTrainer
from utils.utils import load_jsonl, ensure_dir, reduce_sum, is_t5_model_def, ceil_div, get_jsonl_line_data, add_control_code
from utils.constants import ROUGE1, ROUGE2, ROUGEL, ROUGELSUM, all_rouges

import argparse
from accelerate import Accelerator, DistributedType
from datetime import datetime
import time
from rouge_score import rouge_scorer
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
import pickle
from preprocess_utils.preprocessor import get_special_tokens_constants
import pandas as pd


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


scores_to_print = ["precision_reward", "recall_reward", "combined_reward", 
                "highlights_rouge1_F1", "highlights_rouge2_F1", "highlights_rougeL_F1", "highlights_rougeLsum_F1",
                 "unhighlights_rouge1_F1", "unhighlights_rouge2_F1", "unhighlights_rougeL_F1", "unhighlights_rougeLsum_F1", 
                 "gold_rouge1_F1", "gold_rouge2_F1", "gold_rougeL_F1", "gold_rougeLsum_F1"]


def get_output_dirs(dataset_test, dataset_dev, saved_model_path):
    saved_model_path = saved_model_path.split("/")[saved_model_path.split("/").index("outputs")+1:] # take all the path after "outdir" (in case given full path)
    saved_model_path.remove("model") # remove the "model" subdir (to make the output shorter)
    output_subdirs = os.path.join(*saved_model_path)

    test_outdir = os.path.join("outputs", "eval", "test", output_subdirs)
    dev_outdir = os.path.join("outputs", "eval", "dev", output_subdirs)
    test_outdir_tmp = os.path.join(test_outdir, "tmp") # for temporary saved files that would be deleted later
    dev_outdir_tmp = os.path.join(dev_outdir, "tmp") # for temporary saved files that would be deleted later

    test_outdir_ref = os.path.join(test_outdir, "ref_model") # for temporary saved files that would be deleted later
    dev_outdir_ref = os.path.join(dev_outdir, "ref_model") # for temporary saved files that would be deleted later

    if not dataset_test == None:
        ensure_dir(test_outdir)
        ensure_dir(test_outdir_tmp)
        ensure_dir(test_outdir_ref)

    if not dataset_dev == None:
        ensure_dir(dev_outdir)
        ensure_dir(dev_outdir_tmp)
        ensure_dir(dev_outdir_ref)

    return test_outdir, dev_outdir, test_outdir_tmp, dev_outdir_tmp, test_outdir_ref, dev_outdir_ref

def get_all_rouges_specific(input_jsonl_ids_full, output_reductions_full, dataset_path, reference_type):
    rouge_metric_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    all_scores = []
    for i,output_reduction in enumerate(output_reductions_full):
        ref_text = get_jsonl_line_data(path=dataset_path, jsonl_id=input_jsonl_ids_full[i], attrib=reference_type) 
        curr_scores = rouge_metric_scorer.score(output_reduction, ref_text)
        all_scores.append(curr_scores)
    return all_scores

def get_all_rouges(input_jsonl_ids_full, output_reductions_full, dataset_path):
    h_concats_rouges = get_all_rouges_specific(input_jsonl_ids_full, output_reductions_full, dataset_path, "highlights_concatenation")
    uh_concats_rouges = get_all_rouges_specific(input_jsonl_ids_full, output_reductions_full, dataset_path, "unhighlights_concatenation")
    gold_summary_rouges = get_all_rouges_specific(input_jsonl_ids_full, output_reductions_full, dataset_path, "gold_summary")
    return h_concats_rouges, uh_concats_rouges, gold_summary_rouges

def get_rouge_avg(all_rouges):
    precision_rouge = {f"{key}_precision":round(np.mean([elem[key].precision for elem in all_rouges]), 3) for key in all_rouges[0].keys()}
    recall_rouge = {f"{key}_recall":round(np.mean([elem[key].recall for elem in all_rouges]), 3) for key in all_rouges[0].keys()}
    fmeasure_rouge = {f"{key}_F1":round(np.mean([elem[key].fmeasure for elem in all_rouges]), 3) for key in all_rouges[0].keys()}
    total_dict = dict()
    for key in all_rouges[0].keys():
         total_dict[f"{key}_F1"] = fmeasure_rouge[f"{key}_F1"]
         total_dict[f"{key}_precision"] = precision_rouge[f"{key}_precision"]
         total_dict[f"{key}_recall"] = recall_rouge[f"{key}_recall"]
    return total_dict


def get_csv_dict(input_docs_full, input_jsonl_ids_full, doc_ids_full, summary_ids_full, output_reductions_full, ungathered_precision_scores_full, ungathered_recall_scores_full, ungathered_final_rewards_full, h_concats_rouges, uh_concats_rouges, gold_summary_rouges):
    csv_dict = {"input" : input_docs_full, 
                "input_jsonl_ids" : input_jsonl_ids_full,
                "generated_text" : output_reductions_full,
                "precision_reward_highlights" : ungathered_precision_scores_full,
                "recall_reward_highlights" : ungathered_recall_scores_full,
                "combined_reward_highlights" : ungathered_final_rewards_full,
                "highlights_rouge1": [round(elem[ROUGE1].fmeasure, 3) for elem in h_concats_rouges],
                "highlights_rouge2": [round(elem[ROUGE2].fmeasure, 3) for elem in h_concats_rouges],
                "highlights_rougeL": [round(elem[ROUGEL].fmeasure, 3) for elem in h_concats_rouges],
                "highlights_rougeLSum": [round(elem[ROUGELSUM].fmeasure, 3) for elem in h_concats_rouges],
                "unhighlights_rouge1": [round(elem[ROUGE1].fmeasure, 3) for elem in uh_concats_rouges],
                "unhighlights_rouge2": [round(elem[ROUGE2].fmeasure, 3) for elem in uh_concats_rouges],
                "unhighlights_rougeL": [round(elem[ROUGEL].fmeasure, 3) for elem in uh_concats_rouges],
                "unhighlights_rougeLSum": [round(elem[ROUGELSUM].fmeasure, 3) for elem in uh_concats_rouges],
                "gold_rouge1": [round(elem[ROUGE1].fmeasure, 3) for elem in gold_summary_rouges],
                "gold_rouge2": [round(elem[ROUGE2].fmeasure, 3) for elem in gold_summary_rouges],
                "gold_rougeL": [round(elem[ROUGEL].fmeasure, 3) for elem in gold_summary_rouges],
                "gold_rougeLSum": [round(elem[ROUGELSUM].fmeasure, 3) for elem in gold_summary_rouges],
                "highlights_rouge1_recall": [round(elem[ROUGE1].recall, 3) for elem in h_concats_rouges],
                "highlights_rouge2_recall": [round(elem[ROUGE2].recall, 3) for elem in h_concats_rouges],
                "highlights_rougeL_recall": [round(elem[ROUGEL].recall, 3) for elem in h_concats_rouges],
                "highlights_rougeLSum_recall": [round(elem[ROUGELSUM].recall, 3) for elem in h_concats_rouges],
                "unhighlights_rouge1_recall": [round(elem[ROUGE1].recall, 3) for elem in uh_concats_rouges],
                "unhighlights_rouge2_recall": [round(elem[ROUGE2].recall, 3) for elem in uh_concats_rouges],
                "unhighlights_rougeL_recall": [round(elem[ROUGEL].recall, 3) for elem in uh_concats_rouges],
                "unhighlights_rougeLSum_recall": [round(elem[ROUGELSUM].recall, 3) for elem in uh_concats_rouges],
                "gold_rouge1_recall": [round(elem[ROUGE1].recall, 3) for elem in gold_summary_rouges],
                "gold_rouge2_recall": [round(elem[ROUGE2].recall, 3) for elem in gold_summary_rouges],
                "gold_rougeL_recall": [round(elem[ROUGEL].recall, 3) for elem in gold_summary_rouges],
                "gold_rougeLSum_recall": [round(elem[ROUGELSUM].recall, 3) for elem in gold_summary_rouges],
                "highlights_rouge1_precision": [round(elem[ROUGE1].precision, 3) for elem in h_concats_rouges],
                "highlights_rouge2_precision": [round(elem[ROUGE2].precision, 3) for elem in h_concats_rouges],
                "highlights_rougeL_precision": [round(elem[ROUGEL].precision, 3) for elem in h_concats_rouges],
                "highlights_rougeLSum_precision": [round(elem[ROUGELSUM].precision, 3) for elem in h_concats_rouges],
                "unhighlights_rouge1_precision": [round(elem[ROUGE1].precision, 3) for elem in uh_concats_rouges],
                "unhighlights_rouge2_precision": [round(elem[ROUGE2].precision, 3) for elem in uh_concats_rouges],
                "unhighlights_rougeL_precision": [round(elem[ROUGEL].precision, 3) for elem in uh_concats_rouges],
                "unhighlights_rougeLSum_precision": [round(elem[ROUGELSUM].precision, 3) for elem in uh_concats_rouges],
                "gold_rouge1_precision": [round(elem[ROUGE1].precision, 3) for elem in gold_summary_rouges],
                "gold_rouge2_precision": [round(elem[ROUGE2].precision, 3) for elem in gold_summary_rouges],
                "gold_rougeL_precision": [round(elem[ROUGEL].precision, 3) for elem in gold_summary_rouges],
                "gold_rougeLSum_precision": [round(elem[ROUGELSUM].precision, 3) for elem in gold_summary_rouges]
                }
    if doc_ids_full != []:
        csv_dict["doc_ids"] = doc_ids_full 
    if summary_ids_full != []:
        csv_dict["summary_ids"] = summary_ids_full 
    return csv_dict      

def get_full_scores_dict(precision_score_avg, recall_score_avg, final_reward_avg, h_concats_rouges_avg, uh_concats_rouges_avg, gold_summary_rouges_avg):
    total_scores_dict = {"precision_reward":str(round(precision_score_avg, 3)),
                        "recall_reward":str(round(recall_score_avg, 3)),
                        "combined_reward":str(round(final_reward_avg, 3))}
    
    total_scores_dict.update({f"highlights_{key}":str(value) for key,value in h_concats_rouges_avg.items()})
    total_scores_dict.update({f"unhighlights_{key}":str(value) for key,value in uh_concats_rouges_avg.items()})
    total_scores_dict.update({f"gold_{key}":str(value) for key,value in gold_summary_rouges_avg.items()})
    return total_scores_dict


def get_samples(dataloader, policy, args, score_model, dataset_path, accelerator, best_cat_id, data_split, tmp_dir):
    input_texts, input_jsonl_ids, generated_outputs, ungathered_precision_scores, ungathered_recall_scores, ungathered_final_rewards = [], [], [], [], [], []
    doc_ids, summary_ids = [], []
    precision_scores, recall_scores, final_rewards = [], [], [] # gathered for the totel score (the ungathered - to align with the generated outputs and save later on in a csv file)
    policy.model.eval()
    for i, (input_ids, attention_mask, global_attention_mask, jsonl_ids) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            if not best_cat_id is None: # best_cat_id is None when evaluating the performance of the reference (pre-finetuning) model
                input_ids, attention_mask, global_attention_mask = add_control_code(input_ids, attention_mask, global_attention_mask, best_cat_id) 
            rollouts = policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=args.top_p, global_attention_mask=global_attention_mask)
            input_doc = [policy.tokenizer.decode(p, clean_up_tokenization_spaces=True) for p in rollouts['input_doc/input_ids'][:, 1:]] # the 1: is so as to ignore the tree_token 
            # remove all the special tokens, except the "highlight_start" and "highlight_end" ones (which appeared in the original text)
            is_t5_model = is_t5_model_def(policy.model_name, policy.model_type)
            special_tokens_constants = get_special_tokens_constants(is_t5_model)
            tokens_to_keep = [value for key,value in special_tokens_constants.items() if key in ['highlight_start', 'highlight_end']]
            all_special_tokens = sum([[value]  if type(value)==str else value for value in policy.tokenizer.special_tokens_map.values()], [])
            tokens_to_remove = [tkn for tkn in all_special_tokens if not tkn in tokens_to_keep]
            for tkn in tokens_to_remove:
                input_doc = [elem.replace(tkn, "") for elem in input_doc]
            output_reduction = rollouts['output_reduction/text']
            score = score_model.get_reward(input_docs=input_doc, jsonl_ids=jsonl_ids, output_reductions=output_reduction, epoch="", jsonl_path=dataset_path, is_sampling=False)
            precision_score, recall_score, final_reward = score["precision_score"], score["recall_score"], score["final_reward"]
            precision_score = accelerator.gather_for_metrics((torch.Tensor(precision_score).to(accelerator.device)))
            recall_score = accelerator.gather_for_metrics((torch.Tensor(recall_score).to(accelerator.device)))
            final_reward = accelerator.gather_for_metrics((torch.Tensor(final_reward).to(accelerator.device)))
            input_texts.extend(input_doc)
            input_jsonl_ids.extend(jsonl_ids)

            # doc_id and summary_id
            with open(dataset_path, 'r') as f1:
                jsonl_content = f1.readlines()
                curr_jsn_lines = [json.loads(jsonl_content[int(jsnl_i)]) for jsnl_i in jsonl_ids]
                if curr_jsn_lines != [] and "doc_id" in curr_jsn_lines[0].keys():
                    doc_ids.extend([curr_jsn_line["doc_id"] for curr_jsn_line in curr_jsn_lines])
                if curr_jsn_lines != [] and "summary_id" in curr_jsn_lines[0].keys():
                    summary_ids.extend([curr_jsn_line["summary_id"] for curr_jsn_line in curr_jsn_lines])


            generated_outputs.extend(rollouts['output_reduction/text'])
            ungathered_precision_scores.extend(score["precision_score"])
            ungathered_recall_scores.extend(score["recall_score"])
            ungathered_final_rewards.extend(score["final_reward"])
            # gathered - to remove duplicates and calculate the total scores
            precision_scores.extend(precision_score) 
            recall_scores.extend(recall_score)
            final_rewards.extend(final_reward)
    
    process_ind = "0" if accelerator.device.index is None else accelerator.device.index # when not multi-GPU then self.accelerator.device.index is None
    with open(os.path.join(tmp_dir, f"samples_generation_{process_ind}.json"), 'w') as f:
        f.write(json.dumps({"input_docs":input_texts, "input_jsonl_ids":input_jsonl_ids, "doc_ids":doc_ids, "summary_ids":summary_ids, "output_reductions":generated_outputs, "P_scores":ungathered_precision_scores, "R_scores":ungathered_recall_scores, "combined_rewards":ungathered_final_rewards}, indent=None))
    accelerator.wait_for_everyone()   
    input_docs_full, input_jsonl_ids_full, output_reductions_full, ungathered_precision_scores_full, ungathered_recall_scores_full, ungathered_final_rewards_full = [], [], [], [], [], []
    doc_ids_full, summary_ids_full = [], []
    for proc_i in range(accelerator.num_processes):
        with open(os.path.join(tmp_dir, f"samples_generation_{proc_i}.json")) as f1:
            curr_data = json.load(f1)
            input_docs_full.extend([elem.strip() for elem in curr_data["input_docs"]])
            input_jsonl_ids_full.extend([elem for elem in curr_data["input_jsonl_ids"]])
            doc_ids_full.extend([elem for elem in curr_data["doc_ids"]])
            summary_ids_full.extend([elem for elem in curr_data["summary_ids"]])
            output_reductions_full.extend([elem.strip() for elem in curr_data["output_reductions"]])
            ungathered_precision_scores_full.extend([elem for elem in curr_data["P_scores"]])
            ungathered_recall_scores_full.extend([elem for elem in curr_data["R_scores"]])
            ungathered_final_rewards_full.extend([elem for elem in curr_data["combined_rewards"]])

    accelerator.wait_for_everyone() 
    if accelerator.is_main_process:
        [os.remove(os.path.join(tmp_dir, f"samples_generation_{ind}.json")) for ind in range(accelerator.num_processes) if os.path.exists(os.path.join(tmp_dir, f"samples_generation_{ind}.json"))]
    
    # remove duplicates
    unique_inds = [input_jsonl_ids_full.index(elem) for elem in set(input_jsonl_ids_full)]
    input_docs_full = [elem for i,elem in enumerate(input_docs_full) if i in unique_inds]
    input_jsonl_ids_full = [elem for i,elem in enumerate(input_jsonl_ids_full) if i in unique_inds]
    doc_ids_full = [elem for i,elem in enumerate(doc_ids_full) if i in unique_inds] if doc_ids_full != [] else []
    summary_ids_full = [elem for i,elem in enumerate(summary_ids_full) if i in unique_inds] if summary_ids_full != [] else []
    output_reductions_full = [elem for i,elem in enumerate(output_reductions_full) if i in unique_inds]
    ungathered_precision_scores_full = [elem for i,elem in enumerate(ungathered_precision_scores_full) if i in unique_inds]
    ungathered_recall_scores_full = [elem for i,elem in enumerate(ungathered_recall_scores_full) if i in unique_inds]
    ungathered_final_rewards_full = [elem for i,elem in enumerate(ungathered_final_rewards_full) if i in unique_inds]

    # calculate total scores
    precision_score_avg = np.mean([elem.cpu() for elem in precision_scores])
    recall_score_avg = np.mean([elem.cpu() for elem in recall_scores])
    final_reward_avg = np.mean([elem.cpu() for elem in final_rewards]) 

    # get all the rouge scores (Rouge1, Rouge2, RougeL, RougeLSum) with the highlights and the unhighlights and gold summary
    h_concats_rouges, uh_concats_rouges, gold_summary_rouges = get_all_rouges(input_jsonl_ids_full, output_reductions_full, dataset_path)
    # get average of all the rouges    
    h_concats_rouges_avg, uh_concats_rouges_avg, gold_summary_rouges_avg = get_rouge_avg(h_concats_rouges), get_rouge_avg(uh_concats_rouges), get_rouge_avg(gold_summary_rouges),


    # get the full csv_file dictionary
    out_csv_dict = get_csv_dict(input_docs_full, input_jsonl_ids_full, doc_ids_full, summary_ids_full, output_reductions_full, ungathered_precision_scores_full, ungathered_recall_scores_full, ungathered_final_rewards_full, h_concats_rouges, uh_concats_rouges, gold_summary_rouges)

    # get full scores and rouges dict (to be saved in a json file)
    full_scores_dict = get_full_scores_dict(precision_score_avg, recall_score_avg, final_reward_avg, h_concats_rouges_avg, uh_concats_rouges_avg, gold_summary_rouges_avg)

    return out_csv_dict, full_scores_dict


def save_results(out_csv_dict, full_scores_dict, outdir):
    csv_path = os.path.join(outdir, "generated_outputs.csv")
    scores_json_path = os.path.join(outdir, "results.json")
    csv_df = pd.DataFrame(out_csv_dict)
    csv_df.to_csv(csv_path)

    with open(scores_json_path, 'w') as f:
            f.write(json.dumps(full_scores_dict, indent=None))


def print_results(full_scores_dict, accelerator, split):
    accelerator.print(f"\nresults for {split}:")
    [accelerator.print(f"{scr.replace('F1', '')}:{full_scores_dict[scr]}") for scr in scores_to_print]



def main(params):
    dataset_test = params.dataset_test
    dataset_dev = params.dataset_dev
    saved_model_path = params.saved_model_path
    
    if not os.path.exists(saved_model_path):
        raise FileNotFoundError(f"No such directory {saved_model_path}")
    if dataset_test == None and dataset_dev == None:
        raise AttributeError("At least one of --dataset-test or --dataset-dev need to be provided")
    
    train_args_path = os.path.join(saved_model_path, "..", "..", "args.json")

    if not os.path.exists(train_args_path):
        raise FileNotFoundError (f'no args.json file was found (should be under {train_args_path})')

    test_outdir, dev_outdir, test_outdir_tmp, dev_outdir_tmp, test_outdir_ref, dev_outdir_ref = get_output_dirs(dataset_test, dataset_dev, saved_model_path)
    with open(train_args_path) as f1:
                args = json.load(f1) # args is of the original training, whereas params is for the sampling
                if not "R_P_rewards_alternate_ratio" in args.keys(): # added this argument after already some models were finetuned so setting the default.
                    args["R_P_rewards_alternate_ratio"] = "[1,1]"
                args = argparse.Namespace(**args)

    accelerator = Accelerator(cpu=False, mixed_precision='fp16')
    device = accelerator.device
    time = datetime.now()
    date_time = time.strftime("%m-%d-%Y_%H:%M:%S")
    accelerator.print(f"evaluating model in ckpt: {saved_model_path}")

    if dataset_test != None:
        accelerator.print(f"save test results to {test_outdir}")
    if dataset_dev != None:
        accelerator.print(f"save dev results to {dev_outdir}")



    tree_tokens = [' _TREE_TOKEN_{}'.format(str(idx).zfill(5)) for idx in range(args.n_extra_tokens)] + \
                [' _TREE_TOKEN_ZERO_COMMENTS'] # tokens of the "categories" --> i.e., the quantization quantiles
    ref_policy = Policy(model_name=args.init_model, temperature=args.temperature, device=device, args=args, logger=log, last_checkpoint=None, accelerator=accelerator)
    policy = Policy(model_name=args.ref_model, temperature=args.temperature, device=device,
                        reward_cond=True, tree_tokens=tree_tokens, args=args, logger=log, last_checkpoint=None, accelerator=accelerator)
    rouge_metric_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True) 
    reward = Reward(save_path=args.reward_dir, batch_size=args.batch_size, R_P_rewards_ratio=args.R_P_rewards_ratio, R_P_rewards_alternate=args.R_P_rewards_alternate, R_P_rewards_alternate_ratio=args.R_P_rewards_alternate_ratio, reward_type=args.reward_type, rouge_metric_scorer=rouge_metric_scorer, sample_interval=args.sample_interval, P_reward_type=args.P_reward_type, R_reward_type=args.R_reward_type)
    data_pool = DataPool(tree_tokens=tree_tokens, n_extra_tokens=args.n_extra_tokens, model_name=args.init_model, data_path = args.dataset_train)
    is_t5_model = is_t5_model_def(policy.model_name, policy.model_type)
    if args.source_prefix is None and is_t5_model:
        log.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    prompt_collator = PromptCollator(tokenizer=policy.tokenizer, args=args)
    if not dataset_test is None:
        test_dataset = PromptDataset(path=dataset_test, model_name=args.init_model)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=prompt_collator)
        accelerator.print(f'Load test set with {len(test_dataset)} examples')
    if not dataset_dev is None:
        dev_dataset = PromptDataset(path=dataset_dev, model_name=args.init_model)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=prompt_collator)
        accelerator.print(f'Load dev set with {len(dev_dataset)} examples')
    optimizer = Adam(policy.model.parameters(), lr=args.lr, eps=args.eps)
    args.total_steps = ceil_div(args.total_episodes, args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.total_steps)
    scheduler, optimizer = accelerator.prepare(scheduler, optimizer)
    if not dataset_test is None:
        test_dataloader = accelerator.prepare(test_dataloader)
    if not dataset_dev is None:
        dev_dataloader = accelerator.prepare(dev_dataloader)

    # load ckpt
    accelerator.load_state(saved_model_path)    

    if not dataset_test is None:
        test_out_csv_dict, test_full_scores_dict = get_samples(test_dataloader, policy, args, reward, dataset_test, accelerator, policy.tokenizer.convert_tokens_to_ids(tree_tokens[0]), "test", test_outdir_tmp)
        save_results(test_out_csv_dict, test_full_scores_dict, test_outdir)
        accelerator.wait_for_everyone()

        # calc the performance of the model before training
        test_ref_out_csv_dict, test_ref_full_scores_dict = get_samples(test_dataloader, ref_policy, args, reward, dataset_test, accelerator, None, "test", test_outdir_tmp)
        save_results(test_ref_out_csv_dict, test_ref_full_scores_dict, test_outdir_ref)

        # save args and params
        with open(os.path.join(test_outdir, 'orig_train_args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        with open(os.path.join(test_outdir, 'sampling_params.json'), 'w') as f:
            json.dump(params.__dict__, f, indent=2)


    if not dataset_dev is None:
        dev_out_csv_dict, dev_full_scores_dict = get_samples(dev_dataloader, policy, args, reward, dataset_dev, accelerator, policy.tokenizer.convert_tokens_to_ids(tree_tokens[0]), "dev", dev_outdir_tmp)
        save_results(dev_out_csv_dict, dev_full_scores_dict, dev_outdir)
        accelerator.wait_for_everyone()

        # calc the performance of the model before training
        dev_ref_out_csv_dict, dev_ref_full_scores_dict = get_samples(dev_dataloader, ref_policy, args, reward, dataset_dev, accelerator, None, "dev", dev_outdir_tmp)
        save_results(dev_ref_out_csv_dict, dev_ref_full_scores_dict, dev_outdir_ref)

        # save args and params
        with open(os.path.join(dev_outdir, 'orig_train_args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        with open(os.path.join(dev_outdir, 'sampling_params.json'), 'w') as f:
            json.dump(params.__dict__, f, indent=2)


    accelerator.print("\n################################ RESULTS ################################")
    if not dataset_test is None:
        print_results(test_full_scores_dict, accelerator, "test")
        print_results(test_ref_full_scores_dict, accelerator, "test (reference model - before finetuning)")

    if not dataset_dev is None:
        print_results(dev_full_scores_dict, accelerator, "dev")
        print_results(dev_ref_full_scores_dict, accelerator, "dev (reference model - before finetuning)")



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument("--dataset-test", type=str, default=None, help="path to test dataset.")
    argparser.add_argument("--dataset-dev", type=str, default=None, help="path to dev dataset.")

    argparser.add_argument("--saved-model-path", type=str, required=True, help="path to checkpoint of the saved model.")
    main(argparser.parse_args())