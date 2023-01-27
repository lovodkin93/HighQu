import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from utils.constants import NEGATIVE_INF
from utils.utils import logits_to_entropy, mask_pad, is_t5_model_def
from preprocess_utils.preprocessor import get_special_tokens_constants
import wandb
from torch.nn.parallel.distributed import DistributedDataParallel



class Policy:
    def __init__(self, model_name, temperature, device, reward_cond=False, tree_tokens=None, args={}, logger=None, last_checkpoint=None, accelerator=None):
        model_args_dict={}
        model_args_dict.update(args.__dict__)
        model_args_dict['max_length'] = args.max_target_length
        model_args_dict['num_beams'] = args.num_beams
        model_args_dict['no_repeat_ngram_size'] = args.no_repeat_ngram_size
        model_args_dict['length_penalty'] = args.length_penalty
        model_args_dict = { k: v for k,v in model_args_dict.items() if v is not None}
        self.config = AutoConfig.from_pretrained(model_name, **model_args_dict) # Important otherwise it might override default values
        self.logger = logger
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                cache_dir=args.cache_dir,
                use_fast=args.use_fast_tokenizer,
                revision=args.model_revision,
                use_auth_token=True if args.use_auth_token else None
            )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    from_tf=bool(".ckpt" in model_name),
                    config=self.config,
                    cache_dir=args.cache_dir,
                    revision=args.model_revision,
                    use_auth_token=True if args.use_auth_token else None,
                )
        if not last_checkpoint == None:
            self.model.load_state_dict(last_checkpoint['policy_model'])
        self.model_name = self.model.name_or_path
        self.model_type = self.model.config.model_type

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            self.tokenizer.lang_code_to_id[args.forced_bos_token] if args.forced_bos_token is not None else None
        )
        self.model.config.forced_bos_token_id = forced_bos_token_id

        is_t5_model = is_t5_model_def(self.model_name, self.model_type)        
        self.tokenizer.add_special_tokens({'additional_special_tokens': [get_special_tokens_constants(is_t5_model)["highlight_start"], get_special_tokens_constants(is_t5_model)["highlight_end"]]}) 
        self.model.resize_token_embeddings(len(self.tokenizer))
        if (hasattr(self.model.config, "max_position_embeddings") and self.model.config.max_position_embeddings < args.max_source_length):
            if args.resize_position_embeddings is None:
                logger.warning(f"Increasing the model's number of position embedding vectors from {self.model.config.max_position_embeddings} to {args.max_source_length}.")
                self.model.resize_position_embeddings(args.max_source_length)
            elif args.resize_position_embeddings:
                self.model.resize_position_embeddings(args.max_source_length)
            else:
                raise ValueError(
                    f"`--max_source_length` is set to {args.max_source_length}, but the model only has {self.model.config.max_position_embeddings}"
                    f" position encodings. Consider either reducing `--max_source_length` to {self.model.config.max_position_embeddings} or to automatically "
                    "resize the model's position encodings by passing `--resize_position_embeddings`."
                )
        
        self.device = device

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if reward_cond:
            self.tokenizer.add_tokens(tree_tokens, special_tokens=True)

            weights = self.model.get_input_embeddings().weight.detach().numpy()
            mean_weights, std_weights = np.mean(weights, axis=0), np.std(weights, axis=0)
            new_inits = np.vstack([np.random.normal(loc=mean_weights, scale=std_weights) for _ in tree_tokens])

            self.model.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                new_inits = torch.tensor(new_inits)
                self.model.get_input_embeddings().weight[-len(tree_tokens):, :] = new_inits

        self.model = accelerator.prepare(self.model)

        if hasattr(self.model, 'parallelize'):
            self.model.parallelize()
        else:
            logger.warning(f"the finetuned model ({model_name}) doesn't have a parallelize method, so training might be slower than usual.")

        self.temperature = temperature
        self.accelerator=accelerator

    def sample(self,
               input_ids: torch.Tensor = None,
               attention_mask: torch.Tensor = None,
               max_len: int = 20,
               min_len: int = 3,
               sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None,
               global_attention_mask: torch.Tensor = None) -> Dict[str, Union[torch.Tensor, List[str]]]:
        if temperature is None:
            temperature = self.temperature


        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        global_attention_mask = global_attention_mask.to(self.device) if global_attention_mask != None else None

            
        model_kwargs = {'attention_mask': attention_mask}
        model_kwargs["max_length"] = self.config.max_length
        model_kwargs["num_beams"] = self.config.num_beams
        if not self.config.length_penalty == None:
            model_kwargs["length_penalty"] = self.config.length_penalty
        if not self.config.no_repeat_ngram_size == None:
            model_kwargs["no_repeat_ngram_size"] = self.config.no_repeat_ngram_size

        if global_attention_mask != None:
            model_kwargs["global_attention_mask"] = global_attention_mask
            

        self.model.eval()
        with torch.no_grad():
            
            # extract the model from the DistributedDataParallel wrapper
            curr_model = self.accelerator.unwrap_model(self.model) if type(self.model) == DistributedDataParallel else self.model

            outputs = curr_model.generate(
                    input_ids,
                    **model_kwargs,
                    return_dict_in_generate=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    output_scores=True
            )

        output_ids = outputs.sequences
        output_texts = [self.tokenizer.decode(output_ids_instance, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        for output_ids_instance in output_ids] 

        input_ids = input_ids
        input_text = [self.tokenizer.decode(input_ids_instance, clean_up_tokenization_spaces=False)
                        for input_ids_instance in input_ids] 
        
        # remove all the special tokens, except the "highlight_start" and "highlight_end" ones (which appeared in the original text)
        is_t5_model = is_t5_model_def(self.model_name, self.model_type)
        special_tokens_constants = get_special_tokens_constants(is_t5_model)
        tokens_to_keep = [value for key,value in special_tokens_constants.items() if key in ['highlight_start', 'highlight_end']]
        all_special_tokens = sum([[value]  if type(value)==str else value for value in self.tokenizer.special_tokens_map.values()], [])
        tokens_to_remove = [tkn for tkn in all_special_tokens if not tkn in tokens_to_keep]
        for tkn in tokens_to_remove:
            input_text = [elem.replace(tkn, "") for elem in input_text]
        
        logits = torch.stack(outputs.scores, dim=1)
        log_prob = F.log_softmax(logits, dim=-1)
        output_logprob = torch.gather(log_prob, 2, output_ids[:, :, None][:,1:,:]).squeeze(2) # omitting the first token from output_ids because it is simply the start decoding token that is alawys prefixed to the output

        # output mask will mask out pad tokens
        output_mask = output_ids == self.tokenizer.pad_token_id
        output_mask = 1 - output_mask.to(int) # convert to int and then False --> 0 and True --> 1 so one minus will yield 1 for when not pad token and 0 for pad token

        return {
            'input_doc/input_ids': input_ids,
            'input_doc/text': input_text,
            'input_doc/mask': attention_mask,
            'input_doc/global_mask': global_attention_mask,
            'output_reduction/input_ids': output_ids,
            'output_reduction/text': output_texts,
            'output_reduction/mask': output_mask,
            'output_reduction/log_prob': output_logprob
        }

    def forward_pass(self,
                     input_texts_input_ids: torch.Tensor,
                     input_texts_attention_mask: torch.Tensor,
                     input_texts_global_attention_mask: torch.Tensor,
                     generated_reductions_input_ids: torch.Tensor,
                     generated_reductions_attention_mask: torch.Tensor):


        
        input_texts_input_ids = input_texts_input_ids.to(self.device)
        input_texts_attention_mask = input_texts_attention_mask.to(self.device)
        input_texts_global_attention_mask = input_texts_global_attention_mask.to(self.device)
        generated_reductions_input_ids = generated_reductions_input_ids.to(self.device)
        generated_reductions_attention_mask = generated_reductions_attention_mask.to(self.device)

        batch_size, generated_reductions_max_length = generated_reductions_input_ids.shape

        model_kwargs = {'attention_mask': input_texts_attention_mask}
        if input_texts_global_attention_mask != None:
            model_kwargs["global_attention_mask"] = input_texts_global_attention_mask
        model_inputs = self.prepare_inputs_for_generation(input_texts_input_ids, generated_reductions_input_ids, **model_kwargs)
        
        # forward pass to get next summary
        self.model.train()
        outputs = self.model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )        


        logits = outputs.logits
        if logits.isnan().any():
            self.logger.warning(f"Received logits that are NaN")
        if  logits.isinf().any():
            self.logger.warning(f"Received logits that are infinite")       

        log_prob = F.log_softmax(logits, dim=-1)
        output_logprob = torch.gather(log_prob, 2, generated_reductions_input_ids[:, :, None]).squeeze(2) # finds the probability for each token in the gold summary at the step it was chosen
        output_entropy = logits_to_entropy(logits)
        lm_loss = -1. * output_logprob

        output_sequences_ids = torch.argmax(log_prob, 2, keepdim=False)
        output_texts = [self.tokenizer.decode(output_sequences_ids[ind,:]) for ind in range(log_prob.shape[0])]

        return {
            'generated_reduction/log_prob': mask_pad(output_logprob, generated_reductions_attention_mask),
            'generated_reduction/lm_loss': mask_pad(lm_loss, generated_reductions_attention_mask),
            'generated_reduction/entropy': mask_pad(output_entropy, generated_reductions_attention_mask),
            'generated_reduction/logits': logits,
            'generated_reduction/masks': generated_reductions_attention_mask,
            'generated_reduction/output_text': output_texts
        }

    def prepare_inputs_for_generation(self, input_ids, generated_reductions_input_ids, past=None, **kwargs):


        attention_mask = kwargs.get("attention_mask", None)
        pre_generation_token = self.config.decoder_start_token_id
        pad_pre_generation_token_tensor = (pre_generation_token)*torch.ones([generated_reductions_input_ids.shape[0],1]).to(generated_reductions_input_ids.device).to(generated_reductions_input_ids.dtype)
        generated_reductions_input_ids = torch.cat([pad_pre_generation_token_tensor, generated_reductions_input_ids], dim=1)
        decoder_input_ids = generated_reductions_input_ids[:, :-1].contiguous()

        
        return_dict = {
            "input_ids": input_ids,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids
        }

        if  "global_attention_mask" in kwargs.keys():
            return_dict["global_attention_mask"] = kwargs.get("global_attention_mask", None)
        
        return return_dict
