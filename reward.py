import json
from pathlib import Path
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from typing import List, Iterable, Dict, Any

from utils.utils import load_jsonl, get_jsonl_line_data
from utils.constants import ROUGE1, ROUGE2, ROUGEL, ROUGELSUM, all_rouges

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)
import pdb
import re
import ast


class Reward:
    def __init__(self, save_path: str, batch_size: int, R_P_rewards_ratio: float, R_P_rewards_alternate: bool, R_P_rewards_alternate_ratio: str, reward_type: str, rouge_metric_scorer, sample_interval, P_reward_type, R_reward_type):
        self.path = save_path
        self.batch_size = batch_size
        self.R_P_rewards_ratio = R_P_rewards_ratio
        self.R_P_rewards_alternate = R_P_rewards_alternate
        self.R_P_rewards_alternate_ratio = {'R':ast.literal_eval(R_P_rewards_alternate_ratio)[0], 'P':ast.literal_eval(R_P_rewards_alternate_ratio)[1]}
        self.rouge_metric_scorer = rouge_metric_scorer
        self.reward_type = reward_type
        self.sample_interval = int(sample_interval)
        self.P_reward_type = P_reward_type
        self.R_reward_type = R_reward_type

        self.P_reward_rouge_version = P_reward_type.split("_")[2] if len(P_reward_type.split("_")) == 3 else ROUGEL 
        self.R_reward_rouge_version = R_reward_type.split("_")[2] if len(R_reward_type.split("_")) == 3 else ROUGEL 

    def get_all_rouges(self, output_reductions, jsonl_path, jsonl_ids):
        h_rouges, uh_rouges = [], []
        h_rouges_precision, uh_rouges_precision = [], []
        h_rouges_recall, uh_rouges_recall = [], []
        for i,output_reduction in enumerate(tqdm(output_reductions, total=len(output_reductions), desc='Calculating samples Rouge scores with highlights and unhighlights')):
            curr_coverage_cocnats = get_jsonl_line_data(path=jsonl_path, jsonl_id=jsonl_ids[i], attrib="highlights_concatenation") 
            curr_faithfulness_cocnats = get_jsonl_line_data(path=jsonl_path, jsonl_id=jsonl_ids[i], attrib="unhighlights_concatenation") 
            h_rouges.append({key:value.fmeasure for key,value in self.rouge_metric_scorer.score(curr_coverage_cocnats, output_reduction).items()})
            uh_rouges.append({key:value.fmeasure for key,value in self.rouge_metric_scorer.score(curr_faithfulness_cocnats, output_reduction).items()})

            h_rouges_precision.append({key:value.precision for key,value in self.rouge_metric_scorer.score(curr_coverage_cocnats, output_reduction).items()})
            uh_rouges_precision.append({key:value.precision for key,value in self.rouge_metric_scorer.score(curr_faithfulness_cocnats, output_reduction).items()})

            h_rouges_recall.append({key:value.recall for key,value in self.rouge_metric_scorer.score(curr_coverage_cocnats, output_reduction).items()})
            uh_rouges_recall.append({key:value.recall for key,value in self.rouge_metric_scorer.score(curr_faithfulness_cocnats, output_reduction).items()})



        total_h_rouge_scores = {f"highlights_{key}":[elem[key] for elem in h_rouges] for key in h_rouges[0].keys()}
        total_uh_rouge_scores = {f"unhighlights_{key}":[elem[key] for elem in uh_rouges] for key in uh_rouges[0].keys()}

        total_h_rouge_precision_scores = {f"highlights_{key}_precision":[elem[key] for elem in h_rouges_precision] for key in h_rouges_precision[0].keys()}
        total_uh_rouge_precision_scores = {f"unhighlights_{key}_precision":[elem[key] for elem in uh_rouges_precision] for key in uh_rouges_precision[0].keys()}

        total_h_rouge_recall_scores = {f"highlights_{key}_recall":[elem[key] for elem in h_rouges_recall] for key in h_rouges_recall[0].keys()}
        total_uh_rouge_recall_scores = {f"unhighlights_{key}_recall":[elem[key] for elem in uh_rouges_recall] for key in uh_rouges_recall[0].keys()}
        return {**total_h_rouge_scores , **total_uh_rouge_scores, **total_h_rouge_precision_scores, **total_uh_rouge_precision_scores, **total_h_rouge_recall_scores, **total_uh_rouge_recall_scores}
        



    def get_subspans_reward(self, input_docs: List[str], jsonl_ids: List[str], output_reductions: List[str], epoch: str, jsonl_path: str, is_sampling: bool) -> List[float]:
        faithfulness_scores, coverage_scores = [], []
        # faithfulness
        for i,output_reduction in enumerate(tqdm(output_reductions, total=len(output_reductions), desc='Calculating samples rewards:')):
            curr_unhighlights = get_jsonl_line_data(path=jsonl_path, jsonl_id=jsonl_ids[i], attrib="unhighlights") 
            unhighlights_in_reduction_ranges = {ind:[range(m.start(),m.end()) for m in re.finditer(re.escape(unhighlight_dct["text"]), output_reduction)] for ind,unhighlight_dct in enumerate(curr_unhighlights)} 
            # if one of the subspans in the generated reduction is a sabspan of another one - remove it (so it doesn't affect twice - on its own and as part of the larger subspan)
            all_ranges = [rng for elem in unhighlights_in_reduction_ranges.values() for rng in elem]
            unhighlights_in_reduction_ranges_filtered = {ind:[] for ind in unhighlights_in_reduction_ranges.keys()}
            for i,rng_list in unhighlights_in_reduction_ranges.items():
                curr_filter_rng = [rng for rng in rng_list if not [rng2 for rng2 in all_ranges if set(rng).issubset(rng2) and rng2 != rng]]
                unhighlights_in_reduction_ranges_filtered[i].extend(curr_filter_rng)
            # faithfulness_score=sum(score[i]*number_of_occurences_in_reduction[i]) 
            curr_faithfulness_score = sum([curr_unhighlights[uh_ind]["score"]*len(unhighlights_in_reduction_ranges_filtered[uh_ind]) for uh_ind,elem in enumerate(curr_unhighlights)])
            faithfulness_scores.append(curr_faithfulness_score)
        
        # coverage
        for i,output_reduction in enumerate(output_reductions):
            curr_highlights = get_jsonl_line_data(path=jsonl_path, jsonl_id=jsonl_ids[i], attrib="highlights")
            rouge_scores = [{key:value.precision for key,value in self.rouge_metric_scorer.score(curr_highlight["text"], output_reduction).items() if key in rouges_to_consider} for curr_highlight in curr_highlights]
            rouge_scores = [sum(elem.values())/len(elem.values()) for elem in rouge_scores]
            curr_recall_score = sum([curr_highlights[i]["score"]*rouge_scores[i] for i,_ in enumerate(curr_highlights)])
            coverage_scores.append(curr_recall_score)

        if self.R_P_rewards_alternate and is_sampling:
            curr_step = int(epoch.replace("step", ""))
            if (curr_step / self.sample_interval) % sum(self.R_P_rewards_alternate_ratio.values()) < self.R_P_rewards_alternate_ratio["P"]:
                final_rewards = [-faithfulness_scores[i] for i,_ in enumerate(coverage_scores)]
            else:
                final_rewards = [coverage_scores[i] for i,_ in enumerate(coverage_scores)]
        else:
            final_rewards = [(1-self.R_P_rewards_ratio)*(-faithfulness_scores[i]) + self.R_P_rewards_ratio*coverage_scores[i] for i,_ in enumerate(coverage_scores)]
        
        all_rouges = self.get_all_rouges(output_reductions, jsonl_path, jsonl_ids) if not is_sampling else {}
        scores_dicr=  {
                        "precision_score": [-scr for scr in faithfulness_scores], 
                        "recall_score": coverage_scores,
                        "final_reward": final_rewards
                      }
        return {**scores_dicr, **all_rouges}


    def get_score(self, scr, isCoverage): 
        curr_reward_type = self.R_reward_type if isCoverage else self.P_reward_type
        if curr_reward_type.split("_")[1] == "precision":
            curr_scr = scr.precision
        elif curr_reward_type.split("_")[1] == "recall":
            curr_scr = scr.recall
        elif curr_reward_type.split("_")[1] == "F1":
            curr_scr = scr.fmeasure
        else:
            raise TypeError("\"R_reward_type\"'s and \"P_reward_type\"'s second word needs to be one of the following: \'precision\', \'recall\', \'F1\'")
        if curr_reward_type.split("_")[0] == "unhighlights":
            return -curr_scr
        else:
            return curr_scr    


    def get_concats_reward(self, input_docs: List[str], jsonl_ids: List[str], output_reductions: List[str], epoch: str, jsonl_path: str, is_sampling: bool) -> List[float]:
        faithfulness_scores, coverage_scores = [], []
        for i,output_reduction in enumerate(tqdm(output_reductions, total=len(output_reductions), desc='Calculating samples rewards:')):
            # faithfulness
            curr_faithfulness_cocnats = get_jsonl_line_data(path=jsonl_path, jsonl_id=jsonl_ids[i], attrib=f"{self.P_reward_type.split('_')[0]}_concatenation") 
            faithfulness_rouge_scores = {key:self.get_score(value, False) for key,value in self.rouge_metric_scorer.score(curr_faithfulness_cocnats, output_reduction).items() if key==self.P_reward_rouge_version}
            faithfulness_score = sum(faithfulness_rouge_scores.values())/len(faithfulness_rouge_scores.values())
            faithfulness_scores.append(faithfulness_score)

            # coverage
            curr_coverage_cocnats = get_jsonl_line_data(path=jsonl_path, jsonl_id=jsonl_ids[i], attrib=f"{self.R_reward_type.split('_')[0]}_concatenation") 
            coverage_rouge_scores = {key:self.get_score(value, True) for key,value in self.rouge_metric_scorer.score(curr_coverage_cocnats, output_reduction).items() if key==self.R_reward_rouge_version}
            recall_score = sum(coverage_rouge_scores.values())/len(coverage_rouge_scores.values())
            coverage_scores.append(recall_score)
        if self.R_P_rewards_alternate and is_sampling:
            curr_step = int(epoch.replace("step", ""))
            if (curr_step / self.sample_interval) % sum(self.R_P_rewards_alternate_ratio.values()) < self.R_P_rewards_alternate_ratio["P"]:
                final_rewards = [np.abs(faithfulness_scores[i]) for i,_ in enumerate(coverage_scores)]
            else:
                final_rewards = [coverage_scores[i] for i,_ in enumerate(coverage_scores)]
        else:
            final_rewards = [(1-self.R_P_rewards_ratio)*(faithfulness_scores[i]) + self.R_P_rewards_ratio*coverage_scores[i] for i,_ in enumerate(coverage_scores)]
        
        all_rouges = self.get_all_rouges(output_reductions, jsonl_path, jsonl_ids)
        scores_dicr =  {
                        "precision_score": [scr for scr in faithfulness_scores], 
                        "recall_score": coverage_scores,
                        "final_reward": final_rewards
                        }
        return {**scores_dicr, **all_rouges}





    def get_reward(self, input_docs: List[str], jsonl_ids: List[str], output_reductions: List[str], epoch: str, jsonl_path: str, is_sampling: bool) -> List[float]:
        if self.reward_type == "subspans":
            return self.get_subspans_reward(input_docs=input_docs, jsonl_ids=jsonl_ids, output_reductions=output_reductions, epoch=epoch, jsonl_path=jsonl_path, is_sampling=is_sampling)
        elif self.reward_type == "concats":
            return self.get_concats_reward(input_docs=input_docs, jsonl_ids=jsonl_ids, output_reductions=output_reductions, epoch=epoch, jsonl_path=jsonl_path, is_sampling=is_sampling)

