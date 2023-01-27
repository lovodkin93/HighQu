import json
from pathlib import Path
from typing import TypeVar, Iterable, List, Union, Any
import numpy as np
import torch
from tqdm.auto import tqdm
import os
import collections
from utils.constants import NEGATIVE_INF
import pdb
T = TypeVar('T')
from preprocess_utils.preprocessor import get_special_tokens_constants
import logging

log = logging.getLogger(__name__)




def add_control_code(input_ids, attention_mask, global_attention_mask, best_cat_id):
    input_ids = torch.cat([input_ids.new([best_cat_id] * len(input_ids))[:, None], input_ids], dim=1)
    attention_mask = torch.cat([attention_mask.new([1] * len(attention_mask))[:, None], attention_mask], dim=1)
    if global_attention_mask != None:
        global_attention_mask = torch.cat([global_attention_mask.new([1] * len(global_attention_mask))[:, None], global_attention_mask], dim=1)
    return input_ids, attention_mask, global_attention_mask




def is_t5_model_def(model_name, model_type):
    return model_name in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"] or model_type == 't5'


def reduce_sum(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask)
    return torch.sum(value * mask, axis)


def reduce_mean(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask) / torch.sum(mask)
    return reduce_sum(value, mask, axis) / torch.sum(mask, axis)


def reduce_std(value, mask):
    return torch.sqrt(reduce_mean(torch.square(value), mask) - torch.square(reduce_mean(value, mask)))


def logits_to_entropy(logits):
    distribution = torch.distributions.Categorical(logits=logits)
    return distribution.entropy()


def mask_pad(value, mask):
    return value * mask + NEGATIVE_INF * (1 - mask)


def clamp(value, min_value, max_value):
    return torch.max(torch.min(value, max_value), min_value)


def ceil_div(a, b):
    return (a - 1) // b + 1


def ensure_dir(d):
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except FileExistsError:
            log.warning(f"folder {d} already exists.")


def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file) as f:
        for line in f:
            yield json.loads(line)


def load_cache(file: Path):
    if file.exists():
        with file.open() as f:
            for line in tqdm(f, desc=f'Loading cache from {file}'):
                yield json.loads(line)


def get_jsonl_line_data(path: str, jsonl_id: str, attrib: str):
    """
    returns the desirable attribute from the relevant jsonl line.
    """
    with open(path, 'r') as f1:
        if attrib in ["highlights", "unhighlights"]:
            return [{"text":h["text"].strip(), "score":h["score"]} for h in json.loads(f1.readlines()[int(jsonl_id)].strip())[attrib]]
        elif attrib in ["highlights_concatenation", "unhighlights_concatenation", "gold_summary", "input"]:
            # extra split and join is to remove excess spaces and new lines.
            return " ".join(json.loads(f1.readlines()[int(jsonl_id)].strip())[attrib].strip().split())
        else:
            raise ValueError(f"Attribute {attrib} is not supported yet by the get_jsonl_line_data function.")

def custom_clean_special_tokens(input_doc, model_name, model_type, special_tokens):
    is_t5_model = is_t5_model_def(model_name, model_type)
    special_tokens_constants = get_special_tokens_constants(is_t5_model)
    tokens_to_keep = [value for key,value in special_tokens_constants.items() if key in ['highlight_start', 'highlight_end']]
    all_special_tokens = sum([[value]  if type(value)==str else value for value in special_tokens], [])
    tokens_to_remove = [tkn for tkn in all_special_tokens if not tkn in tokens_to_keep]
    for tkn in tokens_to_remove:
        input_doc = [elem.replace(tkn, "") for elem in input_doc]
    return input_doc