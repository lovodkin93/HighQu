from typing import List
from copy import deepcopy




class DataPool:
    def __init__(self, tree_tokens, n_extra_tokens, model_name, data_path):
        self.tree_tokens = tree_tokens
        self.n_extra_tokens = n_extra_tokens
        self.model_name = model_name
        self.data_path = data_path

        self.cat_tokens = None
        self.input_pool, self.jsonl_id_pool, self.generated_reduction_pool, self.score_pool = [], [], [], []

    def add(self, inputs: List[str], jsonl_ids: List[str], generated_reductions: List[str], scores: List[float]):

        self.input_pool.extend(inputs)
        self.jsonl_id_pool.extend(jsonl_ids)
        self.generated_reduction_pool.extend(generated_reductions)
        self.score_pool.extend(scores)

        data = zip(self.input_pool, self.jsonl_id_pool, self.generated_reduction_pool, self.score_pool)
        data = [x for x in data if x[-1] is not None]
        sorted_data = sorted(data, key=lambda x: x[-1], reverse=True)
        self.input_pool, self.jsonl_id_pool, self.generated_reduction_pool, self.score_pool = [list(x) for x in list(zip(*sorted_data))]

        
        cat_pos = [[i] * (len(sorted_data) // self.n_extra_tokens) for i in range(self.n_extra_tokens)]
        cat_pos = [y for x in cat_pos for y in x]
        cat_pos = cat_pos + [self.n_extra_tokens - 1] * (len(sorted_data) - len(cat_pos))
        self.cat_tokens = [self.tree_tokens[i] for i in cat_pos]

    def get_data(self):
        return deepcopy(self.input_pool), deepcopy(self.jsonl_id_pool), deepcopy(self.generated_reduction_pool), deepcopy(self.cat_tokens)


