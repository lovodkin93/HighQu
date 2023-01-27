import pandas as pd
import os
import json
import spacy
from spacy import displacy
from tqdm import tqdm
from string import punctuation, whitespace
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import re
import nltk
from nltk.tree import Tree
from itertools import product

pos_that_must_include = ["ADJ", "ADV", "NOUN", "NUM", "PRON", "PROPN", "VERB", "INTJ"]
pos_conjuction = ["CONJ", "CCONJ", "SCONJ"]
pos_cconj = "CCONJ"
pos_adp = "ADP"
pos_pronoun = ["PRON"] # a span cannot consist only of pronouns.
pos_ignore_cnt = ["SPACE", "PUNCT"]

DIVIDE_CCONJ=True
DIVIDE_INTO_DIRECT_S_SUBSPANS=True
DIVIDE_INTO_ALL_S_SUBSPANS=True
ADD_COMPLEMENTARY_SUBSPANS=True # for each S span, add the complemenetary subspan to it "children" S spans

def remove_new_lines(str_i):
    return str_i.replace("\n", " ").replace("\t", " ")

def get_consecutive_subspans(idx_list, full_text):
    if not idx_list:
        return []
    idx_spans = []
    idx_list.sort()
    span_min = -1
    for i,idx in enumerate(idx_list):
        if span_min==-1:
            span_min=idx
            continue
        if idx != idx_list[i-1]+1: # new span
            if not full_text[span_min:idx_list[i-1]+1].strip():
                span_min=idx
                continue
            idx_spans.append([span_min, idx_list[i-1]+1])
            span_min=idx
    idx_spans.append([span_min, idx+1])
    return idx_spans

def remove_end_of_sents_idx(idx_list, spacy_tkns):
    """
    remove the idx of end of sentences so as to divide spans with new lines or EOS marks in the middle
    """
    end_sent_idx_list = [sent.end_char-1  if sent[-1].pos_ in ('PUNCT', 'SPACE') else sent.end_char for sent in spacy_tkns.sents] # remove the idx of end of sentence (to create separate spans), unless the last token is not punctuation (whose pos_='PUNCT') or a new line (whose pos_='SPACE'), in which case take the next idx.
    new_idx_list = [idx for idx in idx_list if not idx in end_sent_idx_list]
    idx_to_be_removed = [idx for idx in idx_list if idx in end_sent_idx_list]
    return new_idx_list, idx_to_be_removed

def remove_cconj(idx_list, spacy_tkns):
    cconj_idx_list = [idx for tkn in spacy_tkns if tkn.pos_==pos_cconj for idx in range(tkn.idx,tkn.idx+len(tkn.text))]
    new_idx_list = [idx for idx in idx_list if not idx in cconj_idx_list]
    removed_i = [tkn.i for tkn in spacy_tkns if tkn.pos_==pos_cconj]
    return new_idx_list, removed_i

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll
    # if sublist wasn't found in the for loop then it means spacy tokenized a space as a separate token, whereas the constituency parser ignored it. So we need to treat it
    l_tuples = [(i,elem) for i,elem in enumerate(l) if elem != " "] # save original indexing of each token, but remove the space tokens
    for ind in (i for i,e in enumerate(l_tuples) if e[1]==sl[0]):
        if [tpl[1] for tpl in l_tuples[l_tuples[ind:ind+sll][0][0]:l_tuples[ind:ind+sll][-1][0]]] == sl: # the list of strings of the version without extra spaces is the same (the span between the first tuple and the last tuple in that sub-list)
            return l_tuples[ind:ind+sll][0][0],l_tuples[ind:ind+sll][-1][0]+1

def traverse_tree_all_s_tags(tree):
    if not "children" in tree.keys(): # leaves
        return []
    curr_subspan_list = [str.split(tree["word"], " ")] if tree["nodeType"] in ['S', 'SINV', 'SBARQ', 'SQ'] else []
    if tree["nodeType"] != tree["link"]:
        print("WARNING!! nodeType != link of current tree")
    for subtree in tree["children"]:
        tmp_subspan_list = traverse_tree_all_s_tags(subtree)
        if tmp_subspan_list:
            curr_subspan_list = curr_subspan_list + tmp_subspan_list
    return curr_subspan_list

def get_consecutive_tkn_subspans(idx_list):
    if not idx_list:
        return []
    idx_spans = []
    idx_list.sort()
    span_min = -1
    for i,idx in enumerate(idx_list):
        if span_min==-1:
            span_min=idx
            continue
        if idx != idx_list[i-1]+1: # new span
            idx_spans.append((span_min, idx_list[i-1]+1))
            span_min=idx
    idx_spans.append((span_min, idx+1))
    return idx_spans

def get_complementary_span(span_1, span_2):
    complementary_ids = [i for i in range(span_1[0], span_1[1]) if not i in range(span_2[0], span_2[1])] 
    complementary_ids.sort()
    return get_consecutive_tkn_subspans(complementary_ids)

def i_span_to_idx_span(curr_spacy_sent, span):
    """
    convert span in tkn_i to span in idx (chars)
    """
    return (curr_spacy_sent[span[0]].idx, curr_spacy_sent[span[1]-1].idx + len(curr_spacy_sent[span[1]-1].text))

def complementary_subspan_to_str_statistics(complementary_subspan_statistic, curr_spacy_sent):
    doc_text = curr_spacy_sent.doc.text
    complementary_subspan_statistic_idx = [{"original_span": i_span_to_idx_span(curr_spacy_sent, dct["original_span"]), 
                                            "sub_span": i_span_to_idx_span(curr_spacy_sent, dct["sub_span"]), 
                                            "complementary_subspans": [i_span_to_idx_span(curr_spacy_sent, dct_subspan) for dct_subspan in dct["complementary_subspans"]]} 
                                                        for dct in complementary_subspan_statistic]
    complementary_subspan_statistic_str = [{"original_span": doc_text[dct["original_span"][0]:dct["original_span"][1]],
                                            "sub_span": doc_text[dct["sub_span"][0]:dct["sub_span"][1]],
                                            "complementary_subspans": [doc_text[x0:x1] for (x0,x1) in dct["complementary_subspans"]]} 
                                                            for dct in complementary_subspan_statistic_idx]
    return complementary_subspan_statistic_str

def separate_into_constituents(curr_spacy_sent, constituency_parser, curr_content_tokens):
    # curr_str_sent_in_tkns = " ".join([tkn.text for tkn in curr_spacy_sent])
    complementary_subspan_statistic = []
    curr_sent_tkns_str_list = [tkn.text for tkn in curr_spacy_sent]
    sent_cs = constituency_parser.predict(sentence = curr_spacy_sent.text) # constituency parsing of the sentence
    if DIVIDE_INTO_ALL_S_SUBSPANS:
        all_sent_s_labels_cs = traverse_tree_all_s_tags(sent_cs["hierplane_tree"]["root"])
        all_sent_s_labels_cs_tkn_lims = [find_sub_list(cs, curr_sent_tkns_str_list) for cs in all_sent_s_labels_cs]
        if None in all_sent_s_labels_cs_tkn_lims: # happens when spacy tokenizes a space as a separate token - FIX later (AVIVSL)
            all_sent_s_labels_cs_tkn_lims = [elem for elem in all_sent_s_labels_cs_tkn_lims if elem]
            print("WARNING: didn't find the subspan!!!!")
        if ADD_COMPLEMENTARY_SUBSPANS:
            complementary_spans = []
            for (span_1, span_2) in list(product(all_sent_s_labels_cs_tkn_lims, repeat=2)):
                if span_1 == None or span_2 == None:
                    continue
                curr_compl_subspans = []
                curr_compl_subspans = get_complementary_span(span_1, span_2)

                [complementary_spans.append(c_span) for c_span in curr_compl_subspans if (c_span and not c_span in complementary_spans)]
                if curr_compl_subspans: # not empty
                    complementary_subspan_statistic.append({"original_span":span_1, "sub_span":span_2, "complementary_subspans":curr_compl_subspans})
            all_sent_s_labels_cs_tkn_lims = all_sent_s_labels_cs_tkn_lims + complementary_spans
        cs_idx_lims = [(curr_spacy_sent[start_id].idx, curr_spacy_sent[end_id-1].idx + len(curr_spacy_sent[end_id-1].text)) for (start_id, end_id) in all_sent_s_labels_cs_tkn_lims]
    else:
        sent_1_order_cs = sent_cs["hierplane_tree"]["root"]["children"] # first order constituents
        sent_1_order_cs_tkn_lims = [find_sub_list(str.split(cs["word"], " "), curr_sent_tkns_str_list) for cs in sent_1_order_cs]
        cs_idx_lims = [(curr_spacy_sent[start_id].idx, curr_spacy_sent[end_id-1].idx + len(curr_spacy_sent[end_id-1].text)) for (start_id, end_id) in sent_1_order_cs_tkn_lims]
    sub_token_spans = []
    for idx_lim in cs_idx_lims:
        curr_sub_token_span = [tkn for tkn in curr_content_tokens if tkn.idx >=idx_lim[0] and tkn.idx<=idx_lim[1]-1]
        if curr_sub_token_span and not curr_sub_token_span in sub_token_spans:
            sub_token_spans.append(curr_sub_token_span)

    # convert to strings for the statistics    
    complementary_subspan_str_statistics = complementary_subspan_to_str_statistics(complementary_subspan_statistic, curr_spacy_sent)

    return sub_token_spans, complementary_subspan_str_statistics

def get_uhs(spans, spacy_tkns, constituency_parser):
    """
    will clean out uhs that are just conjuctions, punctuations, ar non-sensical content (and for uhs with actual content, will clean out cleaning an trailing non-sensical content)
    """
    uhs = []
    uhs_spans = [] # for statistics
    constituent_num_lst = [] # for statistics
    nonsensical_spans_cnt, nonsensical_conj_spans_cnt, nonsensical_pron_spans_cnt, nonsensical_pron_spans_after_cs_parsing_cnt = 0, 0, 0, 0 # for statistics
    complementary_subspan_str_statistics = [] # for statistics

    for span in spans:
        curr_tokens = [tkn for tkn in spacy_tkns if tkn.idx in range(span[0], span[1])]
        if not curr_tokens: # spans that cover only spaces
            continue
        curr_content_tokens = [tkn for tkn in curr_tokens if tkn.pos_ in pos_that_must_include]
        if not [tkn for tkn in curr_content_tokens if not tkn.pos_ in pos_pronoun]: # spans that are nonsensical or whose only content is a pronoun should be ignored
            if [tkn for tkn in curr_content_tokens if tkn.pos_ in pos_pronoun]: # namely the span consists only of nonsense and a pronoun
                nonsensical_pron_spans_cnt = nonsensical_pron_spans_cnt + 1
            elif [tkn for tkn in curr_tokens if tkn.pos_ in pos_conjuction]: # the span consists only of nonsense and some conjuction
                nonsensical_conj_spans_cnt = nonsensical_conj_spans_cnt + 1
            elif [tkn for tkn in curr_tokens if not tkn.pos_ in pos_ignore_cnt]: # the span consists only of nonsense
                nonsensical_spans_cnt = nonsensical_spans_cnt + 1
            # else:
            #     print("breakpoint")
            continue
        curr_content_tokens.sort(key=lambda x: x.idx)
        constituent_subspans, complementary_statistics = separate_into_constituents(curr_content_tokens[0].sent, constituency_parser, curr_content_tokens)
        complementary_subspan_str_statistics.extend(complementary_statistics)
        constituent_num_lst.append(len(constituent_subspans)) # for statistics

        for constituent_subspan in constituent_subspans:
            if not [tkn for tkn in constituent_subspan if not tkn.pos_ in pos_pronoun]: # spans that are nonsensical or whose only content is a pronoun should be ignored
                nonsensical_pron_spans_after_cs_parsing_cnt = nonsensical_pron_spans_after_cs_parsing_cnt + 1
                continue
            new_span_min = constituent_subspan[0].idx
            new_span_max = constituent_subspan[-1].idx + len(constituent_subspan[-1].text)
            curr_uh = spacy_tkns.text[new_span_min:new_span_max]
            if not curr_uh in uhs:
                uhs.append(curr_uh)
                uhs_spans.append([new_span_min,new_span_max]) # for statistics
        # # visualize the dependency tree
        # b = constituent_subspan[0].sent
        # displacy.serve(b, style="dep")
    # filter duplicates
    complementary_subspan_str_statistics_filt = []
    [complementary_subspan_str_statistics_filt.append(dct) for dct in complementary_subspan_str_statistics if not dct in complementary_subspan_str_statistics_filt]
    return uhs, uhs_spans, constituent_num_lst, nonsensical_spans_cnt, nonsensical_conj_spans_cnt, nonsensical_pron_spans_cnt, nonsensical_pron_spans_after_cs_parsing_cnt, complementary_subspan_str_statistics_filt

def get_hs(spans, spacy_tkns):
    hs = []
    hs_spans = []
    for span in spans:
        curr_tokens = [tkn for tkn in spacy_tkns if tkn.idx in range(span[0], span[1])]
        if not curr_tokens: # spans that cover only spaces
            continue
        curr_content_tokens = [tkn for tkn in curr_tokens if not tkn.pos_ in pos_ignore_cnt]
        if not curr_content_tokens:
            continue
        new_span_min = curr_content_tokens[0].idx
        new_span_max = curr_content_tokens[-1].idx + len(curr_content_tokens[-1].text)
        hs.append(spacy_tkns.text[new_span_min:new_span_max])
        hs_spans.append([new_span_min,new_span_max])
    return hs, hs_spans



def remove_dups_including_uppercase(str_list):
    str_list = list(set(str_list))
    filtered_str_i_list = []
    for i,str_i in enumerate(str_list):
        if (str_i != str_i[:1].lower() + str_i[1:]) and str_i[:1].lower() + str_i[1:] in str_list:
            continue
        filtered_str_i_list.append(i)
    return filtered_str_i_list

def rm_trailing_punct_and_whitespaces(str_list):
    return [str_i.strip(punctuation+whitespace) for str_i in str_list]

def filt_uhs_subspans_of_hs(h_lst, unhighlighlight_i_lst, full_unhighlight_lst):
    new_uh_i_lst = []
    for uh_i in unhighlighlight_i_lst:
        if [elem for elem in h_lst if elem.lower().find(full_unhighlight_lst[uh_i].lower()) != -1]:
            continue
        else:
            new_uh_i_lst.append(uh_i)
    return new_uh_i_lst

def get_spans_scores(span_list, spacy_tokenization):
    all_idx = list(set([idx for [x1,x2] in span_list for idx in range(x1,x2)]))
    all_tkn_cnt = float(len(list(set([tkn.i for tkn in spacy_tokenization if tkn.idx in all_idx and not tkn.pos_ in pos_ignore_cnt]))))
    per_span_tkn_cnt = [float(len([tkn.i for tkn in spacy_tokenization if tkn.idx in range(x0,x1) and not tkn.pos_ in pos_ignore_cnt])) for [x0,x1] in span_list]
    return [round(float(elem/all_tkn_cnt), 3) for elem in per_span_tkn_cnt]

def get_concats(original_text, spans):
    spans.sort(key= lambda x: x[0])
    # extra split and join is to remove excess spaces and new lines.
    return " ".join((" ".join([original_text[x0:x1] for [x0,x1] in spans])).split())

def main(csv_df, remove_unhighlights_duplicates, only_concats):


    # csv_df = csv_df.iloc[[0,1]] 


    csv_df["doc_text"] = csv_df["doc_text"].replace(list(csv_df["doc_text"]), [remove_new_lines(elem) for elem in list(csv_df["doc_text"])])

    constituency_parser = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
    nlp = constituency_parser._tokenizer.spacy #spacy.load("en_core_web_sm")
    if not 'sentencizer' in nlp.pipe_names:
        nlp.add_pipe('sentencizer')
    spacy_tokenization = {id:nlp(csv_df.iloc[id]["doc_text"]) for id in list(csv_df.index)}

    hs, hs_end_of_sent_idx_removed, hs_spans_orig, hs_spans_separate_sents, hs_spans = {}, {}, {}, {}, {}
    uhs, uhs_end_of_sent_idx_removed, uhs_cconj_i_removed, uhs_spans_orig, uhs_spans_separate_sents, uhs_spans_separate_1_order_cs, uhs_constituent_num_lsts, uhs_nonsensical_spans_cnts, uhs_nonsensical_conj_spans_cnt, uhs_nonsensical_pron_spans_cnt, uhs_nonsensical_pron_spans_after_cs_parsing_cnt, complementary_subspan_str_statistics = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

 


    with tqdm(total=csv_df.shape[0]) as pbar:        
        for index, row in csv_df.iterrows():
            curr_hs_spans = json.loads(row["highlight_spans"])
            curr_hs_spans.sort(key= lambda x: x[0])
            hs_idx = list(set([idx for span in curr_hs_spans for idx in range(span[0], span[1])]))
            hs_idx.sort()
            hs_spans_orig[index] = get_consecutive_subspans(hs_idx, row["doc_text"]) # for statistics (done here and not directly on curr_hs_spans because curr_hs_spans might have nested spans or overlapping spans that are actually a single span).
            hs_spans_orig[index] = [[x0,x1] for [x0,x1] in hs_spans_orig[index] if row["doc_text"][x0:x1].translate(str.maketrans('', '', punctuation)).strip()] # don't count spans that are only new lines or punctuation
            hs_idx, hs_end_of_sent_idx_removed[index] = remove_end_of_sents_idx(hs_idx, spacy_tokenization[index])
            hs_idx.sort()
            if not only_concats:
                hs_spans_separate_sents[index] = get_consecutive_subspans(hs_idx, row["doc_text"])
                hs[index], hs_spans[index] = get_hs(hs_spans_separate_sents[index], spacy_tokenization[index]) #[row["doc_text"][i0:i1] for [i0,i1] in hs_spans_separate_sents[index]]

            uhs_idx = [idx for idx in range(len(row["doc_text"])) if idx not in hs_idx]
            uhs_spans_orig[index] = get_consecutive_subspans(uhs_idx, row["doc_text"]) # for statistics
            uhs_spans_orig[index] = [[x0,x1] for [x0,x1] in uhs_spans_orig[index] if row["doc_text"][x0:x1].translate(str.maketrans('', '', punctuation)).strip()] # don't count spans that are only new lines or punctuation
            
            if not only_concats:
                uhs_idx, uhs_end_of_sent_idx_removed[index] = remove_end_of_sents_idx(uhs_idx, spacy_tokenization[index])
                if DIVIDE_CCONJ:
                    uhs_idx, uhs_cconj_i_removed[index] = remove_cconj(uhs_idx, spacy_tokenization[index])
                uhs_idx.sort()
                uhs_spans_separate_sents[index] = get_consecutive_subspans(uhs_idx, row["doc_text"])
                uhs[index], uhs_spans_separate_1_order_cs[index], uhs_constituent_num_lsts[index], uhs_nonsensical_spans_cnts[index], uhs_nonsensical_conj_spans_cnt[index], uhs_nonsensical_pron_spans_cnt[index], uhs_nonsensical_pron_spans_after_cs_parsing_cnt[index], complementary_subspan_str_statistics[index] = get_uhs(uhs_spans_separate_sents[index], spacy_tokenization[index], constituency_parser)
            pbar.update(1)
    print("done")

    if not only_concats:
        # remove trailing punctuation and whitespaces
        final_uhs = {key:rm_trailing_punct_and_whitespaces(value) for key,value in uhs.items()}
        final_hs = {key:rm_trailing_punct_and_whitespaces(value) for key,value in hs.items()} 

        # remove duplicates
        if remove_unhighlights_duplicates:
            final_uhs_i = {key:remove_dups_including_uppercase(value) for key,value in final_uhs.items()} #{key:list(set(value)) for key,value in uhs.items()} 
            final_hs_i = {key:remove_dups_including_uppercase(value) for key,value in final_hs.items()} 
        else:
            final_uhs_i = {key:[i for i,uh in enumerate(final_uhs[key])] for key in final_uhs.keys()}
            final_hs_i = {key:[i for i,uh in enumerate(final_hs[key])] for key in final_hs.keys()}

        # remove unhighlight spans that are subspans of highlights
        final_uhs_i = {key:filt_uhs_subspans_of_hs(final_hs[key], value, final_uhs[key]) for key,value in final_uhs_i.items()}

        final_filtered_uhs = {key:{ind:final_uhs[key][ind] for ind in final_uhs_i[key]} for key in final_uhs_i.keys()}    
        final_filtered_hs = {key:{ind:final_hs[key][ind] for ind in final_hs_i[key]} for key in final_hs_i.keys()}    


        uhs_score = {key:get_spans_scores(uhs_spans_separate_1_order_cs[key], spacy_tokenization[key]) for key in final_uhs_i.keys()} 
        hs_score = {key:get_spans_scores(hs_spans[key], spacy_tokenization[key]) for key in final_hs_i.keys()} 

        final_filtered_uhs_score = {key:{ind:uhs_score[key][ind] for ind in final_uhs_i[key]} for key in final_uhs_i.keys()}
        final_filtered_hs_score = {key:{ind:hs_score[key][ind] for ind in final_hs_i[key]} for key in final_hs_i.keys()}


        filtered_uhs = {key:len(uhs[key])-len(final_filtered_uhs[key]) for key in final_filtered_uhs.keys()}
        filtered_hs = {key:len(hs[key])-len(final_filtered_hs[key]) for key in final_filtered_hs.keys()}

    # get concatenation of highlights and highlights for evaluation
    hs_concats = [get_concats(spacy_tokenization[index].text, hs_spans_orig[index]) for index in hs_spans_orig.keys()]
    uhs_concats = [get_concats(spacy_tokenization[index].text, uhs_spans_orig[index]) for index in uhs_spans_orig.keys()]

    if only_concats:
        return None, None, None, None, hs_concats, uhs_concats
    else:
        return final_filtered_uhs, final_filtered_hs, final_filtered_uhs_score, final_filtered_hs_score, hs_concats, uhs_concats



if __name__ == "__main__":
    csv_path = r"/home/nlp/sloboda1/controlled_reduction/controlled_reduction_models_dir/data/dev__highlights.csv"
    csv_df = pd.read_csv(csv_path)
    remove_unhighlights_duplicates = False
    only_concats = False
    final_uhs, final_hs = main(csv_df, remove_unhighlights_duplicates, only_concats)