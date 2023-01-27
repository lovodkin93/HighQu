from preprocessor import Preprocessor, get_special_tokens_constants
from get_highlights_and_unhighlights import main as highlights_and_unhighlights_extractor
import pandas as pd
from tqdm import tqdm
import json
import argparse
from concatenate_highlights import concatenate_highlights


def save_hs_and_uhs_to_outdir(final_hs, final_uhs):
    outdir = "extracted_highlights_and_unhighlights_26_11_22_final.txt"
    with open(outdir, 'w', encoding='utf-8') as f1:
        for row_id in final_hs.keys():
            f1.write(f"id: {row_id}\n")
            f1.write("highlights:\n")
            for h_string in final_hs[row_id]:
                f1.write(f"{h_string}")
                f1.write(f"\n-------------------------------------------------------------\n")
            f1.write("\n\n")
            f1.write("unhighlights:\n")
            for uh_string in final_uhs[row_id]:
                f1.write(f"{uh_string}")
                f1.write(f"\n-------------------------------------------------------------\n")
            f1.write("\n\n\n")


def get_full_jsonl(preprocessor, csv_path, remove_unhighlights_duplicates, only_concats):
    csv_df = pd.read_csv(csv_path)
    # csv_df = csv_df.iloc[[0,1]]

    print("Begin extracting input (with highlights) and output")
    inputs, targets = {}, {}
    doc_ids, summary_ids = {}, {}
    with tqdm(total=csv_df.shape[0]) as pbar:        
        for index, row in csv_df.iterrows():
                curr_input = preprocessor.preprocess_input(row['doc_text'], row['highlight_spans'])
                inputs[index] = curr_input
                curr_output = preprocessor.preprocess_output(row['summary_text'], curr_input)
                targets[index] = curr_output
                if 'doc_id' in row:
                    doc_ids[index] = row["doc_id"]
                if 'summary_id' in row:
                    summary_ids[index] = row["summary_id"]

                pbar.update(1)
    # get highlights and unhighlights
    print("Begin extracting the highlights and unhighlights")
    final_uhs, final_hs, final_uhs_scores, final_hs_scores, hs_concats, uhs_concats = highlights_and_unhighlights_extractor(csv_df, remove_unhighlights_duplicates, only_concats)

    return inputs, targets, final_uhs, final_hs, final_uhs_scores, final_hs_scores, hs_concats, uhs_concats, doc_ids, summary_ids

def write_to_jsonl(outdir, inputs, targets, final_uhs, final_hs, final_uhs_scores, final_hs_scores, hs_concats, uhs_concats, only_concats, doc_ids, summary_ids):
    with open(outdir, 'w') as f:
        for i in inputs.keys():
            if only_concats:
                curr_dict = {
                    "input":inputs[i], 
                    "gold_summary":targets[i], 
                    "highlights_concatenation": hs_concats[i], 
                    "unhighlights_concatenation": uhs_concats[i]
                    }
            else:
                curr_dict = {
                            "input":inputs[i], 
                            "gold_summary":targets[i], 
                            "highlights":[{"text":final_hs[i][key], "score":final_hs_scores[i][key]} for key in final_hs[i].keys()], 
                            "unhighlights":[{"text":final_uhs[i][key], "score":final_uhs_scores[i][key]} for key in final_uhs[i].keys()],
                            "highlights_concatenation": hs_concats[i], 
                            "unhighlights_concatenation": uhs_concats[i]
                            }
            if doc_ids != {}:
                curr_dict["doc_id"] = doc_ids[i]
            if summary_ids != {}:
                curr_dict["summary_id"] = summary_ids[i]
            f.write(json.dumps(curr_dict, indent=None))
            f.write('\n')



def main(args):
    csv_path = args.csv_path
    outdir = args.outdir
    is_t5_model = args.is_t5_model
    prefix = "summarize: " if args.is_t5_model else ""
    special_tokens_constants = get_special_tokens_constants(is_t5_model)
    should_preprocess_add_highlights = True
    should_preprocess_only_sents_with_highlights = False
    should_preprocess_keep_only_highlights = False
    add_planning_on_concatenation = False
    add_highlight_delim_planning = False
    add_highlight_labels_to_planning = False
    preprocessor = Preprocessor(prefix, special_tokens_constants, should_preprocess_add_highlights, should_preprocess_only_sents_with_highlights, should_preprocess_keep_only_highlights, add_planning_on_concatenation, add_highlight_delim_planning, add_highlight_labels_to_planning)

    inputs, targets, final_uhs, final_hs, final_uhs_scores, final_hs_scores, hs_concats, uhs_concats, doc_ids, summary_ids = get_full_jsonl(preprocessor, csv_path, args.remove_unhighlights_duplicates, args.only_concats)
    write_to_jsonl(outdir, inputs, targets, final_uhs, final_hs, final_uhs_scores, final_hs_scores, hs_concats, uhs_concats, args.only_concats, doc_ids, summary_ids)
    print("Done")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument("--csv-path", default="/home/nlp/sloboda1/controlled_reduction/controlled_reduction_models_dir/data/dev__highlights.csv", help="path to csv with following columns: \"doc_text\", \"summary_text\", \"highlight_spans\"")
    argparser.add_argument("-o", "--outdir", default="data/controlled_text_reduction/testing.jsonl", help="path to output jsonl file where each line corresponds to a dict with the following keys: \"input\" (input doc + highlights), \"gold_summary\", \"highlights\" (desirable spans in the output - from highlighted content), \"unhighlights\" (undesirable spans in the output - from the unhighlighted content)")
    argparser.add_argument("--is-t5-model", action='store_true', default=False, help="Whether the input is going to be to a T5 model (affects the special tokens) and the prefix token")
    argparser.add_argument("--remove-unhighlights-duplicates", action='store_true', default=False, help="Whether to remove duplicated unhighlights spans (constituents), including such that onply differ by an upper-case")
    argparser.add_argument("--only-concats", action='store_true', default=False, help="Whether to only generate the concatentations of the highlights and the unhighlights.")

    main(argparser.parse_args())