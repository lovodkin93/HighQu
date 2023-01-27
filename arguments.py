import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # dataset
    parser.add_argument(
        '--output-dir', type=str, default='outputs')
    parser.add_argument(
        '--dataset-train', type=str, default='data/controlled_text_reduction/train_set.jsonl',
        help='JSONL file containing train nputs and concatenated highlights.')
    parser.add_argument(
        '--dataset-val', type=str, default='data/controlled_text_reduction/dev_set.jsonl',
        help='JSONL file containing dev inputs and concatenated highlights.')

    # reward
    parser.add_argument(
        '--n_extra_tokens', type=int, default=5, help='number of reward categorization')
    parser.add_argument(
        '--sample-interval', type=int, default=500, help='step interval to sample from current policy')
    parser.add_argument(
        '--horizon', type=float, default=2500, help='horizon value in adaptive controller')
    # reward controlled Reduction
    parser.add_argument(
        '--R-P-rewards-alternate', action='store_true', default=False, help='whether to alternate between the Recall (R) and Precision (P) rewards')
    parser.add_argument(
        '--R-P-rewards-ratio', type=float, default=0.5, help='ratio of Recall reward to Precision reward (the reward will be (R_P_rewards_ratio)*R + (1-R_P_rewards_ratio)*P). applicable only if R_P_rewards_alternate=False')
    parser.add_argument(
        '--P-reward-type', type=str, default="unhighlights_F1", help='The approach to calculate the P (precision) reward when it is of \"concats\" type. structure:"<concat_type>_<rouge_type>". Possible options:\"unhighlights_F1\", \"unhighlights_precision\", \"unhighlights_recall\", \"highlights_F1\", \"highlights_precision\", \"highlights_recall\"')
    parser.add_argument(
        '--R-reward-type', type=str, default="highlights_F1", help='The approach to calculate the R (recall) reward when it is of \"concats\" type. structure:"<concat_type>_<rouge_type>". Possible options:\"unhighlights_F1\", \"unhighlights_precision\", \"unhighlights_recall\", \"highlights_F1\", \"highlights_precision\", \"highlights_recall\"')

    parser.add_argument(
        '--R-P-rewards-alternate-ratio', type=str, default="[1,1]", help='when alternating rewards, enables uneven alterations, with the first element being R\'s proportion and the second P\'s. For example, if passed [2,3], then for every 2 R rewards, there will be 3 P rewards.')
    parser.add_argument(
        '--reward-type', type=str, default="concats", help='type of reward. One of the following: \"concats\" (rouge compared to the concatenations of the highlights and the unhighlights), \"subspans\" (counting number of subspans of unhighlights and rouge-precision to subspans of highlights). Default: \"concats\"')

    # experiments
    


    # KL term
    parser.add_argument(
        '--kl_coef', type=float, default=0.05, help='coefficient for KL term in reward')
    parser.add_argument(
        '--adaptive_kl', action='store_true', default=False, help='whether to use adaptive KL controller')
    parser.add_argument(
        '--target_kl', type=float, default=3, help='target value in adaptive KL controller')
    # entropy term
    parser.add_argument(
        '--entropy_coef', type=float, default=0.06, help='coefficient for entropy term in reward')
    parser.add_argument(
        '--adaptive_entropy', action='store_true', default=False, help='whether to use adaptive entropy controller')
    parser.add_argument(
        '--target_entropy', type=float, default=40, help='target value in adaptive entropy controller')

    # policy
    parser.add_argument(
        '--init-model', type=str, default='gpt2-large', help='language model used for policy.')
    parser.add_argument(
        '--ref-model', type=str, default='gpt2-large', help='language model used for reference policy.')
    parser.add_argument(
        '--response-length', type=int, default=20, help='number of tokens to generate for each prompt.')
    parser.add_argument(
        '--temperature', type=float, default=1.0, help='temperature for sampling policy.')

    # training˚
    parser.add_argument(
        '--total-episodes', type=int, default=3000000, help='total number of episodes')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch size')
    parser.add_argument(
        '--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument(
        '--eps', type=float, default=1e-5, help='epsilon for optimizer')
    parser.add_argument(
        '--num_warmup_steps', type=int, default=500, help='number of warmup steps in lr scheduler')
    parser.add_argument(
        '--clip_grad', action='store_true', default=False, help='whether to clip gradient')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5, help='maximum norm of gradients ')

    # generation
    parser.add_argument(
        '--num-samples', type=int, default=25, help='number of samples to generate for each prompt.')
    parser.add_argument(
        '--top-p', type=float, default=1.0, help='hyperparameter for nucleus sampling')

    # other
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=100, help='step interval to print out logs')
    parser.add_argument(
        '--save-interval', type=int, default=1000, help='step interval to save model checkpoints')
    parser.add_argument(
        '--eval-interval', type=int, default=500, help='step interval to do evaluation')
    parser.add_argument(
        '--cuda-deterministic', action='store_false', default=True,
        help="sets flags for determinism when using CUDA (potentially slow!)")

    # model arguments
    parser.add_argument(
        '--cache_dir', type=str, default=None, help='Where to store the pretrained models downloaded from huggingface.com')
    parser.add_argument(
        '--use_fast_tokenizer', action='store_true', default=True, help='Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.')
    parser.add_argument(
        '--model_revision', type=str, default='main', help='The specific model version to use (can be a branch name, tag name or commit id).')
    parser.add_argument(
        '--use_auth_token', action='store_true', default=False, help="Will use the token generated when running `transformers-cli login` (necessary to use this script with private models).")
    parser.add_argument(
        '--resize_position_embeddings', type=bool, default=None, help="Whether to automatically resize the position embeddings if `max_source_length` exceeds the model's position embeddings.")
    parser.add_argument(
        '--freeze_embeds', type=bool, default=False)
    parser.add_argument(
        '--min_length', type=int, default=None)
    parser.add_argument(
        '--length_penalty', type=float, default=None)
    parser.add_argument(
        '--early_stopping', type=bool, default=False)
    parser.add_argument(
        '--no_repeat_ngram_size', type=int, default=None)
    parser.add_argument(
        '--local_radius', type=int, default=None)
    parser.add_argument(
        '--global_block_size', type=int, default=None)
    parser.add_argument(
        '--encoder_attention_type', type=int, default=None)
    parser.add_argument(
        '--resume_from_checkpoint', type=str, default=None, help="resume training from the checkpoint passed as parameter here (must pass at least the model_name/timestamp path, where the most recent checkpoint can be found)")
    parser.add_argument(
        '--load_prev_args', action='store_true', default=False, help="relevant when resuming from checkpoint. If true - will load the arguments from the previous run e.g. when the run has crashed)")


    # data training arguments
    parser.add_argument(
        '--overwrite_cache', action='store_true', default=False, help="Overwrite the cached training and evaluation sets")
    parser.add_argument(
        '--overwrite_output_dir', action='store_true', default=False, help="Overwrite the output directory")
    parser.add_argument(
        '--max_source_length', type=int, default=1024, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument(
        '--max_target_length', type=int, default=128, help="The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument(
        '--num_beams', type=int, default=1, help="Number of beams to use for evaluation. This argument will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.")
    parser.add_argument(
        '--save_total_limit', type=str, default="2", help="number of checkpoints to save (2 - will save last one and the best one). If want all ckpt - pass \"all\".")
    parser.add_argument(
        '--metric_for_best_model', type=str, default="combined_reward_eval", help="metric with which to compare two models. Can be any og the metrics in a saved ckpt in the \"scores.json\" file.")
    parser.add_argument(
        '--load_best_model_at_end', action='store_true', default=True, help="load the best model in the end - should be paired with the metric_for_best_model parameter")
    
    
    
    parser.add_argument(
        '--ignore_pad_token_for_loss', action='store_true', default=True, help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.")
    parser.add_argument(
        '--source_prefix', type=str, default=None, help="A prefix to add before every source text (useful for T5 models).")
    parser.add_argument(
        '--forced_bos_token', type=str, default=None, help="The token to force as the first generated token after the decoder_start_token_id. Useful for multilingual models like mBART where the first generated token needs to be the target language token (Usually it is the target language token)")
    parser.add_argument(
        '--add_global_attention', action='store_true', default=False)
    parser.add_argument(
        '--add_global_attention_on_highlights', action='store_true', default=False)
    parser.add_argument(
        '--should_preprocess_add_highlights', action='store_true', default=True, help="Decides whether to add highlight tokens or not")

    # wandb
    parser.add_argument(
        '--wandb_run_name', type=str, default=None, help="run name where results will be found in wandb")
    parser.add_argument(
        '--report_to_wandb', action='store_true', default=False)
    parser.add_argument(
        '--wandb_record_step', type=int, default=10, help="step interval to record losses to wandb")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return args
