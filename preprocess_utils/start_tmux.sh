csv_path=$1
outdir=$2
split=$3 
cuda_visible_device=$4
session="generate_data_for_quark_${split}"
only_concats="--only-concats"
config="--csv-path ${csv_path} --outdir ${outdir} --remove-unhighlights-duplicates ${only_concats}"
command="conda activate controlled_reduction_37 && cd ~/controlled_reduction/DL_approach/Quark/preprocess_utils && CUDA_VISIBLE_DEVICES=${cuda_visible_device} python csv_to_jsonl.py ${config}"


tmux new-session -d -s $session
tmux send-keys -t $session "$command" C-m
tmux attach-session -t $session

