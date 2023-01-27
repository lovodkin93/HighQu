# HighQu

## Dataset Pre-processing
The first step is to pre-process the data. 
To do that, follow the steps described in [preprocess_utils/README.md](preprocess_utils/README.md).

## Requirements
We suggest using conda to setup environment. You need to first replace ``prefix`` in [environment.yml](environment.yml) with your home path. With conda installed, create an environment called `HighQu` with:
```
conda env create -f environment.yml
```

## Train

For training HighQu with default hyperparameters, run:
```
conda activate HighQu
python main.py
```

## Evaluate

To evaluate HighQu on the devset and test, run:
```
conda activate HighQu
python sample.py --dataset-test data/controlled_text_reduction/test_set.jsonl --dataset-dev  data/controlled_text_reduction/dev_set.jsonl --saved-model-path /path/to/ckpt
```
where /path/to/ckpt should point to the checkpoint that is evaluated. Additionally, you can evaluate only on the dev/test sets, by only passing the corresponding flag.

## Multi-GPU training
HighQu also supports multi-GPU training.
To run a multi-GPU training, first run:
```
conda activate HighQu
pip install accelerate
```
Then, follow the instructions in [accelerate](https://github.com/huggingface/accelerate) and configure your `default_config.yaml` file to look as follows:
```
command_file: null
commands: null
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: MULTI_GPU
downcast_bf16: 'no'
fsdp_config: {}
gpu_ids: <available_GPUs>
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
megatron_lm_config: {}
mixed_precision: fp16
num_machines: 1
num_processes: <number_of_available_GPUs>
rdzv_backend: static
same_network: true
tpu_name: null
tpu_zone: null
use_cpu: false
```
where `<available_GPUs>` should be replaced by a comma-delimited list of the avaliable GPUs (for example:`gpu_ids: 0,1,2`) and `<number_of_available_GPUs>` should be accordingly replaced, namely it should account for the total number of available GPUs (following the previous example: `num_processes: 3`).

Then, when the `default_config.yaml` is set, to run a multi GPU training of HighQu with default hyperparameters, run: 
```
accelerate launch main.py
```