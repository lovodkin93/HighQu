# HighQu Pre-processing
In order to pre-process the data, you need to first replace ``prefix`` in [preprocess_environment.yml](preprocess_environment.yml) with your home path.
With conda installed, create an environment called `HighQu_preprocess` with:
```
conda env create -f preprocess_environment.yml
```

Now, to perform the pre-processing, follow those steps:
1. Run an initial pre-processing step by following the pre-processing described in [Controlled_Text_Reduction](https://github.com/lovodkin93/Controlled_Text_Reduction). This should generate a csv file with the following columns: `doc_text`,`summary_text`,`highlight_spans`.
2. Next, run another pre-processing step on this generated csv file by running:
```
conda activate HighQu_preprocess
python csv_to_jsonl.py --csv-path /path/to/csv_file --outdir /path/to/outdir --only-concats
```
Where `/path/to/csv_file` should point to the csv dile generated in step 1, and `/path/to/outdir` should point to the out_file, which should be with `jsonl` extension.
