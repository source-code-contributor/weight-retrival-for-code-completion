---

# Project Setup and Execution Flow

## Run Flow

### 0. Environment Configuration
Follow the instructions provided by CCEVAL to configure the environment.  
[Reference Link](https://github.com/amazon-science/cceval)

### 1. `/retrieval/1_classify_lang.py`
- **Purpose**: Categorize the source code repositories into their respective folders based on the programming language.

### 2. `/retrieval/2_original_chunk.py`
- **Purpose**: Chunk source code.

### 3. `/retrieval/3_weight_retrieval.py`
- **Purpose**: Retrieve data.

## Our Retrieval Data
Download the data from the following link: [Download Link](https://drive.google.com/file/d/17BjcCjYdzvN6-Ylr0AezC9m2jsvlvh4j/view?usp=sharing).  
After downloading, extract the files into the `\data` directory.

## Acknowledgements
We are very grateful to CCEVAL for providing the original repository data and source code. Our work is based on their contributions.  
[CCEVAL Repository](https://github.com/amazon-science/cceval)

---

### Generation

```bash
export gpus=1
export model=bigcode/starcoderbase-1b
export crossfile_max_tokens=4096
export language=typescript
export task=typescript_Semantic_Model
export output_dir=./tmp/crosscodeeval_testrun/
python scripts/vllm_inference.py \
  --tp $gpus \
  --task $task \
  --language $language \
  --model $model \
  --output_dir $output_dir \
  --use_crossfile_context \
  --crossfile_max_tokens  $crossfile_max_tokens
```


### Metrics Calculation
After obtaining the generation, we can calculate the final metrics
```bash
export language=typescript
export ts_lib=./build/${language}-lang-parser.so; 
export task=typescript_Semantic_Model
export prompt_file=./data/${language}/${task}.jsonl 
export output_dir=./tmp/crosscodeeval_testrun/;  
python scripts/eval.py \
  --prompt_file $prompt_file \
  --output_dir $output_dir \
  --ts_lib $ts_lib \
  --language $language \
  --only_compute_metric
```