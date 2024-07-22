# Project Setup and Execution Flow

## 1. Run Flow
<span style="color:red;">这是红色的文字</span>

- **step 0: Environment Configuration**

  Follow the instructions provided by CCEVAL to configure the environment.  
[Reference Link](https://github.com/amazon-science/cceval)

- **step 1: `/retrieval/1_classify_lang.py`** 

    Categorize the source code repositories into their respective folders based on the programming language.

- **step 2: `/retrieval/2_original_chunk.py`**

    Chunk source code.

- **step 3: `/retrieval/3_weight_retrieval.py`**

  Retrieve similar code chunks.

## 2. Our Retrieval Data
Download the data from the following link: [Download Link](https://drive.google.com/file/d/17BjcCjYdzvN6-Ylr0AezC9m2jsvlvh4j/view?usp=sharing).  
After downloading, extract the files into the `\data` directory.

## 3. Running Results Displayed on Colab

We show the running results of using the weighted retrieval strategy and the text-embedding-3-large model on Colab (Programming language: Java). You can visually observe the execution process of our code.

- **text-embedding-3-large**: [View on Colab](https://colab.research.google.com/drive/1z-aLTbm-DkYa7SCKD_LnpwMz0WRzIdJB?usp=sharing)
- **Weighted retrieval strategy**: [View on Colab](https://colab.research.google.com/drive/1z7CsgzaNkrGhs2GdzHH2znGvqplQorsC?usp=sharing)

You can change the parameters in the Colab, such as language and retrieval strategy, to see the results under different settings.

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
