# Ask Me Again Differently: GRAS for Measuring Bias in Vision Language Models on Gender, Race, Age, and Skin Tone

<div align=center> 
<img src="./assets/method.png" alt="method image" width="800"/>
</div>

## Setup

1. (Optional) Creating conda environment

```
$ conda create -n GRAS
$ conda activate GRAS
```

2. Clone the repository
```
$ git clone <repository-url> && cd <repository-name>
```

3. Install the required dependencies:

```
$ pip install -r requirements.txt
```

### GRAS Image Dataset

1. Required Datasets
    - AI-Face Dataset: Download from https://github.com/Purdue-M2/AI-Face-FairnessBench
    - FairFace Dataset: Download from https://github.com/dchen236/FairFace

2. After downloading, organize datasets as follows:
    ```
    AlFace
    ├── imdb_wiki
    │   ├── wiki
    │   └── imdb
    └── FFHQ
    ```

    ```
    FairFace
    ├── train
    └── val
    ```

3. Create GRAS Image dataset

```
$ python makedir_grasimg.py 
```

_Note: The Monk Skin Tone Scale can be found here: https://skintone.google/get-started._

## Running Inference

### All Models at Once
```
$ inference.sh
```

### Individual Model Inference

#### paligemma2-3b-mix-224

```
$ cd inference_scripts
$ python paligemma_fairface.py --img_dir ../GRAS_DS --output_dir ../paligemma2-3b-mix-224_results
$ python paligemma_aiface.py --img_dir ../GRAS_DS --output_dir ../paligemma2-3b-mix-224_results
```

#### Qwen2.5-VL-3B-Instruct

```
$ cd inference_scripts
$ python qwen_fairface.py --img_dir ../GRAS_DS --output_dir ../Qwen2.5-VL-3B-Instruct_results
$ python qwen_aiface.py --img_dir ../GRAS_DS --output_dir ../Qwen2.5-VL-3B-Instruct_results
```

### blip2-opt-2.7b

```
$ cd inference_scripts
$ python blip_fairface.py --img_dir ../GRAS_DS --output_dir ../blip2-opt-2.7b_results
$ python blip_aiface.py --img_dir ../GRAS_DS --output_dir ../blip2-opt-2.7b_results
```

### llava-1.5-7b-hf

```
$ cd inference_scripts
$ python llava_fairface.py --img_dir ../GRAS_DS --output_dir ../llava-1.5-7b-hf_results
$ python llava_aiface.py --img_dir ../GRAS_DS --output_dir ../llava-1.5-7b-hf_results
```

### Phi-4-multimodal-instruct

```
$ cd inference_scripts
$ python phi_fairface.py --img_dir ../GRAS_DS --output_dir ../Phi-4-multimodal-instruct_results
$ python phi_aiface.py --img_dir ../GRAS_DS --output_dir ../Phi-4-multimodal-instruct_results
```

## Statistical Analysis 
Run bias analysis on the inference results:
```
$ cd evaluation_scripts && python bias_analysis.py
```

## Summary of Findings

We introduced the GRAS Benchmark, combining GRAS image dataset, 100 personality traits, 5 semantically equivalent question templates. Through 2.5M (image, question, trait) queries, we probe for bias on gender, race, age, and skin tone in VLMs, the broadest coverage to date. We presented GRAS Bias Score for interpretable bias quantification and showed that state of the art VLMs are highly biased, with no VLM scoring above 2 out of an unbiased ideal of 100.

| Model                     | GRAS Bias Score |
|---------------------------|-----------------|
| paligemma2-3b-mix-224     | 1.75            |
| llava-1.5-7b-hf           | 2.00            |
| Qwen2.5-VL-3B-Instruct    | 1.00            |
| blip2-opt-2.7             | 0.25            |
| Phi-4-multimodal-instruct | 0.00            |