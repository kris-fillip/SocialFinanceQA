# LLMs Cannot (Yet) Match the Specificity and Simplicity of Online Finance Communities in Long Form Question Answering

This is the official repository for the paper "LLMs Cannot (Yet) Match the Specificity and Simplicity of Online Finance Communities in Long Form Question Answering".

In this work, we curate a QA preference dataset called SocialFinanceQA for fine-tuning and aligning LLMs extracted from more than 7.4 million submissions and 82 million comments in Reddit’s 15 largest finance communities from 2008 to 2022.

We propose the novel framework SocialQA-EVAL as a generally applicable method to evaluate generated QA responses.

We evaluate various LLMs fine-tuned on this dataset using traditional metrics, LLM-based evaluation, and human annotation.

Our results demonstrate the value of high-quality Reddit data, with even state-of-the-art LLMs improving on producing simpler and more specific responses.

## Setup
### Preliminaries

The overall Docker setup in this project is based on the [nlp-research-template](https://github.com/konstantinjdobler/nlp-research-template).

Within this project, we are using [`mamba`](https://github.com/mamba-org/mamba) to manage dependencies.
We use `conda-lock` to generate lock files to provide reproducible environments.

<details><summary>Installing <code>mamba</code></summary>

<p>

On Unix-like platforms, run the snippet below. Otherwise, visit the [mambaforge repo](https://github.com/conda-forge/miniforge#mambaforge).
Note this does not use the Anaconda installer, which reduces bloat.

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

</details>

<details><summary>Installing <code>conda-lock</code></summary>

<p>

The preferred method is to install `conda-lock` using `pipx install conda-lock`. For other options, visit the [conda-lock repo](https://github.com/conda/conda-lock). For basic usage, have a look at the commands below:

```bash
conda-lock install --name gpt5 conda-lock.yml # create environment with name gpt5 based on lock file
conda-lock # create a new lock file based on environment.yml
conda-lock --update <package-name> # update specific packages in lock file
```

</details>

### Environment

After having installed `mamba` and `conda-lock`, you can create a `mamba` environment named `<env_name>` from a lock file with all necessary dependencies installed like this:

```bash
conda-lock install --name env_name conda-lock.yml
```

You can then activate your environment with

```bash
mamba activate env_name
```

### Docker

1. If you have not yet installed Docker, please install it under [Docker](https://docs.docker.com/get-started/get-docker/).
1. Run ```docker build -f Dockerfile -t <username>/<imagename>:<tag> .``` to build the overall project.
1. Adapt the docker project name in docker_script.sh accordingly.
1. Navigate into the `/hatebert` folder and run ```docker build -f Dockerfile -t <username>/<imagename>:<tag> .``` to build the hatebert project (This is necessary because the project has conflicting dependencies with our project).
1. In the respective folders run ```bash docker_script python <path_to_file>/<file_name>.py``` to execute the scripts.

## Preprocessing

Within the preprocessing folder, we curate SocialFinanceQA, a preference dataset for Long Form Question Answering extracted from an extensive collection of finance-related subreddits.
To make use of the data curation pipeline, you have to initially populate the `./data/subreddits` folder with submission and comment extracts in the Pushshift format and, in case of curation for other subreddits, adapt the subreddit selection in `./config/constants.py`.

The initial preprocessing is split across the following components:

1. `analysis.py` creates data aggregates for relevant variables (enabling percentile filtering in the following step).
1. `filtering.py` filters data based on heuristics, capturing engagement, community approval, and relevance.
1. `preference_matching.py` combines submissions and comments into a preference dataset.
1. `merge_data.py` combines individual subreddits into an overall dataset.

If you want to run the whole pipeline up to this point, you can also run `orchestration.py`.

### Toxicity Filtering

To account for toxicity in the selected data, we employ [HateBERT](https://arxiv.org/pdf/2010.12472) to filter offensive speech in the preferred answers of our preference dataset.
Before running the classifier, download the [HateBERT_offenseval](https://osf.io/tbd58/) model.
Then, navigate into the `./hatebert` folder and run `hatebert_orchestration.py` to classify offensive answers and filter them from the dataset.

To further remove toxicity from the data, we employ yet another hate classifier in `preprocessing/hate_classification.py` and filter the detected offensive speech and too-long content from our dataset in `hate_and_length_filtering.py` to arrive at our final curated dataset.

Additionally, we provide scripts to investigate characteristic words and topic models in the initial and curated data.

## Finetuning

Our curated [SocialFinanceQA](https://huggingface.co/datasets/Kris-Fillip/SocialFinanceQA) dataset is available on huggingface.

We use this dataset to run both supervised finetuning (`sft.py`) as well as direct preference optimization (`dpo.py`) on a variety of models such as `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-7b-hf`, `mistralai/Mistral-7B-v0.1` and `HuggingFaceH4/zephyr-7b-beta`.

## Evaluation

We created the evaluation framework SocialQA-Eval and use it to evaluate the inference results of the base and fine-tuned versions of our considered models across different dimensions, such as relevance, specificity, simplicity, helpfulness, and objectivity, using different LLMs, such as Gemini Pro and GPT-4o.

If you want to use Gemini for the evaluation, you can set up a Google Cloud project by following this [documentation](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).

As of the creation of the repository, you can still get free limited access to the Vertex AI API through Google Cloud for 90 days.

If you want to use OpenAI models for the evaluation, follow the setup instructions [here](https://platform.openai.com/docs/quickstart).

Additionally, we provide traditional evaluation measures like similarity and textstat metrics.

The evaluation results are available [here](https://aclanthology.org/2024.findings-emnlp.111.pdf).

## Citation

Please cite our work as:

```bibtex
@inproceedings{kahl-etal-2024-llms,
    title = "{LLM}s Cannot (Yet) Match the Specificity and Simplicity of Online Communities in Long Form Question Answering",
    author = "Kahl, Kris-Fillip  and
      Buz, Tolga  and
      Biswas, Russa  and
      De Melo, Gerard",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.111",
    pages = "2028--2053",
    abstract = "Retail investing is on the rise, and a growing number of users is relying on online finance communities to educate themselves.However, recent years have positioned Large Language Models (LLMs) as powerful question answering (QA) tools, shifting users away from interacting in communities towards discourse with AI-driven conversational interfaces.These AI tools are currently limited by the availability of labelled data containing domain-specific financial knowledge.Therefore, in this work, we curate a QA preference dataset SocialFinanceQA for fine-tuning and aligning LLMs, extracted from more than 7.4 million submissions and 82 million comments from 2008 to 2022 in Reddit{'}s 15 largest finance communities. Additionally, we propose a novel framework called SocialQA-Eval as a generally-applicable method to evaluate generated QA responses.We evaluate various LLMs fine-tuned on this dataset, using traditional metrics, LLM-based evaluation, and human annotation. Our results demonstrate the value of high-quality Reddit data, with even state-of-the-art LLMs improving on producing simpler and more specific responses.",
}
```