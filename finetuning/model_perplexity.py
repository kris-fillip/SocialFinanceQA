import argparse
import evaluate
import os
import json
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
import pandas as pd
import statistics

from setup_data import preprocess_perplexities

def model_perplexity(dataset_name: str, model_names: list[str]) -> None:
    login()
    # Load the Perplexity evaluation metric
    perplexity = evaluate.load('./pipeline/fintuning/perplexity', module_type="metric")
    overall_result = {
        "models": [],
        "good_answers_perplexity_mean": [],
        "good_answers_perplexity_median": [],
        "bad_answers_perplexity_mean": [],
        "bad_answers_perplexity_median": []
    }
    data = load_dataset(dataset_name, split="train")
    data = data.select(range(len(data) - 1000, len(data)))
    data = data.map(preprocess_perplexities)
    good_answers = data["good_answer"]
    bad_answers = data["bad_answer"]

    for model_name in tqdm(model_names, total=len(model_names)):
        good_answers_results = perplexity.compute(predictions=good_answers, model_id=model_name, batch_size=2, max_length=1024)
        bad_answers_results = perplexity.compute(predictions=bad_answers, model_id=model_name, batch_size=2, max_length=1024)
        overall_result["models"].append(model_name)
        overall_result["good_answers_perplexity_mean"].append(good_answers_results["mean_perplexity"])
        overall_result["good_answers_perplexity_median"].append(statistics.median(good_answers_results["perplexities"]))
        overall_result["bad_answers_perplexity_mean"].append(bad_answers_results["mean_perplexity"])
        overall_result["bad_answers_perplexity_median"].append(statistics.median(bad_answers_results["perplexities"]))

    result_df = pd.DataFrame(overall_result)
    result_df.to_csv("./data/perplexity_results/perplexity_overall_complete.csv", sep=";")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("model_perplexity")
    parser.add_argument("-d", "--dataset_name", help="Data used for the inference",
                        type=str, default="Kris-Fillip/SocialFinanceQA")
    parser.add_argument("-m", "--model_names", nargs="+", help="Model names to calculate perplexity for",
                        type=str, default=["meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-7b-chat-hf", "HuggingFaceH4/zephyr-7b-beta"])
    args = parser.parse_args()
    model_perplexity(dataset_name=args.dataset_name, model_dir_or_name=args.model_dir_or_name)