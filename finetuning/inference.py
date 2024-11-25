import argparse
import json
import torch
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

from setup_data import preprocess_dpo, preprocess_dpo_zephyr, preprocess_sft, preprocess_sft_zephyr, remove_unused_columns


def inference(model_dir_or_name: str, dataset_name: str) -> None:
    login()
    # Load the entire model on the GPU 0
    device_map = {"": 0}
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_dir_or_name,
        device_map=device_map,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        cache_dir="model_cache",
    )
    model.config.use_cache = True
    model.config.pretraining_tp = 1

    if model_dir_or_name in ["meta-llama/Llama-2-7b", "mistralai/Mistral-7B-v0.1"]:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir_or_name, trust_remote_code=True, use_fast=True, add_eos_token=True)
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir_or_name, trust_remote_code=True, use_fast=True)

    # # Set the padding direction to the right in case it isn't yet
    tokenizer.padding_side = "right"

    reloaded_generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1000,
                                  return_full_text=False, clean_up_tokenization_spaces=True, repetition_penalty=1.2)

    data = load_dataset(dataset_name, split="test")
    if "sft" in model_dir_or_name:
        if "zephyr" in model_dir_or_name:
            updated_dataset = data.map(preprocess_sft_zephyr, inference=True)
        else:
            updated_dataset = data.map(preprocess_sft, inference=True)
    elif "dpo" in model_dir_or_name:
        if "zephyr" in model_dir_or_name:
            updated_dataset = data.map(preprocess_dpo_zephyr, inference=True)
        else:
            updated_dataset = data.map(preprocess_dpo, inference=True)
    updated_dataset = remove_unused_columns(
        updated_dataset, ["templated_question", "question", "answer_1", "answer_2"])

    final_result = []
    for el in tqdm(updated_dataset, total=len(updated_dataset)):
        response = reloaded_generator(el["templated_question"])
        response_text = response[0]["generated_text"]
        if "zephyr" in model_dir_or_name:
            potential_starting_strings = [
                " ", "\n", "<", "|", "assistant", "|", ">", "\n"]
            for prefix in potential_starting_strings:
                response_text = response_text.removeprefix(prefix)
        cur_result = {
            "question": el["question"],
            "prediction": response_text,
            "answer_1": el["answer_1"],
            "answer_2": el["answer_2"]
        }
        final_result.append(cur_result)

    with open(f"./data/inference_results/{model_dir_or_name}", "w") as f:
        json.dump(final_result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sft")
    parser.add_argument("-d", "--dataset_name", help="Data used for the inference",
                        type=str, default="Kris-Fillip/SocialFinanceQA")
    parser.add_argument("-m", "--model_dir_or_name", help="Model name to use as base model for the inference",
                        type=str, default="meta-llama/Llama-2-7b-hf")
    args = parser.parse_args()
    inference(dataset_name=args.dataset_name,
              model_dir_or_name=args.model_dir_or_name)
