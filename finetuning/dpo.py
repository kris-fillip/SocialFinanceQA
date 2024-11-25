import argparse
import os
import torch
import transformers
from datasets import Dataset, load_dataset
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer

from config.settings import LORA_DPO, TRAINING_ARGUMENTS_DPO
from setup_data import preprocess_dpo, preprocess_dpo_zephyr, remove_unused_columns
from utils.utils import find_all_linear_names, print_trainable_parameters, return_prompt_and_responses


def fine_tune(model_name: str, train: Dataset, validation: Dataset) -> None:
    print(
        f"using {len(train)} train samples and {len(validation)} validation samples")
    login()
    # Load the entire model on the GPU 0
    device_map = {"": 0}
    output_dir = f"./{model_name}_dpo"
    final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        cache_dir="model_cache"
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # Load the model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Define a custom padding token
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # Set the padding direction to the right
    tokenizer.padding_side = "right"

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)

    print("Formatting data..")
    original_train_columns = train.column_names

    train_dataset = train.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_train_columns
    )

    original_validation_columns = validation.column_names

    validation_dataset = validation.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_validation_columns
    )

    LORA_DPO["target_modules"] = find_all_linear_names(model)
    peft_config = LoraConfig(**LORA_DPO)

    TRAINING_ARGUMENTS_DPO["run_name"] = output_dir
    TRAINING_ARGUMENTS_DPO["output_dir"] = output_dir
    training_args = TrainingArguments(**TRAINING_ARGUMENTS_DPO)

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        beta=0.01,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=512,
        max_length=1024,

    )
    print_trainable_parameters(model)

    transformers.logging.set_verbosity_info()
    print("Starting training..")
    dpo_trainer.train()

    dpo_trainer.save_model(final_checkpoint_dir)
    print("Saved model!")

    # Load the entire model on the GPU 0
    device_map = {"": 0}
    reloaded_model = AutoPeftModelForCausalLM.from_pretrained(
        final_checkpoint_dir,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        cache_dir="model_cache"
    )
    reloaded_tokenizer = AutoTokenizer.from_pretrained(
        final_checkpoint_dir, add_eos_token=True, use_fast=True)
    print("Reloaded Model!")
    # Merge the LoRA and the base model
    merged_model = reloaded_model.merge_and_unload()
    # Save the merged model
    merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
    merged_model.save_pretrained(merged_dir)
    reloaded_tokenizer.save_pretrained(merged_dir)
    print("Saved Merged checkpoint!")


def dpo(dataset_name: str, model_name: str, validation_size: int) -> None:
    data = load_dataset(dataset_name, split="train")
    if "zephyr" in model_name:
        updated_dataset = data.map(preprocess_dpo_zephyr)
    else:
        updated_dataset = data.map(preprocess_dpo)
    updated_dataset = remove_unused_columns(
        updated_dataset, ["prompt", "chosen", "rejected"])
    train_validation = updated_dataset.train_test_split(
        test_size=validation_size, shuffle=False)
    train = train_validation["train"]
    validation = train_validation["test"]
    fine_tune(model_name, train, validation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("dpo")
    parser.add_argument("-d", "--dataset_name", help="Data used for the finetuning",
                        type=str, default="Kris-Fillip/SocialFinanceQA")
    parser.add_argument("-m", "--model_name", help="Model name to use as base model for the finetuning",
                        type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("-val", "--validation_size",
                        help="Size of the validation dataset", type=int, default=1000)
    args = parser.parse_args()
    dpo(dataset_name=args.dataset_name, model_name=args.model_name,
        validation_size=args.validation_size)
