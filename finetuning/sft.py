import argparse
import os
import torch
import transformers
from datasets import Dataset, load_dataset
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from config.settings import LORA_SFT, TRAINING_ARGUMENTS_SFT
from setup_data import remove_unused_columns, preprocess_sft, preprocess_sft_zephyr
from utils.utils import chars_token_ratio, find_all_linear_names, print_trainable_parameters


def fine_tune(model_name: str, train: Dataset, validation: Dataset) -> None:
    print(
        f"using {len(train)} train samples and {len(validation)} validation samples")
    login()
    # Load the entire model on the GPU 0
    device_map = {"": 0}
    output_dir = f"./{model_name}_sft"
    final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False
    )
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=bnb_config,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        cache_dir="model_cache",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load the model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=True, add_eos_token=True)

    # Define a custom padding token
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # Set the padding direction to the right
    tokenizer.padding_side = "right"

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)

    chars_per_token = chars_token_ratio(train, tokenizer)
    print(
        f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    LORA_SFT["target_modules"] = find_all_linear_names(model)
    peft_config = LoraConfig(**LORA_SFT)

    TRAINING_ARGUMENTS_SFT["run_name"] = output_dir
    TRAINING_ARGUMENTS_SFT["output_dir"] = output_dir
    training_args = TrainingArguments(**TRAINING_ARGUMENTS_SFT)

    # Set the supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train,
        eval_dataset=validation,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        max_seq_length=1024,
        chars_per_token=chars_per_token,
    )
    print_trainable_parameters(model)

    transformers.logging.set_verbosity_info()
    print("Starting training..")
    trainer.train()

    trainer.save_model(final_checkpoint_dir)
    print("Saved model!")

    # Load the entire model on the GPU
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


def sft(dataset_name: str, model_name: str, validation_size: int) -> None:
    data = load_dataset(dataset_name, split="train")
    if "zephyr" in model_name:
        updated_dataset = data.map(preprocess_sft_zephyr)
    else:
        updated_dataset = data.map(preprocess_sft)
    updated_dataset = remove_unused_columns(updated_dataset, ["text"])
    train_validation = updated_dataset.train_test_split(
        test_size=validation_size, shuffle=False)
    train = train_validation["train"]
    validation = train_validation["test"]
    fine_tune(model_name, train, validation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sft")
    parser.add_argument("-d", "--dataset_name", help="Data used for the finetuning",
                        type=str, default="Kris-Fillip/SocialFinanceQA")
    parser.add_argument("-m", "--model_name", help="Model name to use as base model for the finetuning",
                        type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("-val", "--validation_size",
                        help="Size of the validation dataset", type=int, default=1000)
    args = parser.parse_args()
    sft(dataset_name=args.dataset_name, model_name=args.model_name,
        validation_size=args.validation_size)
