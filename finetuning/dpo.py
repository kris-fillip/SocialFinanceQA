import argparse
import os
import torch

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from trl import DPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import bitsandbytes as bnb
from huggingface_hub import login

from setup_data import remove_unused_columns, preprocess_dpo, preprocess_dpo_zephyr

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    print(list(lora_module_names))
    return list(lora_module_names)


def return_prompt_and_responses(samples: dict[any]) -> dict[any]:
    return {
        "prompt": samples["prompt"],
        "chosen": samples["chosen"],
        "rejected": samples["rejected"],
    }


def fine_tune(model_name: str, train: Dataset, validation: Dataset) -> None:
    login()
    output_dir = "./llama_chat_sft_dpo_NEW"
    final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        cache_dir="model_cache"
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

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

    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        num_train_epochs=1,
        max_steps=-1,
        save_steps=100,
        eval_steps=100,
        learning_rate=5e-6,
        bf16=True,
        save_total_limit=5,
        logging_steps=10,
        output_dir=output_dir,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        dataloader_drop_last=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
        run_name="llama-chat-7b-sft-dpo-final_NEW",
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    peft_config = LoraConfig(
        r=128,
        lora_alpha=128,
        target_modules=find_all_linear_names(model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

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


def dpo(model_name: str, validation_size: int) -> None:
    data = load_dataset("Kris-Fillip/SocialFinanceQA", split="train")
    if "zephyr" in model_name:
        updated_dataset = data.map(preprocess_dpo_zephyr)
    else:
        updated_dataset = data.map(preprocess_dpo)
    updated_dataset = remove_unused_columns(updated_dataset)
    train_validation = updated_dataset.train_test_split(test_size=validation_size, shuffle=False)
    train = train_validation["train"]
    validation = train_validation["test"]
    fine_tune(model_name, train, validation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("sft")
    parser.add_argument("-m", "--model_name", help="Model name to use as base model for the finetuning",
                        type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("-val", "--validation_size", help="Size of the validation dataset", type=int, default=1000)
    args = parser.parse_args()
    dpo(model_name=args.model_name, validation_size=args.validation_size)