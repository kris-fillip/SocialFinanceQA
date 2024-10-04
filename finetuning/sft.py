import argparse
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer
import transformers
from huggingface_hub import login
from tqdm import tqdm

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example["text"])
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(example["text"]).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(example["text"]))

    return total_characters / total_tokens

def fine_tune(model_name, train_data, validation_data):
    print(f"using {len(train_data)} train samples and {len(validation_data)} validation samples")
    login()
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    # Load the entire model on the GPU 0
    device_map = {"": 0}

    # Set base model loading in 4-bits
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = torch.bfloat16

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True, add_eos_token=True)

    # # Define a custom padding token
    tokenizer.add_special_tokens({"pad_token":"<PAD>"})

    # Set the padding direction to the right
    tokenizer.padding_side = "right"

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)
    
    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    # LoRA attention dimension
    lora_r = 16
    # Alpha for LoRA scaling
    lora_alpha = 32
    # Dropout probability for LoRA
    lora_dropout = 0.05

    # Create the LoRA configuration
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "gate_proj",
            "down_proj",
            "up_proj",
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
        ]
    )

    output_dir = "./results_7b_mistral_sft_no_packing_31_01_24"
    final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")

    num_train_epochs = 1
    max_steps = -1
    bf16 = True
    fp16 = False
    batch_size = 4
    gradient_accumulation_steps = 16
    max_grad_norm = 0.3
    optim = "paged_adamw_8bit"
    learning_rate = 1e-4
    lr_scheduler_type = "cosine"
    warmup_ratio = 0.05
    weight_decay = 0.05
    gradient_checkpointing = True
    save_steps = 100
    logging_steps = 10
    eval_steps = 100
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        run_name="llama-7b-zephyr_no_packing",
        report_to="wandb",
        ddp_find_unused_parameters=False,
        save_total_limit=5,
        load_best_model_at_end=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
    )

    max_seq_length = 1024
    packing = False

    # Set the supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=validation_data,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
        max_seq_length=max_seq_length,
        chars_per_token=chars_per_token,
    )

    transformers.logging.set_verbosity_info()
    trainer.train()

    trainer.save_model(final_checkpoint_dir)

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
    reloaded_tokenizer = AutoTokenizer.from_pretrained(final_checkpoint_dir, add_eos_token=True, use_fast=True)

    # Merge the LoRA and the base model
    merged_model = reloaded_model.merge_and_unload()
    # Save the merged model
    merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
    merged_model.save_pretrained(merged_dir)
    reloaded_tokenizer.save_pretrained(merged_dir)


def remove_unused_columns(data: Dataset):
    all_columns = data.column_names
    columns_to_keep = ["text", "prompt", "chosen", "rejected"]
    for col in columns_to_keep:
        if col in all_columns:
            all_columns.remove(col)
    data = data.remove_columns(column_names=all_columns)
    return data


def preprocess_sft(example):
    example["text"] = f"[INST] {example['text']} {example['context']} [/INST] {example['answer_1']}"
    return example

def preprocess_sft_zephyr(example):
    example["text"] = f"<|system|>\n</s>\n<|user|>\n{example['text']} {example['context']}</s>\n<|assistant|>\n{example['answer_1']}</s>\n"
    return example

def preprocess_dpo(example):
    example["prompt"] = f"[INST] {example['text']} {example['context']} [/INST]"
    example["chosen"] = example['answer_1']
    example["rejected"] = example['answer_2']
    return example

def preprocess_dpo_zephyr(example):
    example["prompt"] = f"<|system|>\n</s>\n<|user|>\n{example['text']} {example['context']}</s>\n",
    example["chosen"] = f"<|assistant|>\n{example['answer_1']}</s>\n",
    example["rejected"] = f"<|assistant|>\n{example['answer_2']}</s>\n",
    return example

def sft(model_name: str) -> None:
    data = load_dataset("Kris-Fillip/SocialFinanceQA", split="train")
    updated_dataset = data.map(preprocess_sft)
    updated_dataset = remove_unused_columns(updated_dataset)
    train_validation = updated_dataset.train_test_split(test_size=1000, shuffle=False)
    train = train_validation["train"]
    validation = train_validation["test"]
    fine_tune(model_name, train, validation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sft")
    parser.add_argument("-m", "--model_name", help="Model name to use as base model for the finetuning",
                        type=str, default="meta-llama/Llama-2-7b-hf")
    args = parser.parse_args()
    sft(model_name=args.model_name)
