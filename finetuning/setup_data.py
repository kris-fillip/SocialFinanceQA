from datasets import Dataset


def remove_unused_columns(data: Dataset) -> Dataset:
    all_columns = data.column_names
    columns_to_keep = ["text", "prompt", "chosen", "rejected"]
    for col in columns_to_keep:
        if col in all_columns:
            all_columns.remove(col)
    data = data.remove_columns(column_names=all_columns)
    return data

def preprocess_sft(example: dict[any]) -> dict[any]:
    example["text"] = f"[INST] {example['text']} {example['context']} [/INST] {example['answer_1']}"
    return example

def preprocess_sft_zephyr(example: dict[any]) -> dict[any]:
    example["text"] = f"<|system|>\n</s>\n<|user|>\n{example['text']} {example['context']}</s>\n<|assistant|>\n{example['answer_1']}</s>\n"
    return example

def preprocess_dpo(example: dict[any]) -> dict[any]:
    example["prompt"] = f"[INST] {example['text']} {example['context']} [/INST]"
    example["chosen"] = example['answer_1']
    example["rejected"] = example['answer_2']
    return example

def preprocess_dpo_zephyr(example: dict[any]) -> dict[any]:
    example["prompt"] = f"<|system|>\n</s>\n<|user|>\n{example['text']} {example['context']}</s>\n",
    example["chosen"] = f"<|assistant|>\n{example['answer_1']}</s>\n",
    example["rejected"] = f"<|assistant|>\n{example['answer_2']}</s>\n",
    return example