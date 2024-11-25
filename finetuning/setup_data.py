from datasets import Dataset


def remove_unused_columns(data: Dataset, columns_to_keep: list[str]) -> Dataset:
    all_columns = data.column_names
    for col in columns_to_keep:
        if col in all_columns:
            all_columns.remove(col)
    data = data.remove_columns(column_names=all_columns)
    return data


def preprocess_sft(example: dict, inference=False) -> dict:
    if inference:
        example["templated_question"] = f"[INST] {example['text']} {example['context']} [/INST]"
        example["question"] = f"{example['text']} {example['context']}"
    else:
        example["text"] = f"[INST] {example['text']} {example['context']} [/INST] {example['answer_1']}"
    return example


def preprocess_sft_zephyr(example: dict, inference=False) -> dict:
    if inference:
        example["templated_question"] = f"<|system|>\n</s>\n<|user|>\n{example['text']} {example['context']}</s>\n<|assistant|>\n"
        example["question"] = f"{example['text']} {example['context']}"
    else:
        example["text"] = f"<|system|>\n</s>\n<|user|>\n{example['text']} {example['context']}</s>\n<|assistant|>\n{example['answer_1']}</s>\n"
    return example


def preprocess_dpo(example: dict, inference=False) -> dict:
    if inference:
        example["templated_question"] = f"[INST] {example['text']} {example['context']} [/INST]"
        example["question"] = f"{example['text']} {example['context']}"
    else:
        example["prompt"] = f"[INST] {example['text']} {example['context']} [/INST]"
        example["chosen"] = example['answer_1']
        example["rejected"] = example['answer_2']
    return example


def preprocess_dpo_zephyr(example: dict, inference=False) -> dict:
    if inference:
        example["templated_question"] = f"<|system|>\n</s>\n<|user|>\n{example['text']} {example['context']}</s>\n",
        example["question"] = f"{example['text']} {example['context']}"
    else:
        example["prompt"] = f"<|system|>\n</s>\n<|user|>\n{example['text']} {example['context']}</s>\n",
        example["chosen"] = f"<|assistant|>\n{example['answer_1']}</s>\n",
        example["rejected"] = f"<|assistant|>\n{example['answer_2']}</s>\n",
    return example

def preprocess_perplexities(example: dict) -> dict:
    example["good_answer"] = f"[INST] {example['text']} {example['context']} [/INST] {example['answer_1']}"
    example["bad_answer"] = f"[INST] {example['text']} {example['context']} [/INST] {example['answer_2']}"
    return example