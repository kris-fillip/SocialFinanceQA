
import json
from tqdm import tqdm
from huggingface_hub import login
from transformers import AutoTokenizer

from config.constants import MAX_TOKENIZED_LENGTH, TEST_SIZE


def tokenized_input_fits(el: dict, tokenizers: list[AutoTokenizer], max_length: int) -> bool:
    qa_good = f"[INST] {el['text']} {el['context']} [/INST] {el['answer_1']}"
    qa_bad = f"[INST] {el['text']} {el['context']} [/INST] {el['answer_2']}"
    for tokenizer in tokenizers:
        qa_good_tokenized = tokenizer(qa_good)
        if len(qa_good_tokenized["input_ids"]) > max_length:
            return False

        qa_bad_tokenized = tokenizer(qa_bad)
        if len(qa_bad_tokenized["input_ids"]) > max_length:
            return False
    return True


def filter_hate_and_length(dataset: list[dict], tokenizers: list[AutoTokenizer], max_length: int, test_size: int) -> None:
    with open("./data/hate_classification_results/hate_classifier_results.json", "r") as inputfile:
        hate_classified = json.load(inputfile)

    hate_classified = [el["idx"]
                       for el in hate_classified if el["reason"] == "OFFENSIVE-LANGUAGE"]
    print(f"Initial Dataset has {len(dataset)} entries")
    final_dataset = []
    for idx, el in tqdm(enumerate(dataset), total=len(dataset)):
        if idx in hate_classified:
            continue
        if not tokenized_input_fits(el, tokenizers, max_length):
            continue
        final_dataset.append(el)

    dataset_length = len(final_dataset)
    print(
        f"Dataset with tokenized tuples smaller than {max_length} has {dataset_length} entries")

    with open("./data/final_dataset/final_filtered_data.json", "w") as output_file:
        json.dump(final_dataset, output_file)

    train_dataset = final_dataset[:len(final_dataset) - test_size]
    test_dataset = final_dataset[len(final_dataset) - test_size:]
    with open("./data/final_dataset/final_train.json", "w") as output_file:
        json.dump(train_dataset, output_file)

    with open("./data/final_dataset/final_test.json", "w") as output_file:
        json.dump(test_dataset, output_file)


def hate_and_length_filtering() -> None:
    login()
    with open("./data/hatebert_results/complete_qa_hatebert.json", "r") as f:
        data = json.load(f)

    llama_base_tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", use_fast=True, revision="8cca527612d856d7d32bd94f8103728d614eb852")
    llama_chat_tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", use_fast=True, revision="c1b0db933684edbfe29a06fa47eb19cc48025e93")
    mistral_tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1", use_fast=True, revision="26bca36bde8333b5d7f72e9ed20ccda6a618af24")
    zephyr_tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta", use_fast=True, revision="b70e0c9a2d9e14bd1e812d3c398e5f313e93b473")
    tokenizers = [llama_base_tokenizer, llama_chat_tokenizer,
                  mistral_tokenizer, zephyr_tokenizer]
    print("Getting proper data format..")
    filter_hate_and_length(
        data, tokenizers, max_length=MAX_TOKENIZED_LENGTH, test_size=TEST_SIZE)


if __name__ == "__main__":
    hate_and_length_filtering()
