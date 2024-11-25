import bitsandbytes as bnb
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def print_trainable_parameters(model: AutoModelForCausalLM | AutoPeftModelForCausalLM) -> None:
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


def find_all_linear_names(model: AutoModelForCausalLM | AutoPeftModelForCausalLM) -> list[str]:
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    print(list(lora_module_names))
    return list(lora_module_names)


def chars_token_ratio(dataset: list[dict], tokenizer: AutoTokenizer, nb_examples: int = 400) -> float:
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

def return_prompt_and_responses(samples: dict) -> dict:
    return {
        "prompt": samples["prompt"],
        "chosen": samples["chosen"],
        "rejected": samples["rejected"],
    }