LORA_DPO = {
    "r":  128,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

LORA_SFT = {
    "r":  16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "inference_mode": False,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

TRAINING_ARGUMENTS_SFT = {
    "dataloader_drop_last": True,
    "evaluation_strategy": "steps",
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 16,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_8bit",
    "save_steps": 100,
    "logging_steps": 10,
    "eval_steps": 100,
    "learning_rate": 1e-4,
    "num_train_epochs": 1,
    "weight_decay": 0.05,
    "fp16": False,
    "bf16": True,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "report_to": "wandb",
    "ddp_find_unused_parameters": False,
    "save_total_limit": 5,
    "load_best_model_at_end": True,
    "gradient_checkpointing_kwargs": {
        "use_reentrant": False
    }
}

TRAINING_ARGUMENTS_DPO = {
    "dataloader_drop_last": True,
    "evaluation_strategy": "steps",
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 8,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_32bit",
    "save_steps": 100,
    "logging_steps": 10,
    "eval_steps": 100,
    "learning_rate": 5e-6,
    "num_train_epochs": 1,
    "fp16": False,
    "bf16": True,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "report_to": "wandb",
    "ddp_find_unused_parameters": False,
    "save_total_limit": 5,
    "load_best_model_at_end": True,
    "gradient_checkpointing_kwargs": {
        "use_reentrant": False
    },
    "remove_unused_columns": False
}