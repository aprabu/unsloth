# trainer.py

# Import necessary libraries and modules
from unsloth.FastLanguageModel import FastLanguageModel
from unsloth.is_bfloat16_supported import is_bfloat16_supported
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers.integrations import WandbCallback  # Correct import statement
from datasets import load_dataset
import wandb

# Initialize wandb
wandb.login()
wandb.init(project="unsloth")

# Configuration
max_seq_length = 2048  # Supports RoPE Scaling internally, so choose any!

# Load Alpaca-cleaned dataset
dataset = load_dataset("yahma/alpaca-cleaned")

# 4bit pre-quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
]  # More models at https://huggingface.co/unsloth

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=False,
)

# Model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    max_seq_length=max_seq_length,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

# Define training arguments with wandb logging enabled
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=60,
    fp16=False,
    bf16=True,
    logging_steps=1,
    output_dir="outputs",
    optim="adamw_8bit",
    seed=3407,
    report_to="wandb"  # Enable wandb reporting
)

# Initialize trainer with WandbCallback
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    dataset_text_field="instruction",  # Adjust based on actual dataset fields
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
    callbacks=[WandbCallback()]  # Add WandbCallback to the trainer
)

# Start training
trainer.train()
