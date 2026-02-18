from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

BASE_MODEL = "/home/spark/models/medgemma-4b-it"
DATA_PATH = "/home/spark/.openclaw/workspace/pk_genesis_v1_identity_dataset.jsonl"
OUT_DIR = "/home/spark/models/pk-genesis-v1-medgemma4b-lora"
MAX_LEN = 512


def build_example(tokenizer, prompt: str, response: str):
    # Identity should appear only when asked (already encoded in dataset responses).
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prefix + response + (tokenizer.eos_token or "")

    tok_full = tokenizer(full_text, truncation=True, max_length=MAX_LEN)
    tok_prefix = tokenizer(prefix, truncation=True, max_length=MAX_LEN)

    input_ids = tok_full["input_ids"]
    attention_mask = tok_full["attention_mask"]
    labels = input_ids.copy()

    p = min(len(tok_prefix["input_ids"]), len(labels))
    labels[:p] = [-100] * p

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    tokenizer = processor.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=DATA_PATH, split="train")

    ds = ds.map(
        lambda ex: build_example(tokenizer, ex["prompt"], ex["response"]),
        remove_columns=ds.column_names,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=8e-5,
        num_train_epochs=2,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=2,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"DONE: saved LoRA adapter to {OUT_DIR}")


if __name__ == "__main__":
    main()
