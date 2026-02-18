# finetune-pk-v1

# PK-Genesis-v1 MedGemma 4B Finetune Source

This repo contains source files for finetuning MedGemma 4B with PK-Genesis-v1 identity behavior while preserving medical capability.

## Files
- `train_pk_genesis_medgemma4b_lora.py` - LoRA training script
- `pk_genesis_v1_identity_dataset.jsonl` - training dataset (identity + company + medical + safety)

## Intended behavior
- Identity only when asked
- Company attribution: `PharmKulen Technology`
- Preserve healthcare response quality and safety rules

## Quick start
```bash
python train_pk_genesis_medgemma4b_lora.py
```

## Note
Current script may need minor adjustments for exact MedGemma PEFT target modules depending on environment package versions.
