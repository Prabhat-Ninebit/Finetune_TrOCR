# =========================================================
# Hindi Handwritten OCR Fine-tuning (Azure + Blob Ready)
# =========================================================

import os
import torch
import pandas as pd
from PIL import Image
from datasets import Dataset
import evaluate

from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)

# =========================================================
# 1. PATHS / SETTINGS (MATCHES YOUR VM)
# =========================================================

MODEL_ID = "sabaridsnfuji/Hindi_Offline_Handwritten_OCR"

DATASET_DIR = "/home/azureuser/hindi_ocr/dataset"   # where csv files are
IMAGE_ROOT  = os.path.join(DATASET_DIR, "HindiSeg") # contains second HindiSeg/
OUTPUT_DIR  = "/mnt/blob/hindicheckpoint"           # blob mount

MAX_LENGTH = 64
BATCH_SIZE = 6          # safe for T4 16GB
EPOCHS = 10
LR = 2e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 2. LOAD CSV → HF DATASET
# =========================================================

def load_csv(name):
    df = pd.read_csv(os.path.join(DATASET_DIR, name))
    return Dataset.from_pandas(df)

train_ds = load_csv("train.csv")
val_ds   = load_csv("val.csv")


# =========================================================
# 3. LOAD PROCESSOR + MODEL (IMPORTANT)
# =========================================================

processor = TrOCRProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

model.to(DEVICE)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id


# =========================================================
# 4. PREPROCESS FUNCTION
#    CSV column names:
#       file_name -> image path
#       text      -> label
# =========================================================

PAD_TOKEN = tokenizer.pad_token_id

def preprocess(batch):

    images = [
        Image.open(os.path.join(IMAGE_ROOT, fname)).convert("RGB")
        for fname in batch["file_name"]
    ]

    inputs = processor(images=images, return_tensors="pt")

    labels = tokenizer(
        batch["text"],
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True
    ).input_ids

    labels = [
        [(l if l != PAD_TOKEN else -100) for l in label]
        for label in labels
    ]

    return {
        "pixel_values": inputs.pixel_values,
        "labels": torch.tensor(labels)
    }


train_ds.set_transform(preprocess)
val_ds.set_transform(preprocess)


# =========================================================
# 5. METRIC (CER)
# =========================================================

cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    label_ids[label_ids == -100] = PAD_TOKEN
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


# =========================================================
# 6. TRAINING ARGUMENTS (Blob + Spot safe)
# =========================================================

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    predict_with_generate=True,

    evaluation_strategy="steps",
    save_strategy="steps",

    save_steps=1000,
    eval_steps=100,
    logging_steps=100,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=4,

    learning_rate=LR,
    num_train_epochs=EPOCHS,

    fp16=True,

    save_safetensors=True,      # safer for blob
    save_total_limit=2,

    remove_unused_columns=False,
    load_best_model_at_end=True,

    metric_for_best_model="cer",
    greater_is_better=False,

    dataloader_num_workers=2,   # good for 4 vCPU
    report_to="none"
)


# =========================================================
# 7. TRAINER
# =========================================================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor
)


# =========================================================
# 8. RESUME IF CHECKPOINT EXISTS (spot safe)
# =========================================================

from transformers.trainer_utils import get_last_checkpoint

last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
print(f"🚀 Starting training. Resume from: {last_ckpt}")

trainer.train(resume_from_checkpoint=last_ckpt)


# =========================================================
# 9. FINAL SAVE
# =========================================================

final_path = os.path.join(OUTPUT_DIR, "final_model")

trainer.save_model(final_path)
processor.save_pretrained(final_path)

print(f"✅ Training complete. Model saved to: {final_path}")
