import os
import torch
import pandas as pd
from PIL import Image
from datasets import Dataset
import evaluate

from transformers import (
    VisionEncoderDecoderModel,
    AutoImageProcessor,
    AutoTokenizer,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)

# ======================
# PATHS
# ======================
MODEL_ID = "sabaridsnfuji/Hindi_Offline_Handwritten_OCR"

DATASET_DIR = "/home/azureuser/hindi_ocr/dataset"
IMAGE_ROOT  = os.path.join(DATASET_DIR, "HindiSeg")
OUTPUT_DIR  = "/mnt/blob/hindicheckpoint"

MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================
# DATASET
# ======================
def load_csv(name):
    df = pd.read_csv(os.path.join(DATASET_DIR, name))
    return Dataset.from_pandas(df)

train_ds = load_csv("train.csv")
val_ds   = load_csv("val.csv")


# ======================
# PROCESSOR (MANUAL FIX)
# ======================
image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

processor = TrOCRProcessor(
    image_processor=image_processor,
    tokenizer=tokenizer
)


# ======================
# MODEL
# ======================
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

model.config.decoder_start_token_id = tokenizer.cls_token_id or 0
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id or 2


# ======================
# PREPROCESS
# ======================
PAD_TOKEN = tokenizer.pad_token_id

def preprocess(batch):

    images = [
        Image.open(os.path.join(IMAGE_ROOT, f)).convert("RGB")
        for f in batch["file_name"]
    ]

    inputs = processor(images=images, return_tensors="pt")

    labels = tokenizer(
        batch["text"],
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True
    ).input_ids

    labels = [[l if l != PAD_TOKEN else -100 for l in lab] for lab in labels]

    return {
        "pixel_values": inputs.pixel_values,
        "labels": torch.tensor(labels)
    }


train_ds.set_transform(preprocess)
val_ds.set_transform(preprocess)


# ======================
# METRIC
# ======================
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    label_ids[label_ids == -100] = PAD_TOKEN
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    return {"cer": cer_metric.compute(predictions=pred_str, references=label_str)}


# ======================
# TRAINING ARGS
# ======================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    predict_with_generate=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=1000,
    eval_steps=100,
    logging_steps=100,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=6,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    fp16=True,
    save_safetensors=True,
    save_total_limit=2,
    remove_unused_columns=False,
    metric_for_best_model="cer",
    greater_is_better=False,
    dataloader_num_workers=2,
    report_to="none"
)


# ======================
# TRAINER
# ======================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor
)


# ======================
# RESUME + TRAIN
# ======================
from transformers.trainer_utils import get_last_checkpoint

ckpt = get_last_checkpoint(OUTPUT_DIR)
print("Resume:", ckpt)

trainer.train(resume_from_checkpoint=ckpt)


# ======================
# SAVE
# ======================
final_path = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(final_path)
processor.save_pretrained(final_path)

print("✅ Done")
