import os
import torch
import evaluate
import pandas as pd
import numpy as np

from PIL import Image
from datasets import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# -----------------------------
# GPU CHECK
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# -----------------------------
# CONFIG
# -----------------------------
MODEL_DIR = "/mnt/blob/checkpoints"   # trained model dir
DATASET_DIR = "/home/azureuser/hindi_ocr/dataset"
IMAGE_DIR   = "/home/azureuser/hindi_ocr/dataset/HindiSeg"

MAX_LABEL_LENGTH = 32
EVAL_BATCH_SIZE = 1   # keep small for GPU safety

# -----------------------------
# LOAD CSV
# -----------------------------
def load_csv(csv_path):
    df = pd.read_csv(
        csv_path,
        header=0,
        usecols=["file_name", "text"],
    )
    return Dataset.from_pandas(df, preserve_index=False)

val_ds = load_csv(os.path.join(DATASET_DIR, "val.csv"))

# -----------------------------
# LOAD MODEL & PROCESSOR
# -----------------------------
processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id

model.to(DEVICE)
model.eval()

# -----------------------------
# PREPROCESS (TEXT → LABELS)
# -----------------------------
def preprocess(batch):
    encoding = processor.tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LABEL_LENGTH,
        add_special_tokens=True,
    )

    labels = [
        token if token != processor.tokenizer.pad_token_id else -100
        for token in encoding.input_ids
    ]

    return {
        "file_name": batch["file_name"],
        "labels": labels,
    }

val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names)

# -----------------------------
# DATA COLLATOR
# -----------------------------
class TrOCRDataCollator:
    def __init__(self, processor, image_dir):
        self.processor = processor
        self.image_dir = image_dir

    def __call__(self, features):
        images = []
        labels = []

        for item in features:
            image_path = os.path.join(self.image_dir, item["file_name"])
            image = Image.open(image_path).convert("RGB")
            images.append(image)
            labels.append(item["labels"])

        pixel_values = self.processor(
            images=images,
            return_tensors="pt"
        ).pixel_values

        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

data_collator = TrOCRDataCollator(processor, IMAGE_DIR)

# -----------------------------
# METRIC (CER)
# -----------------------------
cer_metric = evaluate.load("cer")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # predictions are token IDs (because predict_with_generate=True)
    labels = np.where(
        labels != -100,
        labels,
        processor.tokenizer.pad_token_id
    )

    pred_str = processor.batch_decode(
        predictions,
        skip_special_tokens=True
    )

    label_str = processor.batch_decode(
        labels,
        skip_special_tokens=True
    )

    cer = cer_metric.compute(
        predictions=pred_str,
        references=label_str
    )

    return {"cer": cer}

# -----------------------------
# EVAL ARGS (GPU SAFE)
# -----------------------------
eval_args = Seq2SeqTrainingArguments(
    output_dir="./eval_tmp",
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    fp16=True,
    predict_with_generate=True,   # VERY IMPORTANT
    generation_max_length=MAX_LABEL_LENGTH,
    dataloader_num_workers=0,
    report_to="none",
)

# -----------------------------
# TRAINER (EVAL ONLY)
# -----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=eval_args,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -----------------------------
# RUN EVALUATION
# -----------------------------
print("🔍 Running evaluation...")
metrics = trainer.evaluate()

print("\n📊 Evaluation Results:")
for k, v in metrics.items():
    print(f"{k}: {v}")
