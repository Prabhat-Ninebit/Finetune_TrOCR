import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from datasets import Dataset
import evaluate
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    AutoImageProcessor,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)

# -----------------------------
# 1. SETTINGS & PATHS
# -----------------------------
# Use the Hindi-specialized model as our foundation
MODEL_ID = "sabaridsnfuji/Hindi_Offline_Handwritten_OCR"
DATASET_DIR = "/home/azureuser/hindi_ocr/dataset"
IMAGE_DIR   = "/home/azureuser/hindi_ocr/dataset/HindiSeg"
OUTPUT_DIR  = "/mnt/blob/hindicheckpoint"

MAX_LENGTH = 64
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5  # Low learning rate for fine-tuning
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 2. DATA LOADING
# -----------------------------
def prepare_dataset(csv_file):
    df = pd.read_csv(os.path.join(DATASET_DIR, csv_file))
    return Dataset.from_pandas(df)

train_ds = prepare_dataset("train.csv")
val_ds = prepare_dataset("val.csv")

# -----------------------------
# 3. COMPONENT INITIALIZATION
# -----------------------------
# FIX for OSError: Manual assembly because 'sabaridsnfuji' repo lacks image config
img_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
processor = TrOCRProcessor(image_processor=img_processor, tokenizer=tokenizer)

# Load the model weights
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

# FIX for ValueError: Move generation params to generation_config
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id or processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id

model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
model.generation_config.pad_token_id = model.config.pad_token_id
model.generation_config.eos_token_id = model.config.eos_token_id
model.generation_config.max_length = MAX_LENGTH
model.generation_config.num_beams = 4  # Better for Hindi characters

model.to(DEVICE)

# -----------------------------
# 4. PREPROCESSING PIPELINE
# -----------------------------
# -----------------------------
# 4. PREPROCESSING PIPELINE (INSTANT START)
# -----------------------------
def preprocess(batch):
    # Image processing
    images = [Image.open(os.path.join(IMAGE_DIR, name)).convert("RGB") for name in batch["file_name"]]
    pixel_values = processor(images, return_tensors="pt").pixel_values
    
    # Text processing
    labels = processor.tokenizer(
        batch["text"], 
        padding="max_length", 
        max_length=MAX_LENGTH, 
        truncation=True
    ).input_ids
    
    labels = [[(l if l != processor.tokenizer.pad_token_id else -100) for l in label] for label in labels]
    
    return {
        "pixel_values": torch.tensor(np.array(pixel_values)), 
        "labels": torch.tensor(labels)
    }

# Use set_transform instead of map. This takes 0 seconds.
train_ds.set_transform(preprocess)
val_ds.set_transform(preprocess)

# -----------------------------
# 5. EVALUATION METRIC (CER)
# -----------------------------
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

# -----------------------------
# 6. TRAINING CONFIGURATION
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    predict_with_generate=True,       # Essential to get actual text for CER
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    logging_steps=100,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    remove_unused_columns=False,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    fp16=True,                        # Use mixed precision for faster training
    warmup_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    eval_accumulation_steps=1,        # Keeps VRAM low during evaluation
    dataloader_num_workers=2,
    report_to="none"
)

# -----------------------------
# 7. EXECUTION
# -----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

print("🚀 Pipeline ready. Starting training...")
trainer.train()

# -----------------------------
# 8. FINAL SAVE
# -----------------------------
final_path = os.path.join(OUTPUT_DIR, "final_hindi_model")
trainer.save_model(final_path)
processor.save_pretrained(final_path)
print(f"✅ Training finished. Model saved at {final_path}")
