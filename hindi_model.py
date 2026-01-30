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
    default_data_collator,
    TrainerCallback
)

# -----------------------------
# 1. SETTINGS & PATHS
# -----------------------------
MODEL_ID = "sabaridsnfuji/Hindi_Offline_Handwritten_OCR"
DATASET_DIR = "/home/azureuser/hindi_ocr/dataset"
IMAGE_DIR   = "/home/azureuser/hindi_ocr/dataset/HindiSeg"
OUTPUT_DIR  = "/mnt/blob/hindicheckpoint"

MAX_LENGTH = 64
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5 
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
img_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
processor = TrOCRProcessor(image_processor=img_processor, tokenizer=tokenizer)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

# FORCE THE INTEGER ID
# For RoBERTa-based Hindi decoders, 0 is the <s> token
START_TOKEN_ID = 0 
PAD_TOKEN_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

# Apply to model config
model.config.decoder_start_token_id = START_TOKEN_ID
model.config.pad_token_id = PAD_TOKEN_ID
model.config.eos_token_id = tokenizer.sep_token_id or 2

# Apply to generation config
model.generation_config.decoder_start_token_id = START_TOKEN_ID
model.generation_config.pad_token_id = PAD_TOKEN_ID

# -----------------------------
# 4. UPDATED CALLBACK (FORCED INTEGER)
# -----------------------------
class FixConfigCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            # We hardcode 0 here to ensure it's NEVER None
            model.config.decoder_start_token_id = 0 
            model.config.pad_token_id = PAD_TOKEN_ID
            model.generation_config.decoder_start_token_id = 0
            print(f"✅ FINAL FORCE: decoder_start_token_id set to 0")
            
def preprocess(batch):
    # Fix: Load images properly for set_transform
    images = [Image.open(os.path.join(IMAGE_DIR, name)).convert("RGB") for name in batch["file_name"]]
    
    # Fix: Ensure tensors are created without Numpy 2.0 copy warnings
    inputs = processor(images, return_tensors="pt") 
    
    labels = tokenizer(
        batch["text"], 
        padding="max_length", 
        max_length=MAX_LENGTH, 
        truncation=True
    ).input_ids
    
    # Replace padding with -100
    labels = [[(l if l != PAD_TOKEN else -100) for l in label] for label in labels]
    
    return {
        "pixel_values": inputs.pixel_values, 
        "labels": torch.as_tensor(labels)
    }

train_ds.set_transform(preprocess)
val_ds.set_transform(preprocess)

# -----------------------------
# 5. METRICS & TRAINING ARGS
# -----------------------------
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = PAD_TOKEN
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    return {"cer": cer_metric.compute(predictions=pred_str, references=label_str)}

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    predict_with_generate=True,
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
    fp16=True,
    warmup_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    eval_accumulation_steps=1,
    dataloader_num_workers=0, # Changed to 0 to avoid multiprocessing issues with set_transform
    report_to="none"
)

# -----------------------------
# 6. EXECUTION
# -----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[FixConfigCallback()] # <--- Added the Fix
)

# Check for existing checkpoint
from transformers.trainer_utils import get_last_checkpoint
last_checkpoint = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None

print(f"🚀 Starting training. Resume from: {last_checkpoint}")
trainer.train(resume_from_checkpoint=last_checkpoint)

# Final Save
final_path = os.path.join(OUTPUT_DIR, "final_hindi_model")
trainer.save_model(final_path)
processor.save_pretrained(final_path)
print(f"✅ Saved to {final_path}")
