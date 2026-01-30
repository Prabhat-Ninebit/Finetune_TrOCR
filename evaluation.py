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
from transformers.trainer_utils import get_last_checkpoint

# -----------------------------
# GPU CHECK
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "microsoft/trocr-base-handwritten"

DATASET_DIR = "/home/azureuser/hindi_ocr/dataset"
IMAGE_DIR   = "/home/azureuser/hindi_ocr/dataset/HindiSeg"   # IMPORTANT (see CSV paths)
OUTPUT_DIR  = "/mnt/blob/checkpoints"

MAX_LABEL_LENGTH = 32
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 5e-5

def load_csv(csv_path):
    df = pd.read_csv(
        csv_path,
        header=0,                       # EXPLICIT: first row is header
        usecols=["file_name", "text"],  # enforce schema
    )
    return Dataset.from_pandas(df, preserve_index=False)


val_ds   = load_csv(os.path.join(DATASET_DIR, "val.csv"))

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)

# 2. Determine where to load the model weights from
if os.path.isdir(OUTPUT_DIR) and get_last_checkpoint(OUTPUT_DIR):
    checkpoint_path = get_last_checkpoint(OUTPUT_DIR)
    print(f"✅ Loading fine-tuned weights from: {checkpoint_path}")
    # Load weights from the checkpoint
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)
else:
    print("⚠️ No checkpoint found! Loading the BASE model.")
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# 3. Standard model config setup
# 1. Force the start token (TrOCR usually uses the 'cls_token')
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id

# 2. Ensure vocab sizes match (crucial for some TrOCR versions)
model.config.vocab_size = model.config.decoder.vocab_size

# 3. Update the Generation Config (This is often what's missing)
model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
model.generation_config.eos_token_id = processor.tokenizer.sep_token_id

model.to(DEVICE)

# -----------------------------
# WEIGHT INTEGRITY CHECK
# -----------------------------
# print("🔍 Checking model weights for collapse...")
# has_nan = False
# for name, param in model.named_parameters():
#     if torch.isnan(param).any():
#         print(f"🚨 WEIGHT COLLAPSE DETECTED: {name} contains NaNs!")
#         has_nan = True
#     if torch.isinf(param).any():
#         print(f"🚨 WEIGHT EXPLOSION DETECTED: {name} contains Infs!")
#         has_nan = True

# if not has_nan:
#     print("✅ Weights appear numerically stable (no NaNs or Infs).")
# else:
#     print("❌ Model is corrupted. You must restart training with a lower learning rate.")
# # -----------------------------
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
    
val_ds   = val_ds.map(preprocess, remove_columns=val_ds.column_names)

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

        if len(images) == 0:
            raise ValueError("Empty batch encountered")

        pixel_values = self.processor(
            images=images,
            return_tensors="pt"
        ).pixel_values

        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


data_collator = TrOCRDataCollator(processor, IMAGE_DIR)


# -----------------------------
# METRIC (CER)
# -----------------------------
cer_metric = evaluate.load("cer")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Seq2SeqTrainer may return tuples
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # If predictions are logits, convert to token IDs
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    # Replace -100 in labels so tokenizer can decode
    labels = np.where(
        labels != -100,
        labels,
        processor.tokenizer.pad_token_id
    )

    # VERY IMPORTANT: ensure correct dtype
    predictions = predictions.astype(np.int64)
    labels = labels.astype(np.int64)

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

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    # --- Evaluation Activation ---
    do_train=False,                   # Disable the training loop
    do_eval=True,                    # Enable evaluation mode
    eval_strategy="no",              # Leave as "no" since you'll call .evaluate() manually
    
    # --- Batch & Performance ---
    per_device_eval_batch_size=8,    # Increase from 1 for faster eval if memory allows
    eval_accumulation_steps=10,      # Increase to reduce GPU-to-CPU overhead
    fp16=True,                       # Keep for faster inference on compatible GPUs

    # --- Sequence-to-Sequence Settings ---
    predict_with_generate=True,      # REQUIRED to calculate CER; generates actual text instead of raw loss
    
    # --- Cleanup ---
    report_to="none",
    seed=42,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,                       # Your fine-tuned model
    args=training_args,
    eval_dataset=val_ds,    # Validation data is required for evaluation
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)
print("🔍 Running forced manual generation test...")
sample_batch = [val_ds[i] for i in range(3)]
inputs = data_collator(sample_batch)
pixel_values = inputs["pixel_values"].to(DEVICE)

generated_ids = model.generate(
    pixel_values, 
    max_length=64,
    min_length=5,              # FORCE it to generate at least 5 tokens
    num_beams=4,
    decoder_start_token_id=model.config.decoder_start_token_id,
    use_cache=True
)

manual_preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(f"DEBUG PREDICTIONS: {manual_preds}")
test_text = "नमस्ते" # 'Namaste' in Hindi
tokens = processor.tokenizer.tokenize(test_text)
print(f"Tokenization Test: {tokens}")
print(f"Vocabulary Size: {len(processor.tokenizer)}")


print("📊 Evaluation started...")

# Force the model to predict on just 3 samples to see the text
test_output = trainer.predict(val_ds.select(range(3)))
print("PREDICTIONS:", processor.batch_decode(test_output.predictions, skip_special_tokens=True))

results = trainer.evaluate()

print("Evaluation Results:", results)