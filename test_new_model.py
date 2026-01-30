import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
encoder_model = 'google/vit-base-patch16-224-in21k'
decoder_model = 'surajp/RoBERTa-hindi-guj-san'
trained_model_path = 'sabaridsnfuji/Hindi_Offline_Handwritten_OCR'

# 1. Initialize Components correctly
# Use AutoImageProcessor and name the argument 'image_processor'
image_processor = AutoImageProcessor.from_pretrained(encoder_model)
tokenizer = AutoTokenizer.from_pretrained(decoder_model)
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

# 2. Load the Fine-tuned Model
model = VisionEncoderDecoderModel.from_pretrained(trained_model_path).to(DEVICE)

# 3. Load and Preprocess Image
image_path = '/home/azureuser/hindi_ocr/custom_dataset/sundar.jpg'
image = Image.open(image_path).convert('RGB')

# 4. Generate Text
# We explicitly set decoder_start_token_id to ensure the model triggers Hindi generation
pixel_values = processor(image, return_tensors='pt').pixel_values.to(DEVICE)

generated_ids = model.generate(
    pixel_values,
    decoder_start_token_id=model.config.decoder.bos_token_id, 
    max_length=64,
    num_beams=4
)

# 5. Decode Result
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("--- OCR Result ---")
print("Generated Text:", generated_text)
