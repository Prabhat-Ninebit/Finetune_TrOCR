from transformers import AutoTokenizer, AutoModelForVision2Seq
import torch

model_id = "Lamapi/next-ocr"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16)

img = Image.open("image.jpg")

# ATTENTION: The content list must include both an image and text.
messages = [
    {"role": "system", "content": "You are Next-OCR, an helpful AI assistant trained by Lamapi."},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Read the text in this image and summarize it."}
        ]
    }
]

# Apply the chat template correctly
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=prompt, images=[img], return_tensors="pt").to(model.device)

with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=256)

print(processor.decode(generated[0], skip_special_tokens=True))
