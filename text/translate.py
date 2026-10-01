import os
os.environ["HF_HOME"] = "D:\\hf-cache"

from transformers import MarianTokenizer, MarianMTModel

model_name = "Helsinki-NLP/opus-mt-ru-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

texts = [
    "скучный фильм",
    "смотрела похожий фильм",
    "погода дождливая",
    "ресторан многие советуют",
    "my english is very good"
]

for text in texts:
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"RU: {text}")
    print(f"EN: {translation}")
    print()