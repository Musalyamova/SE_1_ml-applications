import os
os.environ["HF_HOME"] = "D:\\hf-cache"

from transformers import pipeline
classifier = pipeline(
    "sentiment-analysis",
    model="blanchefort/rubert-base-cased-sentiment"
)

texts = [
    "скучный фильм",
    "смотрела похожий фильм",
    "погода дождливая",
    "ресторан многие советуют",
    "i like a banana"
]

for text in texts:
    res = classifier(text)[0]
    print(f"{text} -> {res['label']} ({res['score']:.2%})")
