# ML-applications
4 решения задач с применением ML-моделей (текст, аудио, изображения и видео)

## Стек
- Python 3.13
- Hugging Face
- PyTorch Hub
- TensorFlow Hub
- OpenCV
- Pillow
- matplotlib

## Структура
```
SE_1_ml-applications/
├── text/           # Анализ тональности
├── audio/          # Распознавание речи
├── image/          # Сегментация изображений
└── video/          # Распознавание действий 
```

## Задачи

| # | Папка | Модель | Что делает                             |
|---|------|--------|----------------------------------------|
| 1 | text | `blanchefort/rubert-base-cased-sentiment` | Определяет тональность текста |
| 2 | audio | `openai/whisper-small` | Превращает аудио в текст               |
| 3 | image | `fcn_resnet50` | Выделяет объекты на изображении        |
| 4 | video | `movinet/a0` | Распознаёт действия на видео           |

## Установка
```bash
pip install transformers torch torchvision tensorflow tensorflow-hub opencv-python pillow matplotlib librosa soundfile pandas numpy
```
Также для работы с аудио нужен ffmpeg

## Запуск
```bash
python text/sentiment.py
python audio/speech_to_text.py
python image/segmentation.py
python video/action_recognition.py
```

## Автор
Мусалямова Алина, 11-411