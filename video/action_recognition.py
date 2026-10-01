import os
os.environ["TFHUB_CACHE_DIR"] = "D:\\tfhub-cache"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

model_url = "https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3"
model = hub.load(model_url)

labels_url = "https://raw.githubusercontent.com/tensorflow/models/master/official/projects/movinet/files/kinetics_600_labels.txt"
labels_path = tf.keras.utils.get_file("kinetics_600_labels.txt", labels_url)
with open(labels_path) as f:
    labels = [line.strip() for line in f.readlines()]


def predict_video(video_path, num_frames=64):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (172, 172))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) < num_frames:
        print(f"В видео только {len(frames)} кадров, нужно {num_frames}")
        return

    video_tensor = tf.convert_to_tensor(np.array(frames), dtype=tf.float32) / 255.0
    video_tensor = tf.expand_dims(video_tensor, axis=0)

    outputs = model({"image": video_tensor})
    logits = outputs
    probabilities = tf.nn.softmax(logits[0])

    top5 = tf.argsort(probabilities, direction="DESCENDING")[:5]

    print(f"\nВидео: {os.path.basename(video_path)}")
    print("Действия:")
    for i in top5:
        idx = int(i)
        prob = float(probabilities[idx]) * 100
        print(f"{labels[idx]}: {prob:.2f}%")

predict_video(r"D:\ml-applications\video\test1.mp4")
predict_video(r"D:\ml-applications\video\test2.mp4")