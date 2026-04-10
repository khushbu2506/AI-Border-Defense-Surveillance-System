import cv2
import os

video_folder = "videos_for_training"
output_folder = "dataset/train/images"

os.makedirs(output_folder, exist_ok=True)

for video in os.listdir(video_folder):

    path = os.path.join(video_folder, video)
    cap = cv2.VideoCapture(path)

    count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if count % 10 == 0:
            name = f"{video}_{count}.jpg"
            cv2.imwrite(os.path.join(output_folder, name), frame)

        count += 1

    cap.release()

print("Frames added to dataset")