from ultralytics import YOLO
import os

# use pretrained model first
model = YOLO("yolov8n.pt")

image_folder = "dataset/train/images"
label_folder = "dataset/train/labels"

os.makedirs(label_folder, exist_ok=True)

for img in os.listdir(image_folder):

    path = os.path.join(image_folder, img)

    results = model(path)

    for r in results:
        r.save_txt(os.path.join(label_folder, img.replace(".jpg",".txt")))

print("Auto labeling done")