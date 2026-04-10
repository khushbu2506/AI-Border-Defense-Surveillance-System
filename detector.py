from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO("yolov8n.pt")

def process_video(source):

    cap = cv2.VideoCapture(source)

    os.makedirs("outputs/heatmaps", exist_ok=True)
    os.makedirs("outputs/videos", exist_ok=True)

    width = int(cap.get(3))
    height = int(cap.get(4))

    out = cv2.VideoWriter(
        "outputs/videos/detected_output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (width, height),
    )

    heatmap = None
    prev_frame = None

    person_count = 0
    vehicle_count = 0
    weapon_count = 0
    motion_alert = False

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        if heatmap is None:
            heatmap = np.zeros((h, w), dtype=np.float32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            motion = np.sum(diff)

            if motion > 500000:
                motion_alert = True

        prev_frame = gray

        results = model(frame, imgsz=320, conf=0.25)

        for r in results:

            if r.boxes is None:
                continue

            for box in r.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                label = model.names[cls]

                color = (0, 255, 0)

                if label == "person":
                    person_count += 1

                elif label in ["car", "truck", "bus"]:
                    vehicle_count += 1

                elif label in ["knife", "gun"]:
                    weapon_count += 1
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                text = f"{label} {conf:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                heatmap[y1:y2, x1:x2] += 1

        # save processed video
        out.write(frame)

        # SAVE HEATMAP EVERY FRAME (FIX)
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = heatmap_norm.astype(np.uint8)

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        cv2.imwrite(
            "outputs/heatmaps/final_heatmap.png",
            heatmap_color
        )

        yield frame, heatmap, person_count, vehicle_count, weapon_count, motion_alert

    cap.release()
    out.release()
