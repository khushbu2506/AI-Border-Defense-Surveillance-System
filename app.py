import streamlit as st
import tempfile
import cv2
import pandas as pd
import os
import numpy as np
from backend.detector import process_video

st.set_page_config(layout="wide")

st.title("AI Border Defense Surveillance Dashboard")

# Sidebar
st.sidebar.header("Controls")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Upload Video", "Live Webcam", "Drone Mode"]
)

start = st.sidebar.button("Start Detection")

video_source = None

# Upload mode
if mode == "Upload Video":
    uploaded = st.file_uploader("Upload Surveillance Video")

    if uploaded is not None:
        st.video(uploaded)

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())

        video_source = tfile.name

# Webcam
elif mode == "Live Webcam":
    video_source = 0

# Drone
elif mode == "Drone Mode":
    uploaded = st.file_uploader("Upload Drone Video")

    if uploaded is not None:
        st.video(uploaded)

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())

        video_source = tfile.name


if start and video_source is not None:

    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()
    alert_placeholder = st.empty()
    heatmap_placeholder = st.empty()

    detections = []

    for frame, heatmap, person, vehicle, weapon, motion in process_video(video_source):

        # draw intrusion zone
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 255), 2)
        cv2.putText(
            frame,
            "Restricted Border Zone",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        frame_placeholder.image(frame, channels="BGR")

        risk = weapon * 5 + person * 2 + vehicle

        detections.append([person, vehicle, weapon, risk])

        # analytics
        metrics_placeholder.markdown(
            f"""
### Live Analytics
Persons: **{person}**  
Vehicles: **{vehicle}**  
Weapons: **{weapon}**  
Risk Score: **{risk}**
"""
        )

        # alerts
        if weapon > 0:
            alert_placeholder.error("🚨 Weapon Detected")

        elif motion:
            alert_placeholder.warning("⚠ Suspicious Movement")

        elif risk > 40:
            alert_placeholder.error("HIGH RISK ZONE")

        elif risk > 20:
            alert_placeholder.warning("MEDIUM RISK")

        else:
            alert_placeholder.success("SAFE ZONE")

        # SAFE heatmap conversion
        if heatmap is not None:

            heatmap_np = np.array(heatmap)

            heatmap_norm = cv2.normalize(
                heatmap_np,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

            heatmap_uint8 = heatmap_norm.astype(np.uint8)

            heatmap_color = cv2.applyColorMap(
                heatmap_uint8,
                cv2.COLORMAP_JET
            )

            heatmap_placeholder.image(
                heatmap_color,
                channels="BGR"
            )

    # save backup
    os.makedirs("outputs/logs", exist_ok=True)

    df = pd.DataFrame(
        detections,
        columns=["persons", "vehicles", "weapons", "risk"]
    )

    df.to_csv("outputs/logs/detection_log.csv", index=False)

    st.success("Backup saved in outputs folder")

st.sidebar.success("AI Model Active")