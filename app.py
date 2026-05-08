import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import time
from collections import Counter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Live Detection", layout="wide")

# ---------------- CSS DESIGN ----------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Background */
.stApp {
    background-color: #edf2fa;
}

/* TITLE ONLY (BLACK) */
h1 {
    text-align: center;
    color: #000000;
    font-weight: 700;
}

/* PREFACE / INTRO TEXT ONLY (BLACK) */
.stMarkdown p {
    color: #000000 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #d7e3fc;
}

/* Slider label */
.stSlider label {
    color: #1d3557 !important;
}

/* Card */
.custom-box {
    background: #ccdbfd;
    padding: 15px;
    border-radius: 14px;
    border: 1px solid #c1d3fe;
    margin-bottom: 10px;
}

/* Status text */
.status {
    color: #000000;
    font-size: 18px;
    font-weight: 600;
    text-align: center;
}

/* Buttons */
.stButton>button {
    background-color: #abc4ff;
    color: #1d3557;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

/* Hide footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE + PREFACE ----------------

st.markdown("""
<style>
h1 {
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🎥 Live Object Detection & Tracking")
st.markdown("""
<div class="custom-box">
    <p class="status"> AI Detection Running in Real-Time</p>
</div>
""", unsafe_allow_html=True)

st.write("Allow camera access to detect, track, and analyze live objects.")

# ---------------- SIDEBAR ----------------
confidence = st.sidebar.slider("Confidence Threshold", 0.25, 1.0, 0.5)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# ---------------- FPS STATE ----------------
state = {"prev_time": time.time()}

# ---------------- VIDEO CALLBACK ----------------
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model.predict(img, conf=confidence, verbose=False)
    result = results[0]

    annotated = img.copy()
    counts = Counter()

    if result.boxes is not None:
        names = result.names

        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < confidence:
                continue

            cls = int(box.cls[0])
            label = names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            counts[label] += 1

            # Bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (171, 196, 255),
                2
            )

            # Label text
            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (29, 53, 87),
                1,
                cv2.LINE_AA
            )

    # ---------------- FPS ----------------
    curr = time.time()
    fps = 1 / max(curr - state["prev_time"], 1e-6)
    state["prev_time"] = curr

    cv2.putText(
        annotated,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (29, 53, 87),
        2
    )

    # ---------------- ALERTS ----------------
    for obj in ["person", "cell phone", "bottle"]:
        if obj in counts:
            cv2.putText(
                annotated,
                f"ALERT: {obj.upper()}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

    # ---------------- COUNTS ----------------
    y = 110
    for obj, num in counts.items():
        cv2.putText(
            annotated,
            f"{obj}: {num}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (29, 53, 87),
            1
        )
        y += 20

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ---------------- STREAM ----------------
webrtc_streamer(
    key="fixed-app",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)