import os
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import requests
import streamlit as st
import tensorflow as tf

from streamlit_webrtc import webrtc_streamer


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="VisionAI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final_model.keras"
CASCADE_PATH = BASE_DIR / "assets" / "haarcascade_frontalface_default.xml"

IMG_SIZE = 128
GENDER_THRESHOLD = 0.5


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #090d12;
        color: #f5f7fa;
    }

    section[data-testid="stSidebar"] {
        background: #10161d;
        border-right: 1px solid #202933;
    }

    .brand {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #8d98a5;
        font-size: 14px;
        margin-bottom: 35px;
    }

    .hero {
        padding: 10px 0 25px 0;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .hero p {
        color: #8d98a5;
        font-size: 16px;
    }

    .card {
        background: #11171e;
        border: 1px solid #242d37;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .stat-title {
        color: #7f8b99;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stat-value {
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }

    .online {
        color: #42e878;
        font-weight: 700;
    }

    .offline {
        color: #ff5c5c;
        font-weight: 700;
    }

    .section-title {
        font-size: 25px;
        font-weight: 750;
        margin: 25px 0 15px 0;
    }

    .info {
        background: #14283c;
        border: 1px solid #1e466b;
        border-radius: 10px;
        padding: 15px;
        color: #b9d7f2;
    }

    .result {
        background: #11171e;
        border: 1px solid #242d37;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }

    .result-number {
        font-size: 35px;
        font-weight: 800;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

@st.cache_resource
def load_face_detector():

    if not CASCADE_PATH.exists():
        raise FileNotFoundError(
            f"Haar Cascade file not found: {CASCADE_PATH}"
        )

    detector = cv2.CascadeClassifier(str(CASCADE_PATH))

    if detector.empty():
        raise RuntimeError(
            "Haar Cascade could not be loaded."
        )

    return detector


# ============================================================
# CLOUDFLARE TURN
# ============================================================

@st.cache_data(ttl=3600)
def get_ice_servers():

    # --------------------------------------------------------
    # Streamlit Cloud Secrets
    #
    # [cloudflare]
    # turn_key_id = "..."
    # turn_api_token = "..."
    # --------------------------------------------------------

    try:
        turn_key_id = st.secrets["cloudflare"]["turn_key_id"]
        turn_api_token = st.secrets["cloudflare"]["turn_api_token"]

    except Exception:

        # Optional environment-variable fallback
        turn_key_id = os.getenv("CLOUDFLARE_TURN_KEY_ID")
        turn_api_token = os.getenv(
            "CLOUDFLARE_TURN_KEY_API_TOKEN"
        )

    if not turn_key_id or not turn_api_token:
        raise RuntimeError(
            "Cloudflare TURN credentials are not configured."
        )

    url = (
        "https://rtc.live.cloudflare.com/v1/turn/keys/"
        f"{turn_key_id}/credentials/generate-ice-servers"
    )

    headers = {
        "Authorization": f"Bearer {turn_api_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        url,
        headers=headers,
        json={
            "ttl": 7200
        },
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    if "iceServers" not in data:
        raise RuntimeError(
            "Cloudflare did not return ICE servers."
        )

    return data["iceServers"]


# ============================================================
# SHARED STATE
# ============================================================

state_lock = threading.Lock()

detection_state = {
    "faces": 0,
    "gender": "-",
    "age": "-",
    "confidence": 0.0,
}


# ============================================================
# PREDICTION
# ============================================================

def predict_face(model, face):

    # BGR -> RGB
    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    face = cv2.resize(
        face,
        (IMG_SIZE, IMG_SIZE)
    )

    # Normalize
    face = face.astype(
        np.float32
    ) / 255.0

    # Batch
    face = np.expand_dims(
        face,
        axis=0
    )

    # Predict
    prediction = model.predict(
        face,
        verbose=0
    )

    gender_prediction = float(
        prediction[0][0][0]
    )

    age_prediction = float(
        prediction[1][0][0]
    )

    # IMPORTANT:
    #
    # Your trained model uses:
    #
    # gender >= 0.5 -> Male
    # gender <  0.5 -> Female
    #
    gender = (
        "Male"
        if gender_prediction >= GENDER_THRESHOLD
        else "Female"
    )

    # Distance from decision boundary
    gender_confidence = (
        gender_prediction
        if gender_prediction >= 0.5
        else 1.0 - gender_prediction
    )

    age = max(
        0,
        min(
            100,
            int(round(age_prediction))
        )
    )

    return (
        gender,
        age,
        gender_confidence
    )


# ============================================================
# VIDEO CALLBACK
# ============================================================

def video_frame_callback(frame):

    img = frame.to_ndarray(
        format="bgr24"
    )

    try:

        model = load_model()
        face_detector = load_face_detector()

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        results = []

        for (x, y, w, h) in faces:

            # ------------------------------------------------
            # Add margin around face
            # ------------------------------------------------

            margin_x = int(w * 0.15)
            margin_y = int(h * 0.15)

            x1 = max(
                0,
                x - margin_x
            )

            y1 = max(
                0,
                y - margin_y
            )

            x2 = min(
                img.shape[1],
                x + w + margin_x
            )

            y2 = min(
                img.shape[0],
                y + h + margin_y
            )

            face = img[
                y1:y2,
                x1:x2
            ]

            if face.size == 0:
                continue

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            gender, age, confidence = predict_face(
                model,
                face
            )

            results.append(
                (
                    gender,
                    age,
                    confidence
                )
            )

            # ------------------------------------------------
            # Box
            # ------------------------------------------------

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 120),
                2
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            label = (
                f"{gender} | Age: {age} | "
                f"{confidence * 100:.0f}%"
            )

            label_y = max(
                30,
                y1 - 10
            )

            # Background
            (text_w, text_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            cv2.rectangle(
                img,
                (x1, label_y - text_h - 12),
                (x1 + text_w + 10, label_y),
                (10, 20, 30),
                -1
            )

            cv2.putText(
                img,
                label,
                (x1 + 5, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Update shared state
        # ----------------------------------------------------

        with state_lock:

            detection_state["faces"] = len(
                results
            )

            if results:

                # Show first detected face
                gender, age, confidence = results[0]

                detection_state["gender"] = gender
                detection_state["age"] = age
                detection_state["confidence"] = (
                    confidence
                )

            else:

                detection_state["gender"] = "-"
                detection_state["age"] = "-"
                detection_state["confidence"] = 0.0

    except Exception:

        # Don't kill WebRTC if a frame fails
        pass

    return av.VideoFrame.from_ndarray(
        img,
        format="bgr24"
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    model = load_model()
    face_detector = load_face_detector()

    system_ready = True

except Exception as e:

    system_ready = False

    st.error(
        f"VisionAI initialization failed: {e}"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">👁️ VisionAI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-powered face intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Live Analysis",
            "Image Analysis",
            "Model Information",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if system_ready:

        st.markdown(
            '<p class="online">● SYSTEM ONLINE</p>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<p class="offline">● SYSTEM ERROR</p>',
            unsafe_allow_html=True
        )

    st.caption(
        "TensorFlow • OpenCV • CNN"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>VisionAI</h1>
        <p>
            Real-time gender and age detection
            using a multi-task convolutional neural network.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LIVE ANALYSIS
# ============================================================

if page == "Live Analysis":

    st.markdown(
        '<div class="section-title">Live Analysis</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Cloudflare TURN
    # --------------------------------------------------------

    try:

        ice_servers = get_ice_servers()

        turn_ready = True

    except Exception as e:

        ice_servers = [
            {
                "urls": [
                    "stun:stun.cloudflare.com:3478"
                ]
            }
        ]

        turn_ready = False

        st.warning(
            f"TURN unavailable: {e}"
        )

    # --------------------------------------------------------
    # Status cards
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            """
            <div class="card">
                <div class="stat-title">Model</div>
                <div class="stat-value">CNN</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="card">
                <div class="stat-title">Input</div>
                <div class="stat-value">128×128</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="card">
                <div class="stat-title">Tasks</div>
                <div class="stat-value">2</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        status = (
            "READY"
            if turn_ready
            else "STUN ONLY"
        )

        st.markdown(
            f"""
            <div class="card">
                <div class="stat-title">Network</div>
                <div class="stat-value">{status}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # Camera + Results
    # --------------------------------------------------------

    camera_col, result_col = st.columns(
        [2.3, 1]
    )

    with camera_col:

        st.markdown(
            '<div class="section-title">Camera Feed</div>',
            unsafe_allow_html=True
        )

        st.info(
            "Click START and allow camera access."
        )

        webrtc_ctx = webrtc_streamer(
            key="visionai-live-analysis",

            video_frame_callback=(
                video_frame_callback
            ),

            media_stream_constraints={
                "video": True,
                "audio": False,
            },

            rtc_configuration={
                "iceServers": ice_servers
            },

            media_toggle_controls=True,
        )

    with result_col:

        st.markdown(
            '<div class="section-title">Live Detection</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="card">

            <h3>Face Detection</h3>

            <p style="color:#8d98a5;">
            Predictions are generated from
            the live camera stream.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # Read shared state
        # ----------------------------------------------------

        with state_lock:

            faces = detection_state["faces"]
            gender = detection_state["gender"]
            age = detection_state["age"]
            confidence = detection_state["confidence"]

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Faces",
                faces
            )

        with c2:

            st.metric(
                "Gender",
                gender
            )

        c3, c4 = st.columns(2)

        with c3:

            st.metric(
                "Age",
                age
            )

        with c4:

            st.metric(
                "Confidence",
                f"{confidence * 100:.1f}%"
            )

    # --------------------------------------------------------
    # Detection overview
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Detection Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="result">
                <div class="stat-title">
                    Gender Classification
                </div>
                <div class="result-number">
                    Binary
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="result">
                <div class="stat-title">
                    Age Estimation
                </div>
                <div class="result-number">
                    Regression
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="result">
                <div class="stat-title">
                    Face Detector
                </div>
                <div class="result-number">
                    Haar
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

elif page == "Image Analysis":

    st.markdown(
        '<div class="section-title">Image Analysis</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file:

        file_bytes = np.asarray(
            bytearray(
                uploaded_file.read()
            ),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error(
                "Could not read the image."
            )

        else:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            output = image.copy()

            results = []

            for (x, y, w, h) in faces:

                face = image[
                    y:y+h,
                    x:x+w
                ]

                if face.size == 0:
                    continue

                gender, age, confidence = predict_face(
                    model,
                    face
                )

                results.append(
                    (
                        gender,
                        age,
                        confidence
                    )
                )

                cv2.rectangle(
                    output,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 120),
                    2
                )

                label = (
                    f"{gender} | Age: {age}"
                )

                cv2.putText(
                    output,
                    label,
                    (x, max(30, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 120),
                    2,
                    cv2.LINE_AA
                )

            st.image(
                cv2.cvtColor(
                    output,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

            st.success(
                f"Detected {len(results)} face(s)"
            )

            for i, (
                gender,
                age,
                confidence
            ) in enumerate(results, 1):

                st.write(
                    f"**Face {i}:** "
                    f"{gender}, "
                    f"Age {age}, "
                    f"Confidence "
                    f"{confidence * 100:.1f}%"
                )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.markdown(
        '<div class="section-title">Model Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <h2>VisionAI Multi-Task CNN</h2>

        <p>
        The model performs two predictions from the
        same facial image.
        </p>

        <hr>

        <h3>Input</h3>

        <p>128 × 128 × 3 RGB image</p>

        <h3>Output 1 — Gender</h3>

        <p>Binary classification using
        Binary Crossentropy.</p>

        <h3>Output 2 — Age</h3>

        <p>Age regression using Huber Loss.</p>

        <h3>Computer Vision Pipeline</h3>

        <p>
        Camera/Image → Face Detection →
        Crop → Resize → Normalize →
        CNN → Gender + Age
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### Model outputs"
    )

    st.code(
        """
gender → (None, 1)
age    → (None, 1)
        """,
        language="text"
    )

    st.markdown(
        "### Technology"
    )

    st.write(
        """
        - TensorFlow / Keras
        - OpenCV
        - Streamlit
        - WebRTC
        - Cloudflare TURN
        - NumPy
        """
    )
