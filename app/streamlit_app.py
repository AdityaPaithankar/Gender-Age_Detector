from pathlib import Path
import os
import threading

import av
import cv2
import numpy as np
import requests
import streamlit as st
import tensorflow as tf

from streamlit_webrtc import webrtc_streamer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VisionAI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# IMPORTANT:
# Your GitHub repository contains "model", not "models"
MODEL_PATH = BASE_DIR / "model" / "final_model.keras"

CASCADE_PATH = (
    BASE_DIR
    / "assets"
    / "haarcascade_frontalface_default.xml"
)

IMG_SIZE = 128
GENDER_THRESHOLD = 0.5


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #090d12;
        color: #f5f7fa;
    }

    section[data-testid="stSidebar"] {
        background-color: #10161d;
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
        margin-bottom: 30px;
    }

    .hero {
        padding: 10px 0 25px 0;
    }

    .hero h1 {
        font-size: 42px;
        font-weight: 800;
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
        padding: 20px;
        margin-bottom: 15px;
    }

    .stat-title {
        color: #7f8b99;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stat-value {
        font-size: 28px;
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

    .result {
        background: #11171e;
        border: 1px solid #242d37;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }

    .result-number {
        font-size: 30px;
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

    return tf.keras.models.load_model(
        str(MODEL_PATH)
    )


# ============================================================
# LOAD HAAR CASCADE
# ============================================================

@st.cache_resource
def load_face_detector():

    if not CASCADE_PATH.exists():
        raise FileNotFoundError(
            f"Haar Cascade file not found: {CASCADE_PATH}"
        )

    detector = cv2.CascadeClassifier(
        str(CASCADE_PATH)
    )

    if detector.empty():
        raise RuntimeError(
            "Failed to load Haar Cascade."
        )

    return detector


# ============================================================
# CLOUDFLARE TURN
# ============================================================

@st.cache_data(ttl=3600)
def get_ice_servers():

    try:

        turn_key_id = st.secrets[
            "cloudflare"
        ]["turn_key_id"]

        turn_api_token = st.secrets[
            "cloudflare"
        ]["turn_api_token"]

    except Exception:

        turn_key_id = os.getenv(
            "CLOUDFLARE_TURN_KEY_ID"
        )

        turn_api_token = os.getenv(
            "CLOUDFLARE_TURN_API_TOKEN"
        )

    if not turn_key_id or not turn_api_token:

        return [
            {
                "urls": [
                    "stun:stun.cloudflare.com:3478"
                ]
            }
        ]

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

    return data["iceServers"]


# ============================================================
# SHARED DETECTION STATE
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

    # Batch dimension
    face = np.expand_dims(
        face,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        face,
        verbose=0
    )

    # Your model outputs:
    #
    # prediction[0] -> gender
    # prediction[1] -> age

    gender_prediction = float(
        prediction[0][0][0]
    )

    age_prediction = float(
        prediction[1][0][0]
    )

    # IMPORTANT:
    #
    # 0 = Female
    # 1 = Male
    #
    # based on your training setup.

    if gender_prediction >= GENDER_THRESHOLD:

        gender = "Male"

    else:

        gender = "Female"

    # Confidence
    if gender_prediction >= 0.5:

        confidence = gender_prediction

    else:

        confidence = 1.0 - gender_prediction

    # Age
    age = int(
        round(age_prediction)
    )

    age = max(
        0,
        min(
            100,
            age
        )
    )

    return (
        gender,
        age,
        confidence
    )


# ============================================================
# WEBRTC VIDEO CALLBACK
# ============================================================

def video_frame_callback(frame):

    img = frame.to_ndarray(
        format="bgr24"
    )

    try:

        model = load_model()

        face_detector = load_face_detector()

        # Convert to grayscale
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # Detect faces
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        results = []

        for (x, y, w, h) in faces:

            # ------------------------------------------------
            # Face margin
            # ------------------------------------------------

            margin_x = int(
                w * 0.15
            )

            margin_y = int(
                h * 0.15
            )

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

            gender, age, confidence = (
                predict_face(
                    model,
                    face
                )
            )

            results.append(
                (
                    gender,
                    age,
                    confidence
                )
            )

            # ------------------------------------------------
            # Draw face box
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

            cv2.putText(
                img,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 120),
                2,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Update UI state
        # ----------------------------------------------------

        with state_lock:

            detection_state["faces"] = len(
                results
            )

            if results:

                gender, age, confidence = (
                    results[0]
                )

                detection_state[
                    "gender"
                ] = gender

                detection_state[
                    "age"
                ] = age

                detection_state[
                    "confidence"
                ] = confidence

            else:

                detection_state[
                    "gender"
                ] = "-"

                detection_state[
                    "age"
                ] = "-"

                detection_state[
                    "confidence"
                ] = 0.0

    except Exception:

        # Don't stop WebRTC because of one bad frame
        pass

    return av.VideoFrame.from_ndarray(
        img,
        format="bgr24"
    )


# ============================================================
# INITIALIZE VISIONAI
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
        '<div class="subtitle">'
        'AI-powered face intelligence'
        '</div>',
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
        label_visibility="collapsed"
    )

    st.markdown("---")

    if system_ready:

        st.markdown(
            '<p class="online">'
            '● SYSTEM ONLINE'
            '</p>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<p class="offline">'
            '● SYSTEM ERROR'
            '</p>',
            unsafe_allow_html=True
        )

    st.caption(
        "TensorFlow • OpenCV • WebRTC"
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
            powered by a fine-tuned multi-task
            convolutional neural network.
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
        '<div class="section-title">'
        'Live Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # ICE SERVERS
    # --------------------------------------------------------

    try:

        ice_servers = get_ice_servers()

        network_status = "READY"

    except Exception:

        ice_servers = [
            {
                "urls": [
                    "stun:stun.cloudflare.com:3478"
                ]
            }
        ]

        network_status = "STUN"


    # --------------------------------------------------------
    # SYSTEM CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="card">

                <div class="stat-title">
                    Model
                </div>

                <div class="stat-value">
                    CNN
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">

                <div class="stat-title">
                    Input
                </div>

                <div class="stat-value">
                    128×128
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="card">

                <div class="stat-title">
                    Tasks
                </div>

                <div class="stat-value">
                    2
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="card">

                <div class="stat-title">
                    Network
                </div>

                <div class="stat-value">
                    {network_status}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    camera_col, result_col = st.columns(
        [2.3, 1]
    )

    with camera_col:

        st.markdown(
            '<div class="section-title">'
            'Camera Feed'
            '</div>',
            unsafe_allow_html=True
        )

        st.info(
            "Click START and allow camera access."
        )

        webrtc_streamer(
            key="visionai-camera",

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


    # --------------------------------------------------------
    # LIVE RESULTS
    # --------------------------------------------------------

    with result_col:

        st.markdown(
            '<div class="section-title">'
            'Live Detection'
            '</div>',
            unsafe_allow_html=True
        )

        with state_lock:

            faces = detection_state[
                "faces"
            ]

            gender = detection_state[
                "gender"
            ]

            age = detection_state[
                "age"
            ]

            confidence = detection_state[
                "confidence"
            ]


        st.metric(
            "Faces Detected",
            faces
        )

        st.metric(
            "Gender",
            gender
        )

        st.metric(
            "Estimated Age",
            age
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.1f}%"
        )


    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Detection Overview'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="result">

                <div class="stat-title">
                    Gender
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
                    Age
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
                    Detector
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
        '<div class="section-title">'
        'Image Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
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

                gender, age, confidence = (
                    predict_face(
                        model,
                        face
                    )
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
                    (
                        x,
                        max(
                            30,
                            y - 10
                        )
                    ),
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
            ) in enumerate(
                results,
                start=1
            ):

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
        '<div class="section-title">'
        'Model Information'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

            <h2>VisionAI Multi-Task CNN</h2>

            <p>
            VisionAI uses a single neural network
            to perform two tasks from the same
            facial image.
            </p>

            <hr>

            <h3>Input</h3>

            <p>
            128 × 128 × 3 RGB image
            </p>

            <h3>Gender Detection</h3>

            <p>
            Binary classification using
            Binary Crossentropy.
            </p>

            <h3>Age Estimation</h3>

            <p>
            Regression using Huber Loss.
            </p>

            <h3>Computer Vision Pipeline</h3>

            <p>
            Camera → Face Detection →
            Face Crop → Resize → Normalize →
            CNN → Gender + Age
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Model outputs")

    st.code(
        """
gender → (None, 1)
age    → (None, 1)
        """,
        language="text"
    )

    st.markdown("### Technology")

    st.write(
        """
        • TensorFlow / Keras
        • OpenCV
        • Streamlit
        • WebRTC
        • Cloudflare TURN
        • NumPy
        """
    )
