import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VisionAI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "final_model.keras"
)


# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = 128
GENDER_THRESHOLD = 0.5


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 17px;
        margin-top: 5px;
        margin-bottom: 35px;
    }

    .card {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .stat-title {
        color: #9ca3af;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stat-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 8px;
    }

    .success {
        color: #22c55e;
        font-weight: 600;
    }

    .warning {
        color: #f59e0b;
        font-weight: 600;
    }

    .prediction {
        font-size: 30px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# FACE DETECTOR
# ============================================================

@st.cache_resource
def load_face_detector():

    # Use OpenCV's bundled Haar Cascade.
    # No custom assets folder required.

    cascade_path = os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml"
    )

    if not os.path.exists(cascade_path):
        raise FileNotFoundError(
            f"Haar Cascade not found at: {cascade_path}"
        )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(
            "OpenCV could not initialize Haar Cascade."
        )

    return detector


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
    face = face.astype("float32") / 255.0

    # Batch dimension
    face = np.expand_dims(
        face,
        axis=0
    )

    # Model prediction
    prediction = model.predict(
        face,
        verbose=0
    )

    gender_pred = float(
        prediction[0][0][0]
    )

    age_pred = float(
        prediction[1][0][0]
    )

    # Gender
    if gender_pred >= GENDER_THRESHOLD:
        gender = "Female"
    else:
        gender = "Male"

    # Age
    age = int(
        round(age_pred)
    )

    # Confidence
    gender_confidence = (
        gender_pred
        if gender_pred >= 0.5
        else 1 - gender_pred
    )

    return (
        gender,
        age,
        gender_confidence
    )


# ============================================================
# WEBRTC VIDEO PROCESSOR
# ============================================================

class VisionAIProcessor(VideoProcessorBase):

    def __init__(self):

        self.model = load_model()
        self.detector = load_face_detector()

        self.face_count = 0
        self.gender = "-"
        self.age = "-"
        self.confidence = 0.0

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        self.face_count = len(faces)

        for (x, y, w, h) in faces:

            # Add small margin
            margin_x = int(w * 0.10)
            margin_y = int(h * 0.10)

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

            try:

                gender, age, confidence = predict_face(
                    self.model,
                    face
                )

                self.gender = gender
                self.age = age
                self.confidence = confidence

                label = (
                    f"{gender} | Age: {age}"
                )

                confidence_label = (
                    f"{confidence * 100:.1f}%"
                )

                # Face box
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Background for label
                cv2.rectangle(
                    img,
                    (x1, max(0, y1 - 55)),
                    (x2, y1),
                    (0, 0, 0),
                    -1
                )

                # Prediction
                cv2.putText(
                    img,
                    label,
                    (x1 + 5, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                # Confidence
                cv2.putText(
                    img,
                    confidence_label,
                    (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

            except Exception:
                continue

        return frame.from_ndarray(
            img,
            format="bgr24"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h2>👁️ VisionAI</h2>
        <p style="color:#9ca3af;">
        AI-powered face intelligence
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Live Analysis",
            "Image Analysis",
            "Model Information"
        ]
    )

    st.divider()

    st.markdown(
        """
        <p class="success">
        ● SYSTEM ONLINE
        </p>

        <p style="color:#9ca3af;">
        TensorFlow • OpenCV • CNN
        </p>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">VisionAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Real-time gender and age detection using a fine-tuned
    multi-task convolutional neural network.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INITIALIZATION
# ============================================================

try:

    model = load_model()
    detector = load_face_detector()

    system_ready = True

except Exception as e:

    system_ready = False

    st.error(
        f"VisionAI initialization failed: {e}"
    )


# ============================================================
# LIVE ANALYSIS
# ============================================================

if page == "Live Analysis":

    st.header("Live Analysis")

    if not system_ready:

        st.warning(
            "VisionAI could not initialize. "
            "Check the model and OpenCV installation."
        )

    else:

        # ====================================================
        # MODEL STATS
        # ====================================================

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
                """
                <div class="card">
                    <div class="stat-title">
                        Network
                    </div>

                    <div class="stat-value">
                        READY
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # CAMERA
        # ====================================================

        st.subheader("Camera Feed")

        st.info(
            "Allow camera access when your browser asks for permission."
        )

        rtc_configuration = RTCConfiguration(
            {
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.cloudflare.com:3478"
                        ]
                    }
                ]
            }
        )

        ctx = webrtc_streamer(
            key="visionai",
            video_processor_factory=VisionAIProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

        # ====================================================
        # DETECTION INFO
        # ====================================================

        if ctx.video_processor:

            processor = ctx.video_processor

            st.divider()

            st.subheader(
                "Live Detection"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Faces",
                    processor.face_count
                )

            with c2:

                st.metric(
                    "Gender",
                    processor.gender
                )

            with c3:

                st.metric(
                    "Age",
                    processor.age
                )

            with c4:

                st.metric(
                    "Confidence",
                    f"{processor.confidence * 100:.1f}%"
                )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

elif page == "Image Analysis":

    st.header("Image Analysis")

    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_np = np.array(image)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        gray = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2GRAY
        )

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        result_image = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )

        results = []

        for (x, y, w, h) in faces:

            face = result_image[
                y:y+h,
                x:x+w
            ]

            gender, age, confidence = predict_face(
                model,
                face
            )

            results.append(
                {
                    "gender": gender,
                    "age": age,
                    "confidence": confidence
                }
            )

            cv2.rectangle(
                result_image,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            label = (
                f"{gender} | Age: {age}"
            )

            cv2.putText(
                result_image,
                label,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        st.subheader(
            "Detection Result"
        )

        st.image(
            cv2.cvtColor(
                result_image,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

        st.write(
            f"**Faces detected:** {len(results)}"
        )

        for i, result in enumerate(results):

            st.write(
                f"**Face {i + 1}:** "
                f"{result['gender']} | "
                f"Age: {result['age']} | "
                f"Confidence: "
                f"{result['confidence'] * 100:.1f}%"
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.header("Model Information")

    st.markdown(
        """
        ### VisionAI

        VisionAI uses a **multi-task CNN** to perform:

        - Gender classification
        - Age prediction
        - Face detection using OpenCV Haar Cascade

        ### Input

        `128 × 128 × 3`

        ### Outputs

        ```text
        gender
        age
        ```

        ### Preprocessing

        - Face detection
        - Face cropping
        - RGB conversion
        - Resize to 128×128
        - Pixel normalization to `[0, 1]`

        ### Technology

        - Python
        - TensorFlow / Keras
        - OpenCV
        - Streamlit
        - WebRTC
        - NumPy
        """
    )
