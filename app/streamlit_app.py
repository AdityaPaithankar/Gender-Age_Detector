import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from pathlib import Path
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VisionAI",
    page_icon="👁️",
    layout="wide",
)


# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "final_model.keras"
CASCADE_PATH = (
    BASE_DIR
    / "assets"
    / "haarcascade_frontalface_default.xml"
)

IMG_SIZE = 128
THRESHOLD = 0.5


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #090d12;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .hero-subtitle {
        color: #8b95a5;
        font-size: 15px;
        margin-top: 4px;
    }

    .card {
        background: #11161d;
        border: 1px solid #252d37;
        border-radius: 16px;
        padding: 20px;
    }

    .card-title {
        color: #8b95a5;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .card-value {
        font-size: 30px;
        font-weight: 750;
        margin-top: 7px;
    }

    .online {
        color: #62e58a;
        font-weight: 600;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ============================================================
# FACE DETECTOR
# ============================================================

@st.cache_resource
def load_face_detector():

    if not CASCADE_PATH.exists():
        raise FileNotFoundError(
            f"Haar Cascade file not found:\n{CASCADE_PATH}"
        )

    detector = cv2.CascadeClassifier(
        str(CASCADE_PATH)
    )

    if detector.empty():
        raise RuntimeError(
            f"Failed to load Haar Cascade:\n{CASCADE_PATH}"
        )

    return detector


# ============================================================
# LOAD RESOURCES
# ============================================================

try:

    model = load_model()
    face_detector = load_face_detector()

except Exception as e:

    st.error("Failed to initialize VisionAI.")

    st.code(str(e))

    st.stop()


# ============================================================
# PREDICTION
# ============================================================

def predict_face(face):

    # BGR → RGB
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
        "float32"
    ) / 255.0

    # Batch dimension
    face = np.expand_dims(
        face,
        axis=0
    )

    # Model prediction
    gender_pred, age_pred = model.predict(
        face,
        verbose=0
    )

    gender_score = float(
        gender_pred[0][0]
    )

    age = int(
        round(
            float(age_pred[0][0])
        )
    )

    # IMPORTANT:
    # Keep this mapping identical to your
    # working webcam.py.
    gender = (
        "Female"
        if gender_score >= THRESHOLD
        else "Male"
    )

    # Confidence
    confidence = (
        gender_score
        if gender == "Male"
        else 1 - gender_score
    )

    # Keep age within a sensible range
    age = max(0, min(age, 120))

    return (
        gender,
        age,
        confidence
    )


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.face_count = 0
        self.gender = "-"
        self.age = "-"
        self.confidence = 0.0

    def recv(self, frame):

        # WebRTC frame → OpenCV image
        img = frame.to_ndarray(
            format="bgr24"
        )

        # BGR → grayscale
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # Face detection
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        self.face_count = len(faces)

        # If no face is detected
        if len(faces) == 0:

            self.gender = "-"
            self.age = "-"
            self.confidence = 0.0

        # Process detected faces
        for (x, y, w, h) in faces:

            face = img[
                y:y + h,
                x:x + w
            ]

            if face.size == 0:
                continue

            gender, age, confidence = predict_face(
                face
            )

            self.gender = gender
            self.age = age
            self.confidence = confidence

            # ==================================================
            # BOUNDING BOX
            # ==================================================

            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (0, 255, 120),
                2
            )

            # ==================================================
            # LABEL
            # ==================================================

            label = (
                f"{gender} | Age: {age}"
            )

            label_y = max(
                35,
                y
            )

            # Label background
            cv2.rectangle(
                img,
                (x, label_y - 35),
                (x + w, label_y),
                (0, 255, 120),
                -1
            )

            # Label text
            cv2.putText(
                img,
                label,
                (x + 8, label_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        # OpenCV image → WebRTC frame
        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 👁️ VisionAI"
    )

    st.caption(
        "AI-powered face intelligence"
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
        '<span class="online">● SYSTEM ONLINE</span>',
        unsafe_allow_html=True
    )

    st.caption(
        "TensorFlow • OpenCV • CNN"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero-title">VisionAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Real-time gender and age detection '
    'using a multi-task CNN.'
    '</div>',
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

    left, right = st.columns(
        [2.2, 1],
        gap="large"
    )

    # ========================================================
    # CAMERA
    # ========================================================

    with left:

        st.markdown(
            '<div class="card-title">CAMERA FEED</div>',
            unsafe_allow_html=True
        )

        ctx = webrtc_streamer(

            key="visionai-camera",

            video_processor_factory=VideoProcessor,

            media_stream_constraints={
                "video": True,
                "audio": False,
            },

            # IMPORTANT FOR STREAMLIT CLOUD
            rtc_configuration={
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302"
                        ]
                    }
                ]
            },

            async_processing=True,
        )


    # ========================================================
    # LIVE DETECTION
    # ========================================================

    with right:

        st.markdown(
            '<div class="card">'
            '<div class="card-title">'
            'LIVE DETECTION'
            '</div>'
            '<br>'
            '<b>Face Detection</b>'
            '<br><br>'
            'Face count and predictions are shown '
            'while the camera is running.'
            '</div>',
            unsafe_allow_html=True
        )

        st.write("")

        if ctx.video_processor:

            processor = ctx.video_processor

            # Face + Gender
            c1, c2 = st.columns(2)

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

            # Age + Confidence
            c3, c4 = st.columns(2)

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


    # ========================================================
    # DETECTION OVERVIEW
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Detection Overview'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    MODEL
                </div>
                <div class="card-value">
                    CNN
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    INPUT
                </div>
                <div class="card-value">
                    128×128
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    TASKS
                </div>
                <div class="card-value">
                    2
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    STATUS
                </div>
                <div class="card-value">
                    READY
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

    uploaded = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded:

        # Load image
        image = np.array(
            Image.open(
                uploaded
            ).convert("RGB")
        )

        # RGB → BGR
        frame = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        # Grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Face detection
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # Process faces
        for x, y, w, h in faces:

            face = frame[
                y:y + h,
                x:x + w
            ]

            if face.size == 0:
                continue

            gender, age, confidence = predict_face(
                face
            )

            # Bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 120),
                2
            )

            # Label
            label = (
                f"{gender} | Age: {age}"
            )

            cv2.putText(
                frame,
                label,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 120),
                2
            )

        # BGR → RGB
        result = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Display
        st.image(
            result,
            use_container_width=True
        )

        st.success(
            f"{len(faces)} face(s) detected."
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

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Input Size",
            "128 × 128 × 3"
        )

    with c2:

        st.metric(
            "Outputs",
            "2"
        )

    with c3:

        st.metric(
            "Framework",
            "TensorFlow"
        )

    st.divider()

    st.markdown(
        "### Architecture"
    )

    st.write(
        """
        The model uses a convolutional neural network
        with multiple convolutional blocks followed by
        global average pooling and separate output heads
        for gender classification and age regression.
        """
    )

    st.markdown(
        "### Tasks"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            "**Gender Classification**\n\n"
            "Binary classification using "
            "binary crossentropy."
        )

    with col2:

        st.info(
            "**Age Regression**\n\n"
            "Continuous age prediction using "
            "Huber loss."
        )
