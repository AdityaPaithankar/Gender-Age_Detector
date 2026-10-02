import os
import time
import threading
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
# UI HELPERS
# ============================================================

def stat_card(title, value):
    # IMPORTANT: no blank lines and no deep indentation inside the HTML,
    # otherwise Markdown renders parts of it as a code block.
    st.markdown(
        f'<div class="card">'
        f'<div class="stat-title">{title}</div>'
        f'<div class="stat-value">{value}</div>'
        f'</div>',
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
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Resize
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

    # Normalize
    face = face.astype("float32") / 255.0

    # Batch dimension
    face = np.expand_dims(face, axis=0)

    # Model prediction
    prediction = model.predict(face, verbose=0)

    gender_pred = float(prediction[0][0][0])
    age_pred = float(prediction[1][0][0])

    # Gender
    if gender_pred >= GENDER_THRESHOLD:
        gender = "Female"
    else:
        gender = "Male"

    # Age
    age = int(round(age_pred))

    # Confidence
    gender_confidence = (
        gender_pred
        if gender_pred >= 0.5
        else 1 - gender_pred
    )

    return gender, age, gender_confidence


# ============================================================
# WEBRTC VIDEO PROCESSOR
# ============================================================
#
# Speed strategy:
#   1. recv() only draws cached results, so video never waits on the model.
#   2. A background thread runs detection + prediction on the latest frame
#      (older frames are dropped instead of queued).
#   3. Face detection runs on a downscaled frame.
#   4. All faces are classified in a single batched model call, capped
#      at MAX_FACES (largest first).

DETECT_WIDTH = 320
MAX_FACES = 3


class VisionAIProcessor(VideoProcessorBase):

    def __init__(self):

        self.model = load_model()
        self.detector = load_face_detector()

        self.face_count = 0
        self.gender = "-"
        self.age = "-"
        self.confidence = 0.0

        self._lock = threading.Lock()
        self._latest_frame = None
        self._results = []
        self._running = True

        self._thread = threading.Thread(
            target=self._worker,
            daemon=True
        )
        self._thread.start()

    def on_ended(self):
        self._running = False

    # --------------------------------------------------------
    # Background inference
    # --------------------------------------------------------

    def _worker(self):

        while self._running:

            with self._lock:
                img = self._latest_frame
                self._latest_frame = None

            if img is None:
                time.sleep(0.01)
                continue

            try:
                self._process(img)
            except Exception:
                continue

    def _process(self, img):

        h_img, w_img = img.shape[:2]

        # Detect on a smaller frame
        scale = min(1.0, DETECT_WIDTH / w_img)

        small = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(60 * scale), int(60 * scale))
        )

        if len(faces) == 0:
            with self._lock:
                self._results = []
                self.face_count = 0
            return

        # Largest faces first
        faces = sorted(
            faces,
            key=lambda f: f[2] * f[3],
            reverse=True
        )[:MAX_FACES]

        boxes = []
        batch = []

        for (x, y, w, h) in faces:

            # Map back to full-resolution coordinates
            x, y, w, h = [int(v / scale) for v in (x, y, w, h)]

            margin_x = int(w * 0.10)
            margin_y = int(h * 0.10)

            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w_img, x + w + margin_x)
            y2 = min(h_img, y + h + margin_y)

            face = img[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face.astype("float32") / 255.0

            batch.append(face)
            boxes.append((x1, y1, x2, y2))

        if not batch:
            return

        # One batched call for all faces
        outputs = self.model(
            np.stack(batch),
            training=False
        )

        gender_preds = np.asarray(outputs[0]).reshape(-1)
        age_preds = np.asarray(outputs[1]).reshape(-1)

        results = []

        for box, g, a in zip(boxes, gender_preds, age_preds):

            g = float(g)

            gender = "Female" if g >= GENDER_THRESHOLD else "Male"
            age = int(round(float(a)))
            confidence = g if g >= 0.5 else 1 - g

            results.append((box, gender, age, confidence))

        with self._lock:

            self._results = results
            self.face_count = len(results)

            # Stats panel shows the largest face
            _, self.gender, self.age, self.confidence = results[0]

    # --------------------------------------------------------
    # Fast video path: just draw cached results
    # --------------------------------------------------------

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        with self._lock:
            self._latest_frame = img.copy()
            results = list(self._results)

        for (x1, y1, x2, y2), gender, age, confidence in results:

            label = f"{gender} | Age: {age}"
            confidence_label = f"{confidence * 100:.1f}%"

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.rectangle(
                img,
                (x1, max(0, y1 - 55)),
                (x2, y1),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                img,
                label,
                (x1 + 5, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                confidence_label,
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        return frame.from_ndarray(img, format="bgr24")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<h2>👁️ VisionAI</h2>'
        '<p style="color:#9ca3af;">AI-powered face intelligence</p>',
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
        '<p class="success">● SYSTEM ONLINE</p>'
        '<p style="color:#9ca3af;">TensorFlow • OpenCV • CNN</p>',
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
    '<div class="subtitle">'
    'Real-time gender and age detection using a fine-tuned '
    'multi-task convolutional neural network.'
    '</div>',
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

    st.error(f"VisionAI initialization failed: {e}")


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
            stat_card("Model", "CNN")

        with col2:
            stat_card("Input", "128×128")

        with col3:
            stat_card("Tasks", "2")

        with col4:
            stat_card("Network", "READY")

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
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15}
                },
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

            st.subheader("Live Detection")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Faces", processor.face_count)

            with c2:
                st.metric("Gender", processor.gender)

            with c3:
                st.metric("Age", processor.age)

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

    if not system_ready:

        st.warning(
            "VisionAI could not initialize. "
            "Check the model and OpenCV installation."
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload a face image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:

            image = Image.open(uploaded_file).convert("RGB")

            image_np = np.array(image)

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            result_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            results = []

            for (x, y, w, h) in faces:

                face = result_image[y:y + h, x:x + w]

                gender, age, confidence = predict_face(model, face)

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
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                label = f"{gender} | Age: {age}"

                cv2.putText(
                    result_image,
                    label,
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            st.subheader("Detection Result")

            st.image(
                cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            st.write(f"**Faces detected:** {len(results)}")

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
