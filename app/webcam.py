import cv2
import numpy as np
import tensorflow as tf


MODEL_PATH = "model/final_model.keras"
IMG_SIZE = 128
THRESHOLD = 0.5


def load_model():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    return model


def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    return face


def predict(model, face):
    processed_face = preprocess_face(face)

    gender_pred, age_pred = model.predict(
        processed_face,
        verbose=0
    )

    gender_score = float(gender_pred[0][0])
    age = int(round(float(age_pred[0][0])))

    # Your corrected mapping
    gender = (
        "Female"
        if gender_score >= THRESHOLD
        else "Male"
    )

    return gender, age


def main():

    model = load_model()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar Cascade.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Webcam started.")
    print("Press Q to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame.")
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for x, y, w, h in faces:

            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            gender, age = predict(
                model,
                face
            )

            label = f"{gender} | Age: {age}"

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "Gender & Age Detector",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()