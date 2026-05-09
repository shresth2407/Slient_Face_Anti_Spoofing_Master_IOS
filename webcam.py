import cv2
import time
import numpy as np

from detector import predict


# =========================================
# FACE DETECTOR
# =========================================

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)


# =========================================
# OPEN WEBCAM
# =========================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():

    print("Cannot open webcam")

    exit()

print("Press ESC to exit")


# =========================================
# STABLE PREDICTION VARIABLES
# =========================================

prev_label = ""

stable_count = 0


# =========================================
# MAIN LOOP
# =========================================

while True:

    start_time = time.time()

    ret, frame = cap.read()

    if not ret:
        break

    # mirror effect
    frame = cv2.flip(frame, 1)

    # grayscale for face detection
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # =========================================
    # NO FACE
    # =========================================

    if len(faces) == 0:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # =========================================
    # PROCESS EACH FACE
    # =========================================

    for (x, y, w, h) in faces:

        # crop face
        face = frame[y:y+h, x:x+w]

        try:

            # ======================================
            # PREPROCESS
            # ======================================

            gray_face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2GRAY
            )

            # ======================================
            # SHARPNESS CHECK
            # ======================================

            laplacian_var = cv2.Laplacian(
                gray_face,
                cv2.CV_64F
            ).var()

            # ======================================
            # BRIGHTNESS CHECK
            # ======================================

            brightness = np.mean(gray_face)

            # ======================================
            # EDGE DENSITY CHECK
            # ======================================

            edges = cv2.Canny(
                gray_face,
                100,
                200
            )

            edge_density = np.mean(edges)

            # ======================================
            # SPOOF DETECTION RULES
            # ======================================

            spoof_detected = False

            # smoother spoof checks

            if laplacian_var < 40:
                spoof_detected = True

            if brightness > 240:
                spoof_detected = True

            if edge_density > 60:
                spoof_detected = True

            # ======================================
            # FINAL DECISION
            # ======================================

            if spoof_detected:

                label = "FAKE"

                score = 0.99

            else:

                result = predict(face)

                label = result["label"]

                score = result["score"]

            # ======================================
            # STABLE PREDICTION
            # ======================================

            if label == prev_label:

                stable_count += 1

            else:

                stable_count = 0

            prev_label = label

            # avoid flickering
            if stable_count < 2:
                continue

            # ======================================
            # COLORS
            # ======================================

            if label == "REAL":

                color = (0, 255, 0)

            else:

                color = (0, 0, 255)

            # ======================================
            # TEXT
            # ======================================

            text = f"{label} : {score:.2f}"

            # ======================================
            # DRAW FACE BOX
            # ======================================

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                color,
                2
            )

            # ======================================
            # DRAW RESULT TEXT
            # ======================================

            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        except Exception as e:

            print("Detection Error:", e)

    # =========================================
    # FPS
    # =========================================

    fps = 1 / (time.time() - start_time)

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # =========================================
    # SHOW WINDOW
    # =========================================

    cv2.imshow(
        "Face Anti Spoofing",
        frame
    )

    key = cv2.waitKey(1)

    # ESC
    if key == 27:
        break


# =========================================
# RELEASE
# =========================================

cap.release()

cv2.destroyAllWindows()