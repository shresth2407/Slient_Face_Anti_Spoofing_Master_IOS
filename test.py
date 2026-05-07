# -*- coding: utf-8 -*-

import os
import cv2
import time
import warnings
import numpy as np

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')


MODEL_DIR = "./resources/anti_spoof_models"


def main():

    device_id = 0

    print("Loading models...")

    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    print("Opening webcam...")

    cap = cv2.VideoCapture(0)

    # webcam resolution
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Press ESC to exit")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        image = frame.copy()

        try:

            start = time.time()

            # face detection
            image_bbox = model_test.get_bbox(image)

            prediction = np.zeros((1, 3))

            # anti-spoof prediction
            for model_name in os.listdir(MODEL_DIR):

                h_input, w_input, model_type, scale = parse_model_name(model_name)

                param = {
                    "org_img": image,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }

                if scale is None:
                    param["crop"] = False

                img = image_cropper.crop(**param)

                prediction += model_test.predict(
                    img,
                    os.path.join(MODEL_DIR, model_name)
                )

            label = np.argmax(prediction)
            score = prediction[0][label] / 2

            inference_time = time.time() - start

            # REAL FACE
            if label == 1:

                result_text = f"REAL FACE  {score:.2f}"

                color = (0, 255, 0)

            # FAKE FACE
            else:

                result_text = f"FAKE FACE  {score:.2f}"

                color = (0, 0, 255)

            # bounding box
            cv2.rectangle(
                image,
                (image_bbox[0], image_bbox[1]),
                (image_bbox[0] + image_bbox[2],
                 image_bbox[1] + image_bbox[3]),
                color,
                2
            )

            # prediction text
            cv2.putText(
                image,
                result_text,
                (image_bbox[0], image_bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            # FPS
            fps_text = f"FPS: {1/inference_time:.1f}"

            cv2.putText(
                image,
                fps_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

        except Exception as e:

            print("Detection Error:", e)

        cv2.imshow("Realtime Face Anti-Spoofing", image)

        key = cv2.waitKey(1)

        # ESC key
        if key == 27:
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()