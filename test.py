# -*- coding: utf-8 -*-

import os
import cv2
import time
import warnings
import numpy as np

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings("ignore")

MODEL_DIR = "./resources/anti_spoof_models"


def draw_text(img, text, x, y, color):
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )


def main():

    print("Loading models...")

    device_id = 0

    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    print("Opening webcam...")
    print("Press ESC to exit")

    cap = cv2.VideoCapture(0)

    # webcam size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        image = frame.copy()

        try:

            image_bbox = model_test.get_bbox(image)

            prediction = np.zeros((1, 3))

            # run all anti-spoof models
            for model_name in os.listdir(MODEL_DIR):

                model_path = os.path.join(MODEL_DIR, model_name)

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

                prediction += model_test.predict(img, model_path)

            label = np.argmax(prediction)
            score = prediction[0][label] / 2

            # improve confidence threshold
            if label == 1 and score > 0.80:

                result_text = f"REAL FACE  {score:.2f}"
                color = (0, 255, 0)

            else:

                result_text = f"FAKE FACE  {score:.2f}"
                color = (0, 0, 255)

            # face rectangle
            cv2.rectangle(
                image,
                (image_bbox[0], image_bbox[1]),
                (
                    image_bbox[0] + image_bbox[2],
                    image_bbox[1] + image_bbox[3]
                ),
                color,
                2
            )

            # result text
            draw_text(
                image,
                result_text,
                image_bbox[0],
                image_bbox[1] - 10,
                color
            )

        except Exception as e:

            draw_text(
                image,
                "NO FACE DETECTED",
                20,
                40,
                (0, 0, 255)
            )

        # FPS counter
        frame_count += 1

        elapsed = time.time() - start_time

        if elapsed > 1:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        draw_text(
            image,
            f"FPS: {fps:.1f}",
            20,
            80,
            (255, 255, 0)
        )

        cv2.imshow("Realtime Face Anti-Spoofing", image)

        key = cv2.waitKey(1)

        # ESC key
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()