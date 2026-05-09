import onnxruntime as ort
import numpy as np

from preprocess import preprocess


# =========================================
# LOAD ONNX MODEL
# =========================================

session = ort.InferenceSession(
    "AntiSpoof.onnx",
    providers=["CPUExecutionProvider"]
)


# =========================================
# PREDICT FUNCTION
# =========================================

def predict(image):

    input_image = preprocess(image)

    outputs = session.run(
        None,
        {"input": input_image}
    )

    result = outputs[0][0]

    score = float(np.max(result))

    label = int(np.argmax(result))

    # =====================================
    # LABEL MAPPING
    # =====================================

    # Most models:
    # 0 = REAL
    # 1 = FAKE

    if label == 0:

        return {
            "label": "REAL",
            "score": score
        }

    return {
        "label": "FAKE",
        "score": score
    }