import onnxruntime as ort
import numpy as np

from preprocess import preprocess


session = ort.InferenceSession(
    "AntiSpoof.onnx",
    providers=["CPUExecutionProvider"]
)


def predict(image):

    input_image = preprocess(image)

    outputs = session.run(
        None,
        {"input": input_image}
    )

    result = outputs[0][0]

    score = float(np.max(result))

    label = int(np.argmax(result))

    if label == 1:

        return {
            "label": "REAL",
            "score": score
        }

    return {
        "label": "FAKE",
        "score": score
    }