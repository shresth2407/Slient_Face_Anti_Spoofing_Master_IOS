import coremltools as ct
import onnx


# Load ONNX model
onnx_model = onnx.load("AntiSpoof.onnx")


# Convert ONNX → CoreML
mlmodel = ct.converters.onnx.convert(
    model=onnx_model
)


# Save model
mlmodel.save("AntiSpoof.mlmodel")


print("CoreML model created successfully!")