import torch

from src.model_lib.MiniFASNet import MiniFASNetV2


MODEL_PATH = "./resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"

OUTPUT_PATH = "AntiSpoof.onnx"


device = torch.device("cpu")


# Create model
model = MiniFASNetV2(conv6_kernel=(5, 5)).to(device)

# Load weights
state_dict = torch.load(MODEL_PATH, map_location=device)

new_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        new_key = key[7:]
    else:
        new_key = key

    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)

model.eval()


# Dummy input
dummy_input = torch.randn(1, 3, 80, 80)


# Export ONNX
torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_PATH,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

print("ONNX model exported successfully!")