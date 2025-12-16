import argparse
from pathlib import Path

import onnx
import torch

# from nms import nms  # noqa: F401  # placeholder; intentionally left empty
from spoofnet import SpoofNet

def load_weights(model: torch.nn.Module, weights_path: Path) -> None:
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}

    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        print(f"Missing keys when loading weights (check width_mult matches): {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"Unexpected keys when loading weights: {incompatible.unexpected_keys}")


def export_to_onnx(model: torch.nn.Module, onnx_path: Path, image_size: int, opset: int) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, image_size, image_size)
    dynamic_axes = {
        "input": {0: "batch"},
        "x": {0: "batch"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["x"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"ONNX export completed: {onnx_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export spoofnet to ONNX.")
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("./mobilenetv2-best.pt"),
        help="Path to the spoofnet MobileNetv2 checkpoint.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("model_repository/spoofnet/1/model.onnx"),
        help="Destination path for the exported ONNX file.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Export resolution (square). Adjust to match your inference input size.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # weights_path = download_weights(args.weights_path)

    model = SpoofNet()
    load_weights(model, args.weights_path)
    model.eval()

    export_to_onnx(model, args.onnx_path, args.image_size, args.opset)


if __name__ == "__main__":
    main()
