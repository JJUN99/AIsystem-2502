import argparse
from pathlib import Path

import onnx
import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def download_weights(dest: Path) -> Path:
    """Download MobileNetV2 weights to dest if missing."""
    if dest.exists():
        print(f"Using existing weights: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    weights_enum = MobileNet_V2_Weights.IMAGENET1K_V1
    print(f"Downloading weights from {weights_enum.url} ...")
    torch.hub.download_url_to_file(weights_enum.url, dest)
    print(f"Saved weights to {dest}")
    return dest


def build_model(weights_path: Path) -> torch.nn.Module:
    state_dict = torch.load(weights_path, map_location="cpu")
    model = mobilenet_v2(weights=None)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_to_onnx(model: torch.nn.Module, onnx_path: Path, opset: int) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    dynamic_axes = {"input": {0: "batch_size"}, "logits": {0: "batch_size"}}

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"ONNX export completed: {onnx_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MobileNetV2 to ONNX.")
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("weights/mobilenet_v2-b0353104.pth"),
        help="Where to store or read the .pth weights file.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("artifacts/mobilenet_v2.onnx"),
        help="Destination for the ONNX model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version to export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = download_weights(args.weights_path)
    model = build_model(weights_path)
    export_to_onnx(model, args.onnx_path, args.opset)


if __name__ == "__main__":
    main()
