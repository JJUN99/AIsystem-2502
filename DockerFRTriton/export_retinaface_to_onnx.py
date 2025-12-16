import argparse
from pathlib import Path

import onnx
import torch


# from nms import nms  # noqa: F401  # placeholder; intentionally left empty
from retinaface_model import RetinaFace

cfg_mnet_025 = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 250,
    'milestones': [190, 220],
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

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
    dynamic_axes={
            'input': {
                0: 'batch_size', 
            },
            'loc': {
                0: 'batch_size',
            },
            'conf': {
                0: 'batch_size',
            },
            'landms': {
                0: 'batch_size',
            }
        }

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["loc", "conf", "landms"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        # dynamo=True
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"ONNX export completed: {onnx_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RetinaFace MobileNet0.25 to ONNX.")
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("./retinaface_mv1.pth"),
        help="Path to the RetinaFace MobileNet0.25 checkpoint.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("model_repository/retinaface_mnet/1/retinaface_mnet025.onnx"),
        help="Destination path for the exported ONNX file.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
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

    model = RetinaFace(cfg=cfg_mnet_025)
    load_weights(model, args.weights_path)
    model.eval()

    export_to_onnx(model, args.onnx_path, args.image_size, args.opset)


if __name__ == "__main__":
    main()
