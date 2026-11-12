"""Webcam: NanoDet localiza a trena e o terminal diz o rumo."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import config
from commander import Commander
from infer import Int8TapeDetector, TapeDetector
from tracker import Tracker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rumo da trena pela webcam.")
    parser.add_argument(
        "--mode",
        choices=("float", "int8"),
        default="float",
        help="float usa o checkpoint PyTorch. int8 usa o NCNN quantizado.",
    )
    return parser.parse_args()


def _int8_files(here: Path) -> tuple[Path, Path]:
    folders = [here, here.parent / "optimization" / "out"]
    for folder in folders:
        param = folder / config.INT8_PARAM
        weights = folder / config.INT8_BIN
        if param.is_file() and weights.is_file():
            return param, weights
    looked = ", ".join(str(folder) for folder in folders)
    raise SystemExit(f"Não achei {config.INT8_PARAM} e {config.INT8_BIN} em {looked}.")


def _draw(frame, state, command: str, fps: float, mode: str):
    view = frame.copy()
    if len(state.history) >= 2:
        trail = np.array(
            [(int(sample.cx), int(sample.cy)) for sample in state.history],
            dtype=np.int32,
        )
        cv2.polylines(view, [trail], isClosed=False, color=(0, 200, 255), thickness=2)
    if state.bbox is not None and state.cx is not None and state.cy is not None:
        x, y, w, h = state.bbox
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.circle(view, (int(state.cx), int(state.cy)), 4, (0, 220, 0), -1)

    width_txt = f"{state.width_ratio:.2f}" if state.width_ratio is not None else "-"
    nx_txt = f"{state.nx:+.2f}" if state.nx is not None else "-"
    lines = [
        command,
        f"nx {nx_txt}  largura {width_txt}",
        f"parado acima de {config.NEAR_ENTER:.2f}",
        f"fps {fps:.1f}  {mode}",
    ]
    y0 = 28
    for line in lines:
        cv2.putText(view, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y0 += 26
    return view


def main() -> None:
    args = _parse_args()
    here = Path(__file__).resolve().parent
    config_path = here / config.CONFIG_PATH
    if args.mode == "int8":
        param_path, bin_path = _int8_files(here)
        detector = Int8TapeDetector(param_path, bin_path, config_path)
    else:
        model_path = here / config.MODEL_PATH
        if not model_path.exists():
            raise SystemExit(f"Não achei o modelo em {model_path}")
        detector = TapeDetector(model_path, config_path)
    tracker = Tracker()
    commander = Commander()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        raise SystemExit(f"Não abriu a webcam (índice {config.CAMERA_INDEX}).")
    print("q na janela encerra.", flush=True)

    fps = 0.0
    last = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame vazio da webcam.", flush=True)
                break
            box = detector.detect(frame)
            state = tracker.update(box, frame.shape[1], frame.shape[0])
            command = commander.update(state)
            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                instant = 1.0 / dt
                fps = instant if fps <= 0 else 0.9 * fps + 0.1 * instant
            cv2.imshow("jimbot", _draw(frame, state, command, fps, args.mode))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
