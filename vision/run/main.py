"""Webcam: NanoDet localiza a trena e o terminal diz o rumo."""

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import config
from commander import Commander
from infer import NcnnTapeDetector, TapeDetector
from tracker import Tracker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rumo da trena pela webcam.")
    parser.add_argument(
        "--mode",
        choices=("float", "fp32", "int8"),
        default="float",
        help="float usa o checkpoint PyTorch. fp32 usa o NCNN em ponto flutuante. int8 usa o NCNN quantizado.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Não abre a janela com o resultado.",
    )
    parser.add_argument(
        "--no-serial",
        action="store_true",
        help="Não abre a porta serial. O rumo só aparece no terminal.",
    )
    return parser.parse_args()


def _ncnn_files(here: Path, param_name: str, bin_name: str) -> tuple[Path, Path]:
    folders = [here, here.parent / "optimization" / "out"]
    for folder in folders:
        param = folder / param_name
        weights = folder / bin_name
        if param.is_file() and weights.is_file():
            return param, weights
    looked = ", ".join(str(folder) for folder in folders)
    raise SystemExit(f"Não achei {param_name} e {bin_name} em {looked}.")


class _LatestCapture:
    """Esvazia a fila da câmera e guarda só o frame mais recente."""

    def __init__(self, index: int, width: int, height: int) -> None:
        self._cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            self._cap.release()
            raise SystemExit(f"Não abriu a webcam (índice {index}).")
        self.queue_size = int(self._cap.get(cv2.CAP_PROP_BUFFERSIZE))
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._seen = 0
        self._failed = False
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            with self._cond:
                if self._stop:
                    return
            try:
                ok, frame = self._cap.read()
            except cv2.error:
                ok, frame = False, None
            with self._cond:
                if self._stop:
                    return
                if not ok or frame is None:
                    self._failed = True
                    self._cond.notify_all()
                    return
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self):
        with self._cond:
            while self._seq == self._seen and not self._failed and not self._stop:
                self._cond.wait(timeout=0.5)
            if self._seq == self._seen:
                return False, None
            self._seen = self._seq
            return True, self._frame

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._cap.release()
        self._thread.join(timeout=1.0)


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
    if args.mode == "fp32":
        param_path, bin_path = _ncnn_files(here, config.FP32_PARAM, config.FP32_BIN)
        detector = NcnnTapeDetector(param_path, bin_path, config_path)
    elif args.mode == "int8":
        param_path, bin_path = _ncnn_files(here, config.INT8_PARAM, config.INT8_BIN)
        detector = NcnnTapeDetector(param_path, bin_path, config_path)
    else:
        model_path = here / config.MODEL_PATH
        if not model_path.exists():
            raise SystemExit(f"Não achei o modelo em {model_path}")
        detector = TapeDetector(model_path, config_path)
    tracker = Tracker()
    port = None if args.no_serial else config.SERIAL_PORT
    commander = Commander(port=port)
    camera = None
    try:
        camera = _LatestCapture(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        if args.no_display:
            print(f"fila da câmera: {camera.queue_size}. Ctrl+C encerra.", flush=True)
        else:
            print(f"fila da câmera: {camera.queue_size}. q na janela encerra.", flush=True)

        fps = 0.0
        last = time.perf_counter()
        last_report = last
        while True:
            ok, frame = camera.read()
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
            if args.no_display:
                if now - last_report >= 1.0:
                    print(f"fps {fps:.1f}  {args.mode}", flush=True)
                    last_report = now
                continue
            cv2.imshow("jimbot", _draw(frame, state, command, fps, args.mode))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if camera is not None:
            camera.close()
        commander.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
