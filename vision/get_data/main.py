"""Loop da webcam: rastreia a trena e imprime o comando no terminal."""

import cv2
import numpy as np

import config
from commander import Commander
from dataset_writer import DatasetWriter
from detector import detect
from tracker import Tracker


def _draw(frame: np.ndarray, state, command: str, saved: int) -> np.ndarray:
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

    width_txt = (
        f"{state.width_ratio:.2f}" if state.width_ratio is not None else "-"
    )
    nx_txt = f"{state.nx:+.2f}" if state.nx is not None else "-"
    score_txt = f"{state.score:.2f}" if state.score is not None else "-"
    lines = [
        command,
        f"nx {nx_txt}  largura {width_txt}  score {score_txt}",
        f"centro |nx|<{config.CENTER_DEADZONE:.2f}  perto >{config.NEAR_ENTER:.2f}",
        f"dataset {saved}/{config.DATASET_TARGET}",
    ]
    y0 = 28
    for line in lines:
        cv2.putText(
            view,
            line,
            (12, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            view,
            line,
            (12, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y0 += 26
    return view


def main() -> None:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        raise SystemExit(
            f"Não abriu a webcam (índice {config.CAMERA_INDEX})."
        )

    tracker = Tracker()
    commander = Commander()
    dataset = DatasetWriter()
    print(
        f"dataset {dataset.count}/{config.DATASET_TARGET}. q na janela encerra.",
        flush=True,
    )
    if dataset.done:
        print("Dataset já está completo.", flush=True)
        cap.release()
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame vazio da webcam.", flush=True)
                break

            found = detect(frame, tracker.last_center())
            dataset.observe(frame, found.detection)
            state = tracker.update(found.detection, frame.shape[1], frame.shape[0])
            command = commander.update(state)

            view = _draw(frame, state, command, dataset.count)
            mask_bgr = cv2.cvtColor(found.mask, cv2.COLOR_GRAY2BGR)
            spatial_bgr = cv2.cvtColor(found.spatial, cv2.COLOR_GRAY2BGR)
            cv2.putText(mask_bgr, "proposta", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(spatial_bgr, "forma", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            debug = np.hstack([mask_bgr, spatial_bgr])

            cv2.imshow("jimbot", view)
            cv2.imshow("jimbot-debug", debug)
            key = cv2.waitKey(1) & 0xFF
            if dataset.done:
                print("250 imagens gravadas.", flush=True)
                break
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
