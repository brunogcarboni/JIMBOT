"""Exporta o NanoDet da trena e quantiza para NCNN int8.

A calibração roda no computador que tem o checkpoint e o PyTorch. O
.param/.bin int8 é o mesmo arquivo em x86 e em aarch64: a Raspberry só
precisa da libncnn compilada para ela, na hora de inferir. A versão das
tools que geram o arquivo e a libncnn da placa têm que ser a mesma.

No PC, com a venv do projeto:

    pip install pnnx
    git clone --recursive https://github.com/Tencent/ncnn.git
    cmake -S ncnn -B ncnn/build -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF \
        -DNCNN_AVX512=OFF -DNCNN_AVX512VNNI=OFF -DNCNN_AVX512BF16=OFF \
        -DNCNN_AVX512FP16=OFF -DNCNN_AVXVNNI=OFF -DNCNN_AVXVNNIINT8=OFF \
        -DNCNN_AVXNECONVERT=OFF -DNCNN_FMA4=OFF -DNCNN_XOP=OFF \
        -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
        -DNCNN_BUILD_BENCHMARK=OFF
    cmake --build ncnn/build -j2

O CMake precisa achar o OpenCV, porque o ncnn2table lê as fotos com ele.

    python optimization/quantize_ncnn.py --images get_data/dataset/images

--ncnn-dir aponta para o clone se as tools não estiverem no PATH.
As fotos de calibração são frames da trena, de preferência da mesma câmera.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NANODET_ROOT = ROOT / "third_party" / "nanodet"
DEFAULT_CFG = ROOT / "run" / "trena.yml"
DEFAULT_CKPT = ROOT / "run" / "model_best.ckpt"
DEFAULT_IMAGES = ROOT / "get_data" / "dataset" / "images"
OUT_DIR = Path(__file__).resolve().parent / "out"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta o checkpoint da trena e gera o NCNN int8.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGES,
        help="Pasta com fotos da trena para a calibração int8.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=200,
        help="Quantas fotos entram na tabela. O KL não precisa do dataset inteiro.",
    )
    parser.add_argument(
        "--ncnn-dir",
        type=Path,
        default=None,
        help="Clone do NCNN. Também vale a variável NCNN_DIR.",
    )
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 4)))
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Reusa o ONNX já escrito em optimization/out.",
    )
    return parser.parse_args()


def _load_cfg(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _norm_from_cfg(cfg: dict) -> tuple[list[float], list[float], int, int]:
    """mean em 0–255 e norm = 1/std. Equivale ao Pipeline do NanoDet."""
    mean, std = cfg["data"]["val"]["pipeline"]["normalize"]
    height, width = cfg["data"]["val"]["input_size"]
    norm = [1.0 / float(value) for value in std]
    return [float(value) for value in mean], norm, int(height), int(width)


def _bracket(values: list[float], digits: int = 6) -> str:
    body = ",".join(f"{value:.{digits}f}" for value in values)
    return f"[{body}]"


def _sample_images(folder: Path, limit: int) -> list[Path]:
    if not folder.is_dir():
        raise SystemExit(
            f"Não achei fotos em {folder}. Passe --images com frames da trena."
        )
    found = [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not found:
        raise SystemExit(f"{folder} não tem jpg ou png para calibrar.")
    if len(found) <= limit:
        return found
    step = len(found) / float(limit)
    return [found[int(index * step)] for index in range(limit)]


def _find_tool(name: str, ncnn_dir: Path | None) -> Path:
    folders: list[Path] = []
    if ncnn_dir is not None:
        folders.append(ncnn_dir)
    env = os.environ.get("NCNN_DIR")
    if env:
        folders.append(Path(env))
    for root in folders:
        for candidate in (
            root / "build" / "tools" / name,
            root / "build" / "tools" / "onnx" / name,
            root / "build" / "tools" / "quantize" / name,
            root / "tools" / name,
            root / "tools" / "quantize" / name,
            root / name,
        ):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    raise SystemExit(
        f"Não achei {name}. Compile o NCNN no PC com as tools e o OpenCV, "
        "e passe --ncnn-dir para o clone (ou exporte NCNN_DIR)."
    )


def _stage_images(images: list[Path], folder: Path) -> list[str]:
    """O ncnn2table lê cada linha com sscanf %s e corta em 255 caracteres.

    O caminho deste disco tem espaço, então a lista usa nomes curtos
    relativos à raiz do repositório, sem espaço.
    """
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    listed: list[str] = []
    for index, src in enumerate(images):
        dest = folder / f"{index:04d}{src.suffix.lower()}"
        try:
            dest.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dest)
        listed.append(dest.relative_to(ROOT).as_posix())
    return listed


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def _export_ncnn(cfg_path: Path, checkpoint: Path, param_path: Path, bin_path: Path, height: int, width: int) -> None:
    """O onnx2ncnn deixa o channel_shuffle do ShuffleNet como Shape, e o NCNN não roda isso.

    O PNNX converte o módulo PyTorch direto e vira ShuffleChannel.
    """
    if not NANODET_ROOT.is_dir():
        raise SystemExit(
            f"Não achei o NanoDet em {NANODET_ROOT}. "
            "O clone fica em third_party/nanodet."
        )
    sys.path.insert(0, str(NANODET_ROOT))
    try:
        import pnnx
        import torch
        from nanodet.model.arch import build_model
        from nanodet.util import Logger, cfg, load_config, load_model_weight
    except ImportError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(f"Falta dependência para exportar ({missing}).") from exc

    load_config(cfg, str(cfg_path))
    model = build_model(cfg.model)
    try:
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(checkpoint, map_location="cpu")
    load_model_weight(model, loaded, Logger(-1, str(OUT_DIR), False))
    model.eval()

    dummy = torch.zeros(1, 3, height, width)
    pnnx.export(
        model,
        str(OUT_DIR / "trena.pt"),
        inputs=dummy,
        ncnnparam=str(param_path),
        ncnnbin=str(bin_path),
        fp16=False,
        device="cpu",
    )
    print(f"NCNN fp32 em {param_path}", flush=True)


def _reject_unconverted(param_path: Path) -> None:
    bad = [
        line.split()[1]
        for line in param_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("Shape ")
    ]
    if bad:
        sample = ", ".join(bad[:4])
        raise SystemExit(
            f"{param_path} ainda tem camada Shape ({sample}). "
            "O NCNN não executa esse grafo."
        )


def main() -> None:
    args = _parse_args()
    if not args.config.is_file():
        raise SystemExit(f"Não achei o config em {args.config}")
    if not args.skip_export and not args.checkpoint.is_file():
        raise SystemExit(f"Não achei o checkpoint em {args.checkpoint}")

    cfg = _load_cfg(args.config)
    mean, norm, height, width = _norm_from_cfg(cfg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_param, raw_bin = OUT_DIR / "trena.param", OUT_DIR / "trena.bin"
    opt_param, opt_bin = OUT_DIR / "trena-opt.param", OUT_DIR / "trena-opt.bin"
    table_path = OUT_DIR / "trena.table"
    int8_param = OUT_DIR / "trena-int8.param"
    int8_bin = OUT_DIR / "trena-int8.bin"
    image_list = OUT_DIR / "imagelist.txt"

    if args.skip_export:
        if not raw_param.is_file() or not raw_bin.is_file():
            raise SystemExit(f"--skip-export pediu {raw_param}, e o arquivo não existe.")
    else:
        _export_ncnn(args.config, args.checkpoint, raw_param, raw_bin, height, width)
    _reject_unconverted(raw_param)

    optimize = _find_tool("ncnnoptimize", args.ncnn_dir)
    ncnn2table = _find_tool("ncnn2table", args.ncnn_dir)
    ncnn2int8 = _find_tool("ncnn2int8", args.ncnn_dir)

    _run([str(optimize), str(raw_param), str(raw_bin), str(opt_param), str(opt_bin), "0"])
    _reject_unconverted(opt_param)

    images = _sample_images(args.images, args.max_images)
    listed = _stage_images(images, OUT_DIR / "calib")
    image_list.write_text("".join(f"{line}\n" for line in listed), encoding="utf-8")
    print(f"Calibração com {len(images)} fotos.", flush=True)

    _run(
        [
            str(ncnn2table),
            str(opt_param),
            str(opt_bin),
            str(image_list),
            str(table_path),
            f"mean={_bracket(mean)}",
            f"norm={_bracket(norm)}",
            f"shape={_bracket([float(width), float(height), 3.0], digits=0)}",
            "pixel=BGR",
            f"thread={args.threads}",
            "method=kl",
        ]
    )
    _run(
        [
            str(ncnn2int8),
            str(opt_param),
            str(opt_bin),
            str(int8_param),
            str(int8_bin),
            str(table_path),
        ]
    )

    head = cfg["model"]["arch"]["head"]
    print(
        "\n".join(
            [
                f"Modelo int8: {int8_param}",
                f"Pesos int8:  {int8_bin}",
                "Copie os dois para a Raspberry. A libncnn de lá tem que ser a mesma versão das tools.",
                f"Entrada {width}x{height} BGR  mean={_bracket(mean)}  norm={_bracket(norm)}",
                f"reg_max {head['reg_max']}  strides {head['strides']}  classes {head['num_classes']}",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
