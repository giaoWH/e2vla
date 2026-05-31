#!/usr/bin/env python3
import argparse
import glob
import importlib
import json
import os
import platform
import sys
from pathlib import Path


REQUIRED_IMPORTS = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("h5py", "h5py"),
    ("cv2", "opencv-python/opencv-python-headless"),
    ("einops", "einops"),
    ("pypose", "pypose"),
    ("tyro", "tyro"),
    ("tensorboard", "tensorboard"),
    ("transformers", "transformers"),
    ("diffusers", "diffusers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
]

OPTIONAL_IMPORTS = [
    ("torchcodec", "torchcodec", "Only needed when reading mp4-backed Droid data."),
    ("Pyro4", "Pyro4", "Needed for remote_service, not for train.py itself."),
    ("psutil", "psutil", "Needed for remote_service, not for train.py itself."),
]

PROJECT_IMPORTS = [
    "configs",
    "train",
    "data_utils.dataset_base",
    "data_utils.datasets",
    "models.vla",
    "models.vlm",
    "models.action_expert",
    "models.encoders.dino",
    "models.encoders.siglip",
]


def status(ok, msg):
    mark = "OK" if ok else "FAIL"
    print(f"[{mark}] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


def get_version(module):
    return getattr(module, "__version__", "unknown")


def check_imports(required=True):
    items = REQUIRED_IMPORTS if required else OPTIONAL_IMPORTS
    failures = []

    for item in items:
        if required:
            module_name, pip_name = item
            note = None
        else:
            module_name, pip_name, note = item

        try:
            module = importlib.import_module(module_name)
            suffix = f" version={get_version(module)}"
            if note:
                suffix += f" ({note})"
            status(True, f"import {module_name}{suffix}")
        except Exception as exc:
            if required:
                status(False, f"import {module_name} failed: {exc}")
                failures.append((module_name, pip_name, str(exc)))
            else:
                warn(f"optional import {module_name} failed: {exc}. {note}")

    return failures


def check_project_imports():
    failures = []
    for module_name in PROJECT_IMPORTS:
        try:
            importlib.import_module(module_name)
            status(True, f"project import {module_name}")
        except Exception as exc:
            status(False, f"project import {module_name} failed: {exc}")
            failures.append((module_name, str(exc)))
    return failures


def check_cuda():
    try:
        import torch
    except Exception as exc:
        status(False, f"torch import failed, skip CUDA check: {exc}")
        return False

    print(f"[INFO] torch version: {torch.__version__}")
    print(f"[INFO] torch CUDA build: {torch.version.cuda}")

    cuda_ok = torch.cuda.is_available()
    status(cuda_ok, f"torch.cuda.is_available() = {cuda_ok}")
    if not cuda_ok:
        warn("train.py hardcodes model_device='cuda:0', so CUDA must be available.")
        return False

    count = torch.cuda.device_count()
    print(f"[INFO] CUDA device count: {count}")
    for i in range(count):
        prop = torch.cuda.get_device_properties(i)
        mem_gb = prop.total_memory / (1024 ** 3)
        print(f"[INFO] cuda:{i}: {prop.name}, total_memory={mem_gb:.1f} GB")
    return True


def check_checkpoint(path, load_ckpt=False):
    ckpt = Path(path)
    ok = ckpt.is_file()
    status(ok, f"checkpoint exists: {ckpt}")
    if not ok:
        return False

    json_files = sorted(ckpt.parent.glob("*.json"))
    status(bool(json_files), f"json config in checkpoint dir: {ckpt.parent}")
    for p in json_files:
        print(f"[INFO] found config: {p}")
        try:
            with p.open("r", encoding="utf-8") as fp:
                cfg = json.load(fp)
            print(f"[INFO] config model={cfg.get('model')}, datasets={cfg.get('dataset_classes')}")
        except Exception as exc:
            warn(f"failed to parse {p}: {exc}")

    if load_ckpt:
        try:
            import torch

            print("[INFO] loading checkpoint on CPU for a stronger check...")
            data = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            keys = sorted(data.keys()) if isinstance(data, dict) else []
            status(True, f"torch.load checkpoint OK, top-level keys={keys}")
        except Exception as exc:
            status(False, f"torch.load checkpoint failed: {exc}")
            return False

    return True


def check_data_root(path):
    root = Path(path)
    status(root.exists(), f"data root exists: {root}")
    if not root.exists():
        return False

    files = glob.glob(str(root / "**" / "*.h5"), recursive=True)
    files += glob.glob(str(root / "**" / "*.hdf5"), recursive=True)
    files = sorted(set(files))
    status(bool(files), f"found h5/hdf5 files under data root, count={len(files)}")
    if not files:
        return False

    first = files[0]
    print(f"[INFO] first h5 file: {first}")
    try:
        import h5py

        with h5py.File(first, "r") as f:
            attrs = list(f.attrs.keys())
            print(f"[INFO] attrs: {attrs}")

            keys = []

            def visit(name, obj):
                if hasattr(obj, "shape"):
                    keys.append((name, obj.shape, str(obj.dtype)))
                else:
                    keys.append((name, "group", ""))

            f.visititems(visit)
            print("[INFO] first file items:")
            for name, shape, dtype in keys[:80]:
                print(f"  - {name}: {shape} {dtype}")
            if len(keys) > 80:
                print(f"  ... {len(keys) - 80} more items omitted")
    except Exception as exc:
        status(False, f"open first h5 failed: {exc}")
        return False

    return True


def check_config_name(name):
    try:
        from configs import CONFIGS

        ok = name in CONFIGS
        status(ok, f"training config registered: {name}")
        if ok:
            print(f"[INFO] config object: {CONFIGS[name]}")
        else:
            warn(f"available configs: {sorted(CONFIGS.keys())}")
        return ok
    except Exception as exc:
        status(False, f"checking configs failed: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/nas_data_new/zzj/data_ssd/self_aloha/data_hdf5/pick_place_1031",
    )
    parser.add_argument(
        "--ckpt",
        default="/home/wh/e2vla/checkpoints/0927_e2vla_base_pretrain/ckpt_0600000.pt",
    )
    parser.add_argument(
        "--config",
        default="finetune_self_aloha_pick_place_1031",
    )
    parser.add_argument(
        "--load-ckpt",
        action="store_true",
        help="Actually torch.load the checkpoint on CPU. Slower but stronger.",
    )
    args = parser.parse_args()

    print("[INFO] Python:", sys.version.replace("\n", " "))
    print("[INFO] Executable:", sys.executable)
    print("[INFO] Platform:", platform.platform())
    print("[INFO] CWD:", os.getcwd())

    if not Path("train.py").is_file():
        warn("train.py not found in current directory. Run this script from /home/wh/e2vla.")

    print("\n== Required Python packages ==")
    import_failures = check_imports(required=True)

    print("\n== Optional packages ==")
    check_imports(required=False)

    print("\n== Project imports ==")
    project_failures = check_project_imports()

    print("\n== CUDA ==")
    cuda_ok = check_cuda()

    print("\n== Checkpoint ==")
    ckpt_ok = check_checkpoint(args.ckpt, load_ckpt=args.load_ckpt)

    print("\n== Data root ==")
    data_ok = check_data_root(args.data_root)

    print("\n== Training config ==")
    config_ok = check_config_name(args.config)

    print("\n== Summary ==")
    all_required_ok = (
        not import_failures
        and not project_failures
        and cuda_ok
        and ckpt_ok
        and data_ok
        and config_ok
    )
    status(all_required_ok, "training dependency check")

    if import_failures:
        print("\nMissing required packages. Suggested pip install names:")
        for module_name, pip_name, _ in import_failures:
            print(f"  - {pip_name}  # import name: {module_name}")

    if project_failures:
        print("\nProject import failures:")
        for module_name, exc in project_failures:
            print(f"  - {module_name}: {exc}")

    if not all_required_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
