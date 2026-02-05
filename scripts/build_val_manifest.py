import argparse
import json
from pathlib import Path


def _collect_stems(root: Path) -> list:
    if not root.exists():
        return []
    return sorted({p.stem for p in root.rglob("*.txt")})


def main():
    parser = argparse.ArgumentParser(
        description="Build a JSON val manifest from a YOLO labels directory."
    )
    parser.add_argument(
        "--labels_root",
        required=True,
        help="Path to YOLO labels root (contains train/val subdirs).",
    )
    parser.add_argument(
        "--val_dir",
        default="val",
        help="Val subdirectory name under labels_root (default: val).",
    )
    parser.add_argument(
        "--train_dir",
        default="train",
        help="Train subdirectory name under labels_root (default: train).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for val manifest.",
    )
    parser.add_argument(
        "--check_train",
        action="store_true",
        help="Check for overlap with train split and report.",
    )

    args = parser.parse_args()

    labels_root = Path(args.labels_root)
    val_root = labels_root / args.val_dir

    if not val_root.exists():
        raise FileNotFoundError(f"val dir not found: {val_root}")

    val_stems = _collect_stems(val_root)
    if not val_stems:
        print(f"Warning: no val labels found under {val_root}")

    if args.check_train:
        train_root = labels_root / args.train_dir
        train_stems = set(_collect_stems(train_root))
        overlap = set(val_stems) & train_stems
        if overlap:
            sample = list(sorted(overlap))[:10]
            print(
                f"Warning: {len(overlap)} labels appear in both train and val. "
                f"Example: {', '.join(sample)}"
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"val": val_stems}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {len(val_stems)} val entries to {out_path}")


if __name__ == "__main__":
    main()
