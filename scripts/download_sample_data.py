"""소규모 학습용 샘플 데이터 다운로드.

HuggingFace `datasets` v2.14+ 부터 스크립트 기반 데이터셋이 deprecated 되었으므로,
parquet/arrow 기반의 모던 데이터셋만 사용한다.

사용:
  python scripts/download_sample_data.py --num-samples 5000 --out data/coco_subset

지원 데이터셋(자동 폴백 순서):
  1. lmms-lab/flickr30k          (test split, 31k images, 영문 캡션)
  2. Multimodal-Fatima/COCO_captions_train  (COCO 캡션, 영문)

다른 데이터셋을 쓰려면 --dataset 옵션으로 직접 지정.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


CAPTION_PROMPTS = [
    "Describe this image in detail.",
    "What is in this image?",
    "Provide a caption for this image.",
    "이 이미지를 자세히 설명해 주세요.",
    "사진 속에는 무엇이 있나요?",
]

DATASET_FALLBACKS = [
    ("lmms-lab/flickr30k", "test"),
    ("Multimodal-Fatima/COCO_captions_train", "train"),
]

IMAGE_KEYS = ("image", "img", "pixel_values", "jpg", "photo")
CAPTION_KEYS = (
    "caption",
    "captions",
    "sentences",
    "text",
    "description",
    "sentences_raw",
)


def detect_columns(keys):
    image_col = next((c for c in IMAGE_KEYS if c in keys), None)
    caption_col = next((c for c in CAPTION_KEYS if c in keys), None)
    return image_col, caption_col


def extract_caption(value):
    """캡션 필드가 str / list[str] / list[dict] 등 다양한 형태일 수 있어 일반화."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            for k in ("raw", "caption", "text", "sentence"):
                if k in first and isinstance(first[k], str):
                    return first[k].strip()
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset id (parquet only). 미지정 시 fallback 순회.",
    )
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--out", type=str, default="data/coco_subset")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    candidates = (
        [(args.dataset, args.split or "train")]
        if args.dataset
        else DATASET_FALLBACKS
    )

    ds_iter = None
    chosen = None
    for ds_id, split in candidates:
        print(f"[try] {ds_id} (split={split}, streaming=True)")
        try:
            ds = load_dataset(ds_id, split=split, streaming=True)
            ds_iter = iter(ds)
            chosen = (ds_id, split)
            print(f"[ok] using {ds_id}")
            break
        except Exception as e:
            print(f"[fail] {type(e).__name__}: {str(e)[:200]}")

    if ds_iter is None:
        raise RuntimeError(
            "모든 fallback 데이터셋 로드 실패. "
            "--dataset <hf_id> --split <name> 으로 직접 지정해 보세요."
        )

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    manifest = []
    image_col = caption_col = None

    pbar = tqdm(total=args.num_samples, desc="export")
    for ex in ds_iter:
        if image_col is None:
            image_col, caption_col = detect_columns(list(ex.keys()))
            if image_col is None or caption_col is None:
                raise RuntimeError(
                    f"image/caption 컬럼 자동 감지 실패. keys={list(ex.keys())}"
                )
            print(f"[info] image_col={image_col}, caption_col={caption_col}")

        image = ex.get(image_col)
        caption = extract_caption(ex.get(caption_col))
        if image is None or not caption:
            continue

        try:
            img_path = img_dir / f"{len(manifest):06d}.jpg"
            image.convert("RGB").save(img_path, "JPEG", quality=90)
        except Exception as e:
            print(f"[skip] {e}")
            continue

        manifest.append(
            {
                "image": str(img_path).replace("\\", "/"),
                "question": rng.choice(CAPTION_PROMPTS),
                "answer": caption,
            }
        )
        pbar.update(1)
        if len(manifest) >= args.num_samples:
            break
    pbar.close()

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[done] {len(manifest)} samples (from {chosen[0]}) → {manifest_path}")
    print(f"        images → {img_dir}")


if __name__ == "__main__":
    main()
