"""순수 LoRA 가중치 추출 — 1GB → ~10MB.

문제 상황:
  v2 학습 후 lora_adapter/adapter_model.safetensors 가 약 1GB. 이는 PEFT 가
  embedding resize (`<image>` 토큰 추가) 를 감지하고 embed_tokens / lm_head 를
  자동으로 함께 저장했기 때문 (각 ~540MB).

추론 시 그것들이 정말 필요한가?
  - MiniLLaVA.__init__() 가 항상 tokenizer.add_special_tokens 후 resize_token_embeddings 호출
    → embed_tokens / lm_head 는 inference time에 매번 새로 resize 됨
  - 새 <image> 토큰의 embedding 은 forward 시 image patch features 로 교체됨 (`_merge`)
    → 새 토큰 embedding 의 random init 값은 절대 사용되지 않음
  → 결론: embed_tokens / lm_head 를 저장할 필요 없음

이 스크립트는:
  1. 원본 adapter_model.safetensors 에서 LoRA 키만 추출
  2. embed_tokens, lm_head 는 제외
  3. adapter_config.json 의 modules_to_save 항목 제거
  4. 결과: ~10MB slim adapter (GitHub 100MB 제한 통과)

사용:
  python scripts/extract_lora.py \\
    --input-dir checkpoints/v2_stage2_lora/lora_adapter \\
    --output-dir checkpoints/v2_stage2_lora/lora_adapter_slim
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


# 제외할 모듈 식별 키워드 (이게 키 이름에 포함되면 drop)
DROP_PATTERNS = ("embed_tokens", "lm_head")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="원본 adapter 디렉터리 (e.g., checkpoints/v2_stage2_lora/lora_adapter)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="슬림 adapter 저장 위치 (e.g., checkpoints/v2_stage2_lora/lora_adapter_slim)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 저장 없이 어떤 키가 제외되는지 출력만",
    )
    return p.parse_args()


def fmt_mb(num_bytes: int) -> str:
    return f"{num_bytes / 1e6:.2f} MB"


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    src_safetensors = input_dir / "adapter_model.safetensors"
    src_config = input_dir / "adapter_config.json"

    if not src_safetensors.exists():
        raise FileNotFoundError(f"원본 safetensors 없음: {src_safetensors}")
    if not src_config.exists():
        raise FileNotFoundError(f"adapter_config.json 없음: {src_config}")

    # ── 1. Load original safetensors
    print(f"[load] {src_safetensors}")
    state = load_file(str(src_safetensors))

    total_bytes = sum(v.numel() * v.element_size() for v in state.values())
    print(f"[stat] 원본: {len(state)}개 키, {fmt_mb(total_bytes)}")

    # ── 2. Filter
    keep, drop = {}, {}
    for k, v in state.items():
        if any(pat in k for pat in DROP_PATTERNS):
            drop[k] = v
        else:
            keep[k] = v

    keep_bytes = sum(v.numel() * v.element_size() for v in keep.values())
    drop_bytes = sum(v.numel() * v.element_size() for v in drop.values())

    print(f"[keep] LoRA 키: {len(keep)}개, {fmt_mb(keep_bytes)}")
    print(f"[drop] 제외 키: {len(drop)}개, {fmt_mb(drop_bytes)}")
    if drop:
        print("       제외 대상 (샘플):")
        for k in list(drop.keys())[:5]:
            print(f"         - {k}")
        if len(drop) > 5:
            print(f"         ... 외 {len(drop) - 5}개")

    # ── 3. Modify adapter_config.json
    with open(src_config, "r", encoding="utf-8") as f:
        config = json.load(f)
    original_mts = config.get("modules_to_save")
    if original_mts:
        print(f"[config] modules_to_save 제거: {original_mts}")
        config["modules_to_save"] = None

    if args.dry_run:
        print("\n[dry-run] 실제 저장 없이 종료. --dry-run 빼고 다시 실행하세요.")
        return

    # ── 4. Save slim files
    output_dir.mkdir(parents=True, exist_ok=True)

    dst_safetensors = output_dir / "adapter_model.safetensors"
    save_file(keep, str(dst_safetensors))
    print(f"[save] {dst_safetensors}  ({fmt_mb(dst_safetensors.stat().st_size)})")

    dst_config = output_dir / "adapter_config.json"
    with open(dst_config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[save] {dst_config}")

    # README.md 가 원본에 있으면 복사 (PEFT 가 자동 생성)
    src_readme = input_dir / "README.md"
    if src_readme.exists():
        dst_readme = output_dir / "README.md"
        shutil.copy(src_readme, dst_readme)

    print(f"\n[done] 슬림 adapter 추출 완료 → {output_dir}")
    print(
        f"        원본 {fmt_mb(total_bytes)} → 슬림 {fmt_mb(dst_safetensors.stat().st_size)} "
        f"(축소율 {(1 - dst_safetensors.stat().st_size / total_bytes) * 100:.1f}%)"
    )
    print()
    print("[사용법] 데모 실행 시:")
    print(f"  python app.py --checkpoint checkpoints/v2_stage2_lora/projector.pt \\")
    print(f"                --lora-adapter {output_dir}")
    print()
    print("[주의] embed_tokens / lm_head 는 제외됨.")
    print("       MiniLLaVA.__init__() 가 inference 시 매번 resize_token_embeddings 호출하므로 정상 동작.")
    print("       <image> 토큰 embedding 은 어차피 forward 에서 image patch 로 교체되므로 영향 없음.")


if __name__ == "__main__":
    main()
