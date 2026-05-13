"""순수 LoRA 가중치 추출 — 1GB → ~10MB.

⚠️  **경고: 이 가설은 실험으로 반증되었습니다.**
    실제 슬림 adapter (8.68MB) 로 동일 입력 추론 시 Test A 5문항 중 3문항이
    명확히 다른 응답으로 변함 (Dog→Cat, White→Brown and black, Hat→Mittens).
    → embed_tokens / lm_head 는 단순 저장이 아니라 학습된 상태의 일부였음.
    상세는 README §Step 5 참조: https://github.com/AD-Styles/vlm-from-scratch#step-5--배포-용이성-시도-실패에서-배운-것

    이 스크립트는 **반증된 실험의 기록** 으로만 보존됨. 실제 배포는 원본 1GB 그대로
    Hugging Face Hub 에 업로드하는 방식 (scripts/upload_to_hf.py) 을 사용하십시오.

──────────────────────────────────────────────────────────

원래 가설 (반증 전):
  v2 학습 후 lora_adapter/adapter_model.safetensors 가 약 1GB. PEFT 가
  embedding resize (`<image>` 토큰 추가) 를 감지하고 embed_tokens / lm_head 를
  자동으로 함께 저장했기 때문 (각 ~540MB). 추론 시 MiniLLaVA.__init__() 가
  resize_token_embeddings 를 호출하므로 저장된 값은 무관할 것이라 추정.

  → 가설: embed_tokens / lm_head 를 제거해도 inference 무영향
  → **반증**: 응답 품질 명확히 손실 (위 경고 참조)

추정 원인 (v3 검증 과제):
  - Qwen2.5 의 tie_word_embeddings=True → LoRA gradient 가 lm_head 를 거쳐
    embed_tokens 까지 미세 영향
  - 또는 PEFT 가 resize 감지 시 silent unfreeze

이 스크립트의 동작 (그대로 유지):
  1. 원본 adapter_model.safetensors 에서 LoRA 키만 추출
  2. embed_tokens, lm_head 제외
  3. adapter_config.json 의 modules_to_save 항목 제거
  4. 결과: ~10MB slim adapter (작동은 하나 품질 손실)

사용 (실험 재현 / 검증 목적만):
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
    print("⚠️  [경고] 이 슬림 adapter 는 실험으로 품질 손실 확인됨 (Test A 5문항 중 3문항 응답 변경).")
    print("           실제 배포는 원본 1GB 그대로 Hugging Face Hub 에 업로드하는 방식 권장.")
    print("           상세: README §Step 5 (https://github.com/AD-Styles/vlm-from-scratch#-회고--개선의-여정)")


if __name__ == "__main__":
    main()
