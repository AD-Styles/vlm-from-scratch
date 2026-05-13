"""Hugging Face Hub 에 학습된 v2 가중치 업로드.

사전 준비 (1회만):
  1. https://huggingface.co/settings/tokens 에서 token 생성 (write 권한)
  2. PowerShell:  huggingface-cli login
                  → 토큰 붙여넣기

사용:
  python scripts/upload_to_hf.py --repo-id AD-Styles/mini-llava-stage2

옵션:
  --private              비공개 레포로 생성
  --token <hf_xxx>       cached login 대신 토큰 직접 지정
  --folder <path>        업로드할 폴더 (기본: checkpoints/v2_stage2_lora)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, login


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace 레포 ID (예: AD-Styles/mini-llava-stage2)",
    )
    p.add_argument("--folder", default="checkpoints/v2_stage2_lora")
    p.add_argument("--token", default=None)
    p.add_argument("--private", action="store_true")
    p.add_argument(
        "--card",
        default="scripts/hf_model_card.md",
        help="HF 레포 README 로 업로드할 모델 카드 마크다운",
    )
    return p.parse_args()


def main():
    args = parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"업로드 폴더 없음: {folder}")

    card = Path(args.card)
    if not card.exists():
        print(f"[warn] 모델 카드 없음: {card} — README 없이 업로드")
        card = None

    # 토큰 우선순위: --token 인자 > HF_TOKEN 환경변수 > 캐시된 login
    raw_token = args.token or os.environ.get("HF_TOKEN", "")
    token = raw_token.strip() if raw_token else None  # 앞뒤 공백/newline 제거

    if token:
        source = "--token 인자" if args.token else "HF_TOKEN 환경변수"
        print(f"[init] 토큰 사용처: {source} (길이={len(token)}, prefix={token[:3]!r})")
        if not token.startswith("hf_"):
            print(f"[warn] 토큰이 'hf_' 로 시작하지 않음 — 형식 오류일 수 있음")
        api = HfApi(token=token)
    else:
        print("[init] 토큰 없음 — 캐시된 로그인 시도")
        api = HfApi()

    # 1) 레포 생성 또는 확인
    print(f"[1/3] 레포 생성/확인: {args.repo_id} (private={args.private})")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    # 2) 모델 카드를 README.md 로 업로드
    if card:
        print(f"[2/3] 모델 카드 업로드: {card} → README.md")
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            token=token,
        )

    # 3) 가중치 업로드 (불필요 파일 제외)
    print(f"[3/3] 가중치 업로드: {folder} → repo root  (1GB+, 수 분 소요)")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="model",
        token=token,
        ignore_patterns=[
            "*.md",  # PEFT 자동 생성 README 제외 (위에서 우리 카드 사용)
            "lora_adapter_slim/*",  # 실패한 슬림 버전 제외
            "projector_step*.pt",  # 중간 체크포인트 제외
        ],
    )

    url = f"https://huggingface.co/{args.repo_id}"
    print(f"\n[완료] {url}")
    print(f"\n사용자 다운로드 명령:")
    print(f"  huggingface-cli download {args.repo_id} --local-dir checkpoints/v2_stage2_lora")


if __name__ == "__main__":
    main()
