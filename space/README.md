---
title: Mini-LLaVA Demo (v2)
emoji: 🖼️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.14.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: VLM from scratch (CLIP + Qwen2.5 + LoRA) — v2 demo
---

# Mini-LLaVA — v2 Demo

CLIP-ViT-B/32 + MLP Projector + Qwen2.5-0.5B (+ LoRA r=16) 로 직접 구현한 Vision-Language Model 의 데모 배포.
HuggingFace 의 `LlavaForConditionalGeneration` 같은 고수준 추상화 미사용 — `<image>` 토큰 splice 와 융합 로직 직접 구현.

## ⚠️ 데모 버전의 한계 (정직한 명시)

| 항목 | 상태 |
|------|------|
| 영문 단답 VQA (Dog / White / Yes / Hat) | ✅ 정확 |
| 영문 짧은 묘사 | ✅ 작동 |
| 한국어 응답 | ⚠️ catastrophic forgetting (학습 데이터 100% 영어) |
| OOD (만화/애니 캐릭터) | ⚠️ 환각 (학습 분포 외부) |

**v3 production 버전 개발 중**: 한국어 instruction 데이터 + CLIP-ViT-L/14 + OOD detection.

## 🔗 링크

- 📂 [Code (GitHub)](https://github.com/AD-Styles/vlm-from-scratch)
- 🤗 [Weights (HF Hub)](https://huggingface.co/AD-Styles/mini-llava-stage2)
- 📊 [Test A/B/C 결과 표](https://github.com/AD-Styles/vlm-from-scratch#-results)
- 📖 [v1→v2 회고록 (시행착오 분석)](https://github.com/AD-Styles/vlm-from-scratch#-회고--개선의-여정)

## 💡 사용 팁

- **CPU 환경** 이라 응답에 5-15초 소요됩니다 (첫 응답은 모델 로드 추가).
- 영문 질문 권장 (한국어는 의도적으로 약함).
- 예시 질문 클릭 → 이미지 업로드 → "응답 생성".

---
🤖 김도윤 (AD-Styles) · 2026 · MIT License
