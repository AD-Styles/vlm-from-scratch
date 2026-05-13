# Assets — 스크린샷 파일명 매핑

README.md 가 다음 13개 파일을 참조합니다. 동일한 이름으로 저장하면 자동 표시됩니다.

## 필요한 파일 (13개)

### v1 baseline (1개)
- `v1_dog_response.png` — 강아지 사진 v1 응답 ("frisbee on the beach" 환각)

### Test A — 영문 VQA (5개)
- `test_a_q1.png` — "What is in this image?" → "Dog."
- `test_a_q2.png` — "What color is the dog?" → "White."
- `test_a_q3.png` — "Is the dog wearing anything on its head?" → "Yes."
- `test_a_q4.png` — "What is on the dog's head?" → "Hat."
- `test_a_q5.png` — "Describe this image in one sentence." → "...cat on the floor."

### Test B — 한국어 (4개)
- `test_b_q1.png` — "이 이미지에 무엇이 보이나요?" → "화이트의 소파, 물건."
- `test_b_q2.png` — "이 강아지는 무슨 색이에요?" → "보통."
- `test_b_q3.png` — "이 강아지는 머리에 무엇을 쓰고 있나요?" → "개." ★ 분석의 결정적 증거
- `test_b_q4.png` — "이 이미지를 한 문장으로 설명해 주세요." → "In this picture I can see a cat..."

### Test C — 피카츄 OOD (4개)
- `test_c_q1.png` — "What is in this image?" → "Giraffe."
- `test_c_q2.png` — "What color is the main character?" → "White."
- `test_c_q3.png` — "What is the character wearing on its head?" → "Tie."
- `test_c_q4.png` — "Describe this image in one sentence." → "...human figure..."

## 빠르게 저장하는 방법 (PowerShell 헬퍼)

`scripts/save_clipboard_screenshot.ps1` 을 실행하면, **클립보드의 이미지를 지정한 파일명으로 저장**합니다.

```powershell
# 1. Win+Shift+S 로 화면 캡처 (클립보드에 자동 복사)
# 2. PowerShell에서:
.\scripts\save_clipboard_screenshot.ps1 -Name test_a_q1
```

또는 인자 없이 실행하면 모든 13개 파일을 순서대로 저장하도록 안내합니다:
```powershell
.\scripts\save_clipboard_screenshot.ps1
```

---

저장 완료 후 README.md 의 이미지가 자동으로 표시됩니다.
