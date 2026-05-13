# 클립보드 이미지를 assets/ 폴더에 PNG로 저장하는 헬퍼.
#
# 사용:
#   1. Win+Shift+S 로 화면 캡처 (클립보드에 자동 복사)
#   2. .\scripts\save_clipboard_screenshot.ps1 -Name test_a_q1
#
# 또는 인자 없이 실행 → 13개 파일 순차 저장 모드:
#   .\scripts\save_clipboard_screenshot.ps1

param(
    [string]$Name = "",
    [string]$AssetsDir = "assets"
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

function Save-ClipboardImage {
    param([string]$FilePath)

    $img = [System.Windows.Forms.Clipboard]::GetImage()
    if ($null -eq $img) {
        Write-Host "  [ERROR] 클립보드에 이미지가 없습니다. Win+Shift+S 로 캡처 후 다시 시도하세요." -ForegroundColor Red
        return $false
    }

    $dir = Split-Path -Parent $FilePath
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }

    $img.Save($FilePath, [System.Drawing.Imaging.ImageFormat]::Png)
    $img.Dispose()
    $size = [math]::Round((Get-Item $FilePath).Length / 1KB, 1)
    Write-Host "  [OK] $FilePath ($size KB)" -ForegroundColor Green
    return $true
}

# 단건 저장 모드
if ($Name) {
    $path = Join-Path $AssetsDir ($Name + ".png")
    Save-ClipboardImage -FilePath $path | Out-Null
    return
}

# 일괄 저장 모드 — 13개 파일 순차 안내
$files = @(
    @{Name="v1_dog_response"; Desc="v1 강아지 응답 ('frisbee on the beach' 환각)"},
    @{Name="test_a_q1"; Desc="Test A1: 'What is in this image?' -> 'Dog.'"},
    @{Name="test_a_q2"; Desc="Test A2: 'What color is the dog?' -> 'White.'"},
    @{Name="test_a_q3"; Desc="Test A3: 'Is the dog wearing anything on its head?' -> 'Yes.'"},
    @{Name="test_a_q4"; Desc="Test A4: 'What is on the dog's head?' -> 'Hat.'"},
    @{Name="test_a_q5"; Desc="Test A5: 'Describe this image in one sentence.' -> '...cat on the floor.'"},
    @{Name="test_b_q1"; Desc="Test B1: '이 이미지에 무엇이 보이나요?' -> '화이트의 소파, 물건.'"},
    @{Name="test_b_q2"; Desc="Test B2: '이 강아지는 무슨 색이에요?' -> '보통.'"},
    @{Name="test_b_q3"; Desc="Test B3: '이 강아지는 머리에 무엇을 쓰고 있나요?' -> '개.' (smoking gun)"},
    @{Name="test_b_q4"; Desc="Test B4: '이 이미지를 한 문장으로...' -> 'In this picture I can see a cat...'"},
    @{Name="test_c_q1"; Desc="Test C1: 'What is in this image?' -> 'Giraffe.'"},
    @{Name="test_c_q2"; Desc="Test C2: 'What color is the main character?' -> 'White.'"},
    @{Name="test_c_q3"; Desc="Test C3: 'What is the character wearing on its head?' -> 'Tie.'"},
    @{Name="test_c_q4"; Desc="Test C4: 'Describe this image in one sentence.' -> '...human figure...'"}
)

Write-Host ""
Write-Host "=== 일괄 저장 모드 ($($files.Count)개) ===" -ForegroundColor Cyan
Write-Host "각 파일마다:"
Write-Host "  1. 안내 메시지 확인"
Write-Host "  2. Win+Shift+S 로 해당 응답 캡처"
Write-Host "  3. 캡처 완료되면 Enter 입력 -> 자동 저장"
Write-Host "  4. 건너뛰려면 's' 입력"
Write-Host "  5. 중단하려면 'q' 입력"
Write-Host ""

foreach ($f in $files) {
    $path = Join-Path $AssetsDir ($f.Name + ".png")
    if (Test-Path $path) {
        Write-Host "[skip] $($f.Name).png 이미 존재" -ForegroundColor Yellow
        continue
    }

    Write-Host ""
    Write-Host "[다음] $($f.Name).png" -ForegroundColor Cyan
    Write-Host "  $($f.Desc)"
    $input = Read-Host "  캡처 후 Enter (s=skip, q=quit)"
    if ($input -eq 'q') { break }
    if ($input -eq 's') { continue }

    Save-ClipboardImage -FilePath $path | Out-Null
}

Write-Host ""
Write-Host "=== 완료 ===" -ForegroundColor Cyan
$saved = (Get-ChildItem -Path $AssetsDir -Filter *.png -ErrorAction SilentlyContinue).Count
Write-Host "현재 저장된 파일 수: $saved / 14"
