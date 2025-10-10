# Project

이 저장소는 Python 프로젝트용 템플릿입니다.

## 구조
- Python 소스 코드
- `.gitignore`: Python/Windows/IDE 공통 제외 규칙
- `.gitattributes`: 운영체제별 줄바꿈 충돌 방지
- `examples/`: 플랫폼 독립적인 예제 스크립트 모음 (데이터/결과물은 추적하지 않음)

## 개발
로컬 가상환경 사용을 권장합니다.

```
python -m venv .venv
```

macOS/Linux:
```
source .venv/bin/activate
pip install -U pip
```

Windows (PowerShell):
```
.\\.venv\\Scripts\\Activate.ps1
pip install -U pip
```

## 대용량 산출물 관리
- 학습 로그, 체크포인트, 데이터셋은 `tests/`, `examples/` 내부의 `runs/`, `results/` 등 로컬 전용 디렉터리에 저장하세요.
- `.gitignore`가 위 디렉터리와 이미지/CSV/모델 파일을 자동으로 무시하여 커밋 수가 과도하게 늘어나는 것을 방지합니다.
- 버전 관리가 필요한 산출물은 S3, DVC, 혹은 별도 아티팩트 저장소를 사용하세요.

## 라이선스
MIT (필요 시 수정)
