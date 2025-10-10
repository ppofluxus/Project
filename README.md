# Project

이 저장소는 Python 프로젝트용 템플릿입니다.

## 구조
- Python 소스 코드
- `.gitignore`: Python/Windows/IDE 공통 제외 규칙
- `.gitattributes`: 운영체제별 줄바꿈 충돌 방지

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

## 라이선스
MIT (필요 시 수정)
