# GitHub Pages: OnSense (오늘의 급식 안내)

이 디렉토리는 GitHub Pages로 배포되는 정적 사이트입니다.

- 사이트 루트: `docs/`
- 엔트리: `index.html`
- 오디오 파일: `ONsensE-prototype.mp3`

## 배포(설정) 방법
1. GitHub 저장소의 Settings → Pages 이동
2. Build and deployment 섹션에서 Source를 `Deploy from a branch`로 선택
3. `Branch: main`과 `/docs` 선택 후 Save
4. 잠시 후 `https://<owner>.github.io/<repo>/` 또는 `https://<owner>.github.io/<repo>/index.html` 로 접속

## 로컬 미리보기
GitHub Pages는 정적 호스팅이므로, 로컬에서는 간단한 HTTP 서버로 확인할 수 있습니다.

```bash
python -m http.server -d docs 8000
# 브라우저에서 http://localhost:8000
```
