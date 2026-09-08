# TUG GUARD · 해양 안전 디지털 트윈

항만에서 본선과 ASD 예인선의 관계, 예인줄 장력, 거팅·후류·흡인 위험을 탐색하는 인터랙티브 시뮬레이터입니다.

- [공개 사이트](https://tugboat-safety-twin.vercel.app)
- 기술: React 18 · TypeScript · Vite 5 · Three.js · React Three Fiber · Tailwind CSS

## 화면과 기능

- 넓은 해양 장면, 접을 수 있는 관제 패널, 네 가지 시나리오를 고르는 하단 조작부
- 시간대별 바다·조명, 수면에 합성한 항적과 포말, 부두·크레인 배경
- 개별 적층 컨테이너, 골판 재질, 선체를 따르는 수선 띠, 창문·난간·예인선 방충재
- 자연 장면/위험 분석 레이어, 기존 다섯 시점 + 선미 덱/조타실 CCTV, 고품질/기본 화질
- 운항 개요·레이더·추이·센서 탭과 세부 파라미터
- 상시 접근 가능한 비상 예인줄 분리/재연결
- 실제 시뮬레이션 관측값을 기록하는 시나리오 점검창; 완료/취소 시 원래 설정 복원
- Normal Map 기반 수면, 맑음~해무 무작위화, 45RPM 이상 백색 후류(115RPM에서 방출량·크기 2배)
- 예인줄 데이터 생성: RGB PNG + 이진 마스크 + 기하학적 정답 JSON, 로컬 배치 저장

## Towline Vision 합성 데이터 생성

우측 **Dataset Generation · 개발자** 모드에서 RGB/마스크 미리보기, 단일 샘플 ZIP, 폴더 단위 파일 생성을 사용할 수 있습니다. 기본 500장, 1280×720 PNG + 예인줄 이진 마스크 + 기하학적 정답 JSON이며, 시나리오 단위 70/20/10 분할과 5개 Sag 생성 구간을 사용합니다. 대량 생성은 로컬 스트리밍 명령을 권장합니다.

```sh
npm ci
npm run dataset:generate -- --count 10 --out towline_dataset_smoke
npm run dataset:generate
npm run dataset:validate -- towline_dataset --count 500
```

Chrome이 필요하며, Python 검증기는 Pillow를 사용합니다. 상세 설치·미리보기·생성·스키마·한계는 **[README_DATASET.md](README_DATASET.md)**를 읽으세요. Sag는 기하학적 특징이며 장력·위험 등급이 아닙니다. 이전 JPEG/YOLO 내보내기 설명은 과거 설계 문서에 남아 있고, 현재 개발자 UI는 이 예인줄 데이터셋을 생성합니다.

## 실행

```sh
npm install
npm run dev
npm run test:physics
npm test
npm run build
npm run preview
```

개발 서버 기본 포트는 5173입니다. 별도 API 키나 외부 모델 다운로드 없이 실행됩니다.

## 조작

| 조작 | 기능 |
|---|---|
| 마우스 드래그 / 휠 | 자유 시점 회전 / 확대 |
| 1–5 | 자유 / 추적 / 선교 / 상공 / 시네마틱 |
| T | 주간 → 황혼 → 야간 |
| M | 음향 켜기/끄기 (초기 음소거) |
| F | 관제 패널 접기/열기 |
| Space | 예인줄 분리/재연결 |

입력 필드·버튼·점검창을 조작할 때는 전역 단축키가 개입하지 않습니다. 모바일에서는 상하 스크롤로 관제 패널을 이용합니다.

## 데이터와 검증 범위

이 프로젝트는 **시나리오 시뮬레이터**이며 실제 센서, 카메라, AI 추론, 선박 제어와 연결되어 있지 않습니다. 선박 명칭과 항만은 시각화를 위한 가상 설정입니다. 물리식과 위험 임계값은 교육·데모 목적이며 실제 운항 판단이나 인증을 대체하지 않습니다.

`npm test`는 실제 공용 물리 엔진, 투영·곡률·거품·아카이브·카메라를 검증합니다. 기존 `test:physics`는 계산식 복사본 검사로 유지합니다. 선체 근접 프리셋은 현재 엔진에서 약 12~13m 이격으로 나타나며 반드시 CRITICAL을 발생시키는 프리셋은 아닙니다.

고품질은 그림자와 높은 수면 분할을 사용하며 픽셀 비율을 최대 1.5로 제한합니다. 성능이 부족하면 기본 화질을 선택하세요. 실시간 선박 반사·완전한 부력 해석·실사 모델은 현재 구현 범위에 포함하지 않습니다.

## 문서

- [디자인 검토와 예상 문제](docs/2026-09-07-design-review.md)
- [구현 계획](docs/2026-09-07-implementation-plan.md)
- [트러블슈팅과 검증 기록](docs/2026-09-07-troubleshooting.md)
- [합성 데이터 설계](docs/2026-09-07-synthetic-dataset-plan.md)
- [합성 데이터 검증과 트러블슈팅](docs/2026-09-07-synthetic-dataset-validation.md)

.env 및 API 키 등 민감정보는 저장소에 추가하거나 푸시하지 않습니다.

## 라이선스

MIT License.
