
# Backtester (통합 백테스트/전략 엔진)

이 프로젝트는 **정적(Static) 엔진 + 동적(Dynamic) 엔진**을 하나의 백테스터로 통합한 계층형 자산배분 시스템입니다.
Master가 두 엔진의 비중을 주기적으로 맞추고(예: 4:6), 각 엔진은 자신의 룰/주기대로 종목을 리밸런싱합니다.

핵심 실행 진입점은 1개입니다.
- `python -m research.backtester.cli --config backtester`

---

## 1) 폴더 구조 (핵심)

- `research/backtester/`
	- 백테스터 실행/오케스트레이션(마스터/엔진/리포팅)
- `core/`
	- 동적 엔진 플러그인 로직(상관필터/모멘텀/셀렉터/할당/스톱로스) 및 공용 백테스트 유틸
- `infra/`
	- 설정 로더, 데이터 다운로드/캐시/전처리
- `config/`
	- 백테스터 설정 및 로직 프로파일

---

## 2) 실행 방법

### 2-1. 의존성 설치

```bash
pip install -r requirements.txt
```
### 2-2. 데이터 수집

기본 실행(설정: `config/data.yaml`):

```bash
python -m infra.data_manager.downloader --config data
```


### 2-3. 백테스트 실행

기본 실행(설정: `config/backtester.yaml`):

```bash
python -m research.backtester.cli --config backtester
```

기간 오버라이드:

```bash
python -m research.backtester.cli --config backtester --start 2015-01-01 --end 2024-12-31
```

실행 결과는 `config/backtester.yaml`의 `RESULTS.root` 아래에 타임스탬프 폴더로 저장됩니다.

---

## 3) 결과물(Outputs)

백테스트 실행 시 결과 폴더(예: `results/backtester_YYYYMMDD_HHMMSS/`)에 아래가 생성됩니다.

- `equity_curve.png`: 누적수익(전략 vs 벤치마크)
- `drawdown.png`: 드로우다운(전략 vs 벤치마크)
- `summary.md`: 핵심 성과 요약 및 시즌별/엔진별 브레이크다운(있을 경우)

---

## 4) 설정 파일

### 4-1. 메인 설정: `config/backtester.yaml`

주요 키:

- `START_DATE`, `END_DATE`: 백테스트 기간
- `INITIAL_CAPITAL`: 초기 자본
- `TRADING_DAYS`: 샤프 계산 등 연환산 기준
- `MASTER`: 정적/동적 비중 및 마스터 리밸런싱 주기
	- `STATIC_RATIO`, `DYNAMIC_RATIO`
	- `REBALANCE_FREQ`: `Monthly|Quarterly|...`
- `STATIC`: 정적 엔진
	- `ASSETS`: ETF/자산 티커 목록
	- `WEIGHTS`: 자산 비중(길이 일치 필요, 아니면 동일가중치)
	- `REBALANCE_FREQ`, `FEES`
- `DYNAMIC`: 동적 엔진
	- `SEASONS`: 시즌 구간 및 후보 티커 풀
	- `REBALANCE_FREQ`, `FEES`
	- `MOMENTUM_WINDOW`, `CORRELATION_WINDOW`
	- `LOGIC_PROFILE` 또는 `LOGIC`
- `DATA.root`: 로컬 데이터 경로
- `RESULTS.root`: 결과 저장 경로

### 4-2. 로직 프로파일: `config/logic_*.yaml`

`config/backtester.yaml`에서 아래처럼 지정하면:

```yaml
DYNAMIC:
	LOGIC_PROFILE: fast_follower
```

백테스터는 `config/logic_fast_follower.yaml`을 읽어 `DYNAMIC.LOGIC`에 자동 반영합니다.

현재 제공되는 예시:
- `config/logic_fast_follower.yaml`
- `config/logic_sagye.yaml`

직접 `DYNAMIC.LOGIC`를 설정 파일에 명시하면(=프로파일보다 우선) 그 값이 사용됩니다.

---

## 5) 동적 엔진(플러그인) 개념

동적 엔진은 아래 부품들을 조합합니다(코드: `core/engine.py`에서 빌드).

- 상관관계 필터(correlation filter)
- 모멘텀 랭커(momentum ranker)
- 셀렉터(selector)
- 할당기(allocator)
- 스톱로스(stop loss)

이 조합을 `config/logic_*.yaml`로 관리하면, 전략 실험(파라미터 스윕)이 쉬워집니다.

---

## 6) 빠른 체크리스트

- 실행이 안 된다면
	- `python -m research.backtester.cli ...` 형태로 실행(모듈 실행)했는지 확인
	- `config/backtester.yaml`에서 티커/비중/시즌 설정이 비어있지 않은지 확인
	- 데이터 소스/캐시 설정은 `config/data.yaml` 확인

---

## 7) 개발 메모

- 실거래/라이브 연동은 `live/`를 참고하세요.
- 백테스트 로직 확장은 `core/`(로직) + `research/backtester/`(오케스트레이션)로 나눠 작업하는 것을 권장합니다.
