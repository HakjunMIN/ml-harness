# 발전량 예측 실험 워크플로

발전량 예측 실험 워크플로는 AI 보조 발전량 예측을 위한 오프라인 데모입니다. 발전소/기상 테이블 데이터를 생성하거나 입력받아 레거시 기준 모델을 평가하고, AIDM으로 결정론적 피처 후보를 탐색하며, AIDD로 안전한 매니페스트를 승격하고 감사 가능한 산출물을 생성합니다.

## 목표 및 안전 범위

데이터 계약부터 보고 가능한 산출물까지 반복 가능한 예측 워크플로를 보여주는 것이 목표입니다. 안전 범위는 의도적으로 제한합니다.

- CSV와 유사한 테이블 데이터 및 SQLite 실험 저장소에서 로컬 실행합니다.
- 검증 게이트와 매니페스트 검사를 통과한 피처 명세만 승격합니다.
- 선택한 출력 디렉터리에 결정론적 Python 피처 코드를 생성합니다.
- 모델을 배포하거나, 운영 작업을 시작하거나, 실시간 시스템을 변경하거나, 자율 운영 의사결정을 하거나, 실제 발전소 제어를 관리하지 않습니다.

## 빠른 시작

```bash
uv sync --all-extras
uv run pytest -q
```

커밋된 `uv.lock` 덕분에 `uv sync --all-extras`는 개발 환경을 재현합니다.
기본 설치는 scikit-learn 기반 흐름을 대상으로 합니다. XGBoost, LightGBM, Optuna 기반 모델 탐색을 실행하려면 별도로 `uv sync --extra model-search`(또는 개발 시 `--all-extras`)를 사용해야 합니다.

전체 데모 실행:

```bash
uv run python -m power_forecasting.cli all --output artifacts/demo --days 60 --plants 3 --seed 42
```

## CLI 예시

파일을 생성하는 명령은 `--output` 하위에 저장합니다. `generate-data`, `aidm`, `aidd`, `all`은 산출물을 생성하며, `legacy`는 데이터셋을 읽고 지표만 표준 출력에 표시합니다.

결정론적 합성 데이터셋 생성:

```bash
uv run python -m power_forecasting.cli generate-data --output artifacts/demo --days 60 --plants 3 --seed 42
```

기존 데이터셋으로 레거시 모델 평가:

```bash
uv run python -m power_forecasting.cli legacy --output artifacts/demo --dataset artifacts/demo/dataset.csv --folds 3
```

AIDM 피처 탐색 실행 및 `experiments.db`, `promotion_manifest.json`, 보고서 생성:

```bash
uv run python -m power_forecasting.cli aidm --output artifacts/demo --dataset artifacts/demo/dataset.csv --folds 3 --seed 42
```

에이전트 제안 JSON과 선택적 레거시 예측 CSV를 함께 비교:

```bash
uv run python -m power_forecasting.cli aidm \
  --output artifacts/demo \
  --dataset artifacts/demo/dataset.csv \
  --proposal .agents/fixtures/research-proposal.json \
  --legacy-predictions .agents/fixtures/legacy-predictions.csv \
  --folds 1
```

모델 탐색 제안(`random_forest`, `xgboost`, `lightgbm`, LightGBM Optuna 탐색)을 실행할 때는 선택 extra를 먼저 설치합니다.

```bash
uv sync --extra model-search
uv run python -m power_forecasting.cli aidm \
  --output artifacts/model-search \
  --dataset .agents/fixtures/valid-dataset.csv \
  --proposal .agents/fixtures/model-search-proposal.json \
  --folds 1 \
  --minimum-improvement 0 \
  --max-plant-regression 1
```

`.agents/fixtures/valid-dataset.csv`는 작은 재사용 fixture이므로 `--folds 1` 예시에만 적합합니다. 실제 증거 생성은 시간순 5-fold를 기본으로 사용하고, 선택 extra 없이 `xgboost`, `lightgbm`, Optuna 탐색이 동작한다고 주장하지 않습니다.

승격된 매니페스트에서 피처 모듈 생성:

```bash
uv run python -m power_forecasting.cli aidd --output artifacts/demo --manifest artifacts/demo/promotion_manifest.json
```

전체 워크플로 실행:

```bash
uv run python -m power_forecasting.cli all --output artifacts/demo --days 60 --plants 3 --seed 42
```

실험에서는 선택적으로 게이트 임계값을 재정의할 수 있습니다.

```bash
uv run python -m power_forecasting.cli all --output artifacts/demo --minimum-improvement 0.01 --max-plant-regression 0.03
```

운영 기본값은 최소 NMAE 개선율 `0.01` 및 발전소별 최대 NMAE 저하율 `0.03`입니다.

## 레거시 모델 의미

레거시 비교는 의도적으로 단순 기준 모델과 풍부한 기준 모델을 함께 사용합니다.

- `Mean`: 발전소 ID와 시각을 사용하는 발전소/시간별 과거 평균 기준 모델입니다.
- `Weather`: `actual_*` 컬럼을 사용하는 실제 기상 진단/오라클 기준 모델입니다. 예측 시점에는 실제 기상을 알 수 없으므로 운영 예측 모델이 아닙니다.
- `ForecastWeather`: 예측 일사량·온도·운량·풍속·용량을 사용하는 선형 기준 모델입니다.
- `Ldaps`: LDAPS 형식의 예측 컬럼을 사용하는 선형 기준 모델입니다.
- `SPOT`: 예측 기상, 용량, 위도, 경도를 사용하는 gradient-boosted 예측 시점 기준 모델입니다. AIDM은 피처 후보를 SPOT과 비교합니다.

## 아키텍처 및 데이터 흐름

```text
합성 데이터 생성기 또는 고객 어댑터
        |
        v
데이터 계약 검증
        |
        +--> 레거시 모델 평가
        |
        +--> 시간순 fold 기반 AIDM 피처 탐색
                  |
                  v
            승격 매니페스트 및 SQLite 실험 저장소
                  |
                  v
            AIDD 매니페스트 검증 및 결정론적 코드 생성
                  |
                  v
            성능 보고서, generated/promoted_features.py, 대시보드/Notebook
```

주요 모듈:

- `power_forecasting.data`: 합성 데이터, 타임스탬프 파싱, 계약 검증
- `power_forecasting.models`: 레거시 모델 정의 및 예측 시점 피처 집합
- `power_forecasting.features`: 결정론적 피처 명세 변환
- `power_forecasting.evaluation`: 시간순 검증 및 지표
- `power_forecasting.aidm`: 후보 카탈로그, 실험 기록, 순위, 게이트, 매니페스트 생성
- `power_forecasting.aidd`: 안전한 매니페스트 검증 및 생성 피처 모듈 렌더링
- `power_forecasting.reporting`: Markdown 성능 보고서 생성
- `dashboard/app.py`: Streamlit 산출물 뷰어

## 데이터 계약

모든 데이터셋은 아래 필수 의미 컬럼을 포함해야 합니다. 추가 컬럼은 허용하지만 워크플로는 아래 이름에만 의존합니다.

| 컬럼 | 형식 및 조건 | 의미 |
| --- | --- | --- |
| `plant_id` | 비어 있지 않은 문자열 | 안정적인 발전소 식별자 |
| `timestamp` | 파싱 가능한 datetime | 예측/실측 시각. `plant_id,timestamp` 조합은 유일해야 함 |
| `capacity_mw` | 0보다 큰 유한 수 | 예측값 clipping 및 NMAE 정규화에 사용하는 발전소 용량 |
| `latitude` | 유한 수 | SPOT용 발전소 위도 |
| `longitude` | 유한 수 | SPOT용 발전소 경도 |
| `actual_irradiance` | 유한 수 | 진단 및 합성 타깃 현실성을 위한 실제 일사량. 승격 피처에서 사용 불가 |
| `actual_temperature` | 유한 수 | 진단용 실제 온도. 승격 피처에서 사용 불가 |
| `actual_cloud_cover` | 유한 수 | 진단용 실제 운량. 승격 피처에서 사용 불가 |
| `actual_wind_speed` | 유한 수 | 진단용 실제 풍속. 승격 피처에서 사용 불가 |
| `forecast_irradiance` | 유한 수 | 예측 시점 일사량 입력 |
| `forecast_temperature` | 유한 수 | 예측 시점 온도 입력 |
| `forecast_cloud_cover` | 유한 수 | 예측 시점 운량 입력 |
| `forecast_wind_speed` | 유한 수 | 예측 시점 풍속 입력 |
| `ldaps_irradiance` | 유한 수 | LDAPS 형식 예측 일사량 |
| `ldaps_temperature` | 유한 수 | LDAPS 형식 예측 온도 |
| `ldaps_cloud_cover` | 유한 수 | LDAPS 형식 예측 운량 |
| `ldaps_humidity` | 유한 수 | LDAPS 형식 예측 습도 |
| `generation_mw` | `[0, capacity_mw]` 범위의 유한 수 | 타깃 발전량. 승격 피처에서 사용 불가 |

검증은 필수 컬럼 누락, 파싱 불가 타임스탬프, 중복 `plant_id,timestamp` 키, 비어 있는 발전소 ID, 유한하지 않은 수치, 0 이하 용량, `[0, capacity_mw]` 범위를 벗어난 타깃을 거부합니다.

### 고객 어댑터 요구사항

고객 어댑터는 CLI 또는 Python API를 호출하기 전에 원천 시스템 데이터를 계약 형식으로 변환해야 합니다.

1. 발전소/시각별 한 행을 생성하고 안정적인 발전소 ID와 일관된 타임스탬프 의미를 유지합니다.
2. 모든 필수 수치 컬럼에 유한 값을 제공하고 용량·발전량은 MW 단위를 사용합니다.
3. 예측 시점 컬럼을 `actual_*` 진단 컬럼 및 `generation_mw`와 분리합니다.
4. 승격 대상 피처에는 예측 시점에 이용 가능한 정보만 사용합니다.
5. 평가 전 `validate_dataset`을 실행하고 계약 오류 시 즉시 실패 처리합니다.
6. 시간순을 유지하거나, 워크플로가 fold 전에 타임스탬프를 정렬·검증하도록 합니다.

## AIDM 탐색, 검증, 게이트 및 실험 저장소

AIDM은 SPOT을 예측 시점 기준 모델로 사용합니다. 다음과 같은 제한된 결정론적 피처 명세 카탈로그를 평가합니다.

- 시간 sine/cosine
- 연중 일자 sine/cosine
- 유효 일사량
- 온도 derating
- 운량 감쇠
- 일사량-온도 상호작용

먼저 단일 피처 그룹을 평가하고, 설정된 상위 후보를 남긴 뒤 2개와 3개 그룹 조합을 평가합니다. 검증은 시간순 fold를 사용하므로 각 검증 블록은 학습 데이터보다 이후 시점에만 위치하며 미래 정보 누수를 방지합니다.

모든 기준 모델 및 후보 실행은 파라미터, 지표, 산출물, 상태, 오류와 함께 `experiments.db`에 저장됩니다. 우승 후보는 다음 승격 게이트를 모두 통과해야 합니다.

- 개선율: 기본적으로 `(baseline_nmae - winner_nmae) / baseline_nmae >= 0.01`
- 발전소별 저하: 기본적으로 각 우승 후보 대비 기준 모델 NMAE 차이는 `<= 0.03`
- 피처 가용성: 승격 피처는 `generation_mw`, `actual_*`, 또는 데이터 계약 밖의 입력을 사용할 수 없음

매니페스트에는 seed, 기준 및 우승 모델 지표, 선택 피처 명세, 임계값, 개선율, 발전소별 변화량, 결정, 실패 게이트가 기록됩니다.

### 에이전트 제안 스키마와 모델 레시피

에이전트는 코드를 생성하지 않고 `schema_version: "1"`인 선언적 JSON 제안만 제출할 수 있습니다. 최상위 키는 `proposal_id`, `rationale`, `baseline`, `feature_sets`, `model_recipes`, `budget`, 선택적 `search`로 제한됩니다. `feature_sets`에는 기존 `FeatureSpec` 사전만 들어가며, AIDD와 동일한 예측 시점 입력 검증으로 `generation_mw`, `actual_*`, 계약 밖 입력을 거부합니다. 시간 피처는 `timestamp`만, 날씨 피처는 예측/LDAPS/메타데이터 입력만, history 피처는 발전소별 엄격한 과거 값만 사용할 수 있습니다. history warmup 결측은 각 fold 안에서 학습 통계로 impute되며 미래·현재 행을 보지 않습니다. `budget.max_evaluations`는 1~50, `budget.top_feature_groups`는 1~10입니다.

지원 모델 레시피는 고정된 허용 파라미터 집합만 사용할 수 있습니다.

- `ridge`: `alpha`가 `0.1`, `1.0`, `10.0` 중 하나인 `SimpleImputer` + `StandardScaler` + `Ridge`
- `hist_gradient_boosting`: `max_iter` `50/100/200`, `learning_rate` `0.03/0.1`, `max_leaf_nodes` `15/31/63` 중 하나인 결정론적 `HistGradientBoostingRegressor(random_state=0)`
- `random_forest`: `n_estimators` `100/200/400`, `max_depth` `8/12/null`, `min_samples_leaf` `1/2/4` 중 하나인 결정론적 `RandomForestRegressor(random_state=0, n_jobs=1)`
- `xgboost`: `n_estimators` `100/200/400`, `max_depth` `4/6/8`, `learning_rate` `0.03/0.1`, `subsample` `0.8/1.0` 중 하나인 `XGBRegressor`. 실행에는 `uv sync --extra model-search`가 필요하며, macOS OpenMP 등 native runtime import 오류는 원인 문자열을 보존해 표시합니다.
- `lightgbm`: `n_estimators` `100/300`, `learning_rate` `0.03/0.1`, `num_leaves` `15/31`, `min_child_samples` `10/20` 중 하나인 `LGBMRegressor`. 실행에는 `uv sync --extra model-search`가 필요합니다.

선택적 `search`는 LightGBM에 대한 bounded Optuna TPE만 지원합니다. 각 trial은 허용된 discrete space 값만 선택하고, 최적 trial은 `selected_lightgbm`으로 한 번 더 재평가됩니다. 예산은 `len(feature_sets) * (len(model_recipes) + search.n_trials + 1 selected 재평가)`로 계산되며, 검색이 없으면 `len(feature_sets) * len(model_recipes)`입니다. 예산을 초과하면 어떤 평가도 실행하지 않고 실패합니다. 기본 SPOT 기준선은 항상 유지됩니다. `--legacy-predictions`를 제공하면 `plant_id,timestamp,prediction_mw`가 평가 행과 정확히 1:1로 일치해야 하며, 예측값은 용량으로 clipping된 뒤 NMAE를 계산합니다. 우승 후보가 레거시 예측 NMAE보다 나쁘면 기존 SPOT 게이트를 통과해도 `legacy_regression` 게이트로 거부됩니다.

## AIDD의 제한된 결정론적 안전성

AIDD는 승격 매니페스트만 읽고 `generated/promoted_features.py`를 생성합니다. 다음을 검증합니다.

- 스키마 버전 및 `decision == "promote"`
- 비어 있지 않은 선택 피처 명세
- 알려진 결정론적 변환 및 원시 리터럴 파라미터
- 중복되지 않는 피처 이름
- 타깃 누수, 실제 기상 입력, 데이터 계약 밖의 예측 시점 입력 부재
- 기준 모델 출처가 정확히 `SPOT`
- 개선율 및 발전소별 변화량이 매니페스트 임계값 충족
- 우승 모델 이름과 선택 피처 명세의 일치

생성 모듈에는 `PROMOTED_FEATURE_SPECS`와 `build_promoted_features(frame)`이 포함됩니다. 런타임 피처 엔진과 동일한 결정론적 변환을 적용하며, 학습·배포·네트워크 접근·자율 동작을 포함하지 않습니다. history 피처(`lag`, `rolling_mean`)는 검증은 가능하지만 AIDD 실행 모듈로 렌더링하지 않으며, 사람이 검토할 patch 요청으로만 다룹니다.

에이전트 레시피가 승격된 경우 AIDD는 추가로 `model-recipe-patch.json`을 생성합니다. 이 파일은 `status: "requires_human_review"`인 UTF-8 LF JSON 요청이며, 선택 모델 레시피, 선택 피처 명세 해시, 우승 지표, 매니페스트 해시만 담습니다. 실행 가능한 코드, 고객 경로, 임의 필드 전달은 포함하지 않으며, 사람이 검토하기 전에는 고객 저장소나 운영 설정을 직접 수정하지 않습니다.

## 산출물 구조

성공적인 `all` 실행은 다음을 생성합니다.

```text
artifacts/demo/
├── dataset.csv
├── experiments.db
├── model-recipe-patch.json        # 에이전트 모델 레시피 승격 시, 사람 검토용 요청
├── promotion_manifest.json
├── performance_report.md
└── generated/
    └── promoted_features.py
```

표준 검토 증적은 `artifacts/demo/promotion_manifest.json` 및 `artifacts/demo/generated/promoted_features.py`로 버전 관리합니다. 이 파일들은 코드 리뷰에서 승격 결정과 생성된 운영용 피처 모듈을 검토할 수 있게 합니다. 매니페스트는 안정적인 재생성을 위해 결정론적 논리 실행 ID를 사용합니다. 데이터셋, SQLite 실험 DB, 성능 보고서, 임시 실행 출력 등 다른 데모 산출물은 로컬에만 저장되며 Git에서 무시됩니다.

거부된 경우에도 `dataset.csv`, `experiments.db`, `promotion_manifest.json`, `performance_report.md`는 진단에 유용합니다. `generated/promoted_features.py`는 승격 매니페스트가 AIDD 검증을 통과한 뒤에만 생성됩니다.

## Notebook 및 대시보드

`notebooks/`의 Notebook은 `Path("artifacts/demo")`를 가정한 간단한 워크플로 예제입니다.

- `01_legacy_baseline.ipynb`
- `02_aidm_feature_discovery.ipynb`
- `03_aidd_promotion.ipynb`

산출물을 생성한 뒤 대시보드를 실행합니다.

```bash
uv run streamlit run dashboard/app.py -- --artifacts artifacts/demo
```

대시보드는 로컬 산출물을 읽어 승격 결정, 우승 지표, AIDM 순위, 실험 실행, 보고서 텍스트, 선택 피처 명세, 발견된 산출물 경로를 표시합니다.

## 코딩 에이전트에서 실험 워크플로와 스킬 사용

이 저장소의 실험 워크플로는 CLI 명령으로 직접 실행할 수도 있고, 코딩 에이전트에게 저장소 로컬 스킬을 명시해 안전 절차를 맡길 수도 있습니다. 슬래시 명령을 지원하는 환경에서는 `/legacy-intake`, `/aidm-experiment`, `/aidd-promotion`, `/release-gate`, `/research-diagnostic`, `/research-proposal`, `/research-verification`, `/research-orchestrator`처럼 요청할 수 있습니다. VS Code Copilot처럼 슬래시 스킬 호출이 고정되어 있지 않은 환경에서는 자연어로 스킬명을 명시하는 방식이 가장 명확합니다.

예시 요청:

```text
legacy-intake 스킬로 fixture 레거시 어댑터부터 검증해줘
aidm-experiment 스킬로 fixture AIDM 실험을 실행하고 promotion_manifest를 설명해줘
aidd-promotion 스킬로 promoted manifest를 검증하고 생성 모듈 컴파일 증적을 확인해줘
release-gate 스킬로 baseline, AIDM, AIDD, compile, human approval 증거를 판정해줘
```

## 선택적 Stage 1 자율 연구 루프

기존 수동 흐름은 변경되지 않습니다. `legacy-intake`로 baseline을 확인하고
`aidm-experiment`에서 사람이 제안 JSON과 명령을 선택하는 방식은 여전히 기본 경로이며,
이 루프를 실행하지 않아도 됩니다. 탐색 단계만 선택적으로 상태 머신으로 묶으려면 다음
fixture 명령을 사용합니다.

```bash
.agents/scripts/run-research-loop.sh --config .agents/fixtures/research-loop.json
```

runner는 기존 프로젝트 안의 `--config` 파일만 읽고 저장소 루트에서 정확히
`uv run python -m power_forecasting.cli research-loop`를 실행합니다. 설정의 입력 경로는
설정 파일 디렉터리에 대해 해석되며, fixture는 합성 `valid-dataset.csv`와 baseline
manifest만 사용하고 출력은 `.agents/runs/research-loop-fixture/` 아래에 둡니다. 환경
비밀값을 export하거나 출력하지 않습니다. `--resume`은 동일한 설정과 체크섬을 가진
비터미널 실행만 재개합니다.

Stage 1은 한 프로필당 최대 한 번, 설정된 `max_iterations`(1~10)와 AIDM proposal/search
budget(각 proposal budget 1~50) 안에서만 반복합니다. 각 run은
`research-config.json`, `state.json`, `journal.jsonl`, iteration별
`research-proposal.json`, `research-notes.json`, AIDM `promotion_manifest.json`,
`experiments.db`, `performance_report.md`, `experiment-evidence.json`,
`experiment-failure.json`(실험 실패 시), `verification.json`과 최종
`research-summary.json`을 SHA-256으로 연결합니다. 성공한 진단만 `diagnosis.json`을 만들며,
진단이 실패하면 프로필을 선택하거나 실험을 실행하지 않고 필요한 내부 상태 전이만 거쳐
terminal `diagnostic-failure.json`(안전한 `rejected_conditions`)으로 대체합니다. 이 경로는
`experiment-failure.json`을 만들지 않습니다. 검증기가 입력 evidence를 malformed로 판단하면 검증기가 `verification.json`에
`status: "invalid"`로 fail-closed 기록을 남깁니다. 검증기 반환값/report가 malformed이거나
orchestrator의 evidence 처리 자체가 실패하면 orchestrator가 별도의
`verification-failure.json`(안전한 reason code만 포함)을 기록합니다. 두 경로 모두 raw
evidence data를 기록하지 않으며 서로 대체 관계가 아닙니다. 상태는
`ready_for_human_review`, `exhausted`, 또는 `failed`에서 멈추며, 터미널 상태는 재개하지 않습니다.

이 루프의 경계는 AIDD 호출, 실행 가능한 코드 생성, merge, deploy 직전입니다.
`research-summary.json`은 연구 진단/제안/검증 결과일 뿐 release 또는 deploy 승인 증거가
아닙니다. 사람은 summary와 전체 증적을 검토한 뒤 기존 수동 AIDD 및 release gate 절차를
별도로 실행해야 합니다. 루프는 source, fixture, 고객 데이터, gate 임계값을 수정하지
않고 고객 행·target·secret을 기록하지 않습니다.

## 포함하지 않은 운영 마이그레이션 및 확장 지점

이 저장소는 로컬 실험 환경이며 운영 플랫폼이 아닙니다. 다음은 확장 지점이며 포함하지 않습니다.

- SQLite 대신 MLflow 또는 관리형 실험 추적 사용
- 실시간 원천 시스템에 연결하는 고객 데이터 어댑터
- 배포 에이전트, 스케줄러, 모델 레지스트리, 서빙 엔드포인트, 롤아웃 메커니즘
- 실시간 모니터링, 알림, 롤백, 운영 장애 자동화
- 검토 없이 운영 동작을 변경하는 자동 진단/Self-Improving Loop

운영 마이그레이션에서는 이 구성 요소를 명시적으로 추가하고, 데이터 계약 경계를 유지하며, 시간순 검증과 배포 전 사람 검토를 유지해야 합니다.

## 테스트

기존 테스트 스위트를 사용합니다.

```bash
uv run python -m pytest -q
uv run python -m compileall -q src dashboard artifacts/demo/generated
```

종단 간 smoke test는 운영 CLI 경로를 실행합니다.

```bash
uv run python -m power_forecasting.cli all --output <tmp> --days 45 --plants 2 --seed 21
```

이 테스트는 승격 결정, 기본 `0.01`/`0.03` 게이트, 비어 있지 않은 선택 피처 명세, `generated/promoted_features.py`, 일정 수준 이상의 보고서를 확인합니다.

성공적으로 실행한 뒤에는 승격 증적을 검증합니다. 아래 스니펫은 기본적으로 `artifacts/demo`를 검사하며 다른 출력 디렉터리를 보려면 `OUTPUT_DIR`을 지정합니다.

<!-- readme-evidence-check: start -->
```bash
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/demo}" uv run python - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR"])
manifest = json.loads((root / "promotion_manifest.json").read_text(encoding="utf-8"))
assert manifest["decision"] == "promote"
assert manifest["selected_specs"]
assert (root / "performance_report.md").stat().st_size > 500
winner_nmae = manifest["winner"]["metrics"]["nmae"]
assert math.isfinite(winner_nmae)
print(winner_nmae)
PY
```
<!-- readme-evidence-check: end -->

예상 결과: 승격된 우승 모델의 NMAE를 출력하고 종료 코드 0으로 종료합니다.

## 문제 해결 및 거부 동작

- `ERROR: dataset not found`: 먼저 `generate-data`를 실행하거나 올바른 `--dataset`을 지정합니다.
- `ERROR: missing required columns` 또는 `invalid timestamps`: 고객 어댑터 출력이 데이터 계약을 충족하도록 수정합니다.
- `ERROR: AIDM rejected promotion`: `promotion_manifest.json`, `failed_gates`, `experiments.db`, `performance_report.md`를 확인합니다. 후보가 개선율, 발전소별 저하율, 피처 가용성 게이트를 만족하지 않을 때 거부는 정상입니다.
- `generated/promoted_features.py` 누락: 매니페스트가 거부되었거나 AIDD 검증에 실패한 것입니다. 워크플로는 안전한 승격이 확인되기 전까지 코드를 생성하지 않습니다.
- Streamlit import 오류: `uv sync --extra dashboard`로 dashboard extra를 설치합니다.

## 레거시 어댑터 실행기(`.agents/`) 안전 사용

`.agents/legacy_adapter/contract.py`는 고객 레거시 예측기를 블랙박스 어댑터로 실행합니다. 어댑터 JSON은 `schema_version: "1"`, 비어 있지 않은 `legacy_command` argv, 매니페스트 디렉터리 아래의 상대 `input_dataset`/`predictions_output`, `required_prediction_columns`, `timeout_seconds(1..3600)`만 허용합니다. 명령은 shell 없이 실행되며 `HARNESS_INPUT_DATASET`, `HARNESS_PREDICTIONS_OUTPUT`, `HARNESS_RUN_DIR` 이름의 환경 변수만 전달합니다. 이 환경 변수 이름은 레거시 어댑터 계약의 호환 API이며, 코딩 에이전트 하니스와는 관련이 없습니다.

Fixture-first 순서:

```bash
PYTHONPATH=.agents uv run python -m legacy_adapter.contract --adapter .agents/fixtures/valid-adapter.json --run-dir .agents/runs/fixture
.agents/scripts/run-legacy.sh --adapter .agents/fixtures/valid-adapter.json --run-dir .agents/runs/legacy
.agents/scripts/run-aidm.sh --dataset .agents/fixtures/valid-dataset.csv --run-dir .agents/runs/aidm --folds 1 --top-single-candidates 1
cp .agents/fixtures/promoted-manifest.json .agents/runs/promotion/promotion_manifest.json
.agents/scripts/verify-promotion.sh --run-dir .agents/runs/promotion
```

증거 파일은 `legacy-evidence.json`, `experiments.db`, `promotion_manifest.json`, `performance_report.md`, `promotion-evidence.json`입니다. 증거에는 체크섬과 상태만 남기며 입력 행 내용, 고객 데이터, 비밀, 환경 변수 값은 기록하지 않습니다. 경로 이탈, 빈 CSV, 필수 예측 컬럼 누락, `actual_*`/`generation_mw` 누수, `decision: reject`, 컴파일 실패는 모두 거부로 처리하고 성공 증거를 만들지 않습니다.

실제 고객 데이터 또는 고객 시스템 실행은 사람이 명시적으로 승인하기 전까지 금지됩니다. AIDD가 생성한 코드는 human approval 이후 검토용 패치 요청으로만 다루며, 에이전트는 배포·머지·고객 시스템 편집을 수행하지 않습니다.
