# 고객 ML 실험·프로덕션 연동 가이드

이 문서는 고객의 기존 ML 실험 환경에 ML Harness 에이전트 스킬을 설치하고, 기존 모델을
black-box baseline으로 연결한 뒤, 검증된 AIDD 산출물을 고객 프로덕션 변경 절차로 전달하는
방법을 설명합니다.

이 프로젝트는 고객 시스템을 직접 수정하거나 배포하지 않습니다. 연결의 끝은 자동 배포가 아니라
**체크섬으로 결속된 증적과 사람이 검토할 patch 요청**입니다.

## 전체 구조

```text
고객 ML 실험 리포
  ├── repo-scoped agent skills
  ├── 고객별 AGENTS.md 정책
  └── black-box adapter
          │
          ├── 표준 prediction CSV
          └── legacy-evidence.json
                    │
                    ▼
ML Harness 연구 환경
  ├── 승인된 학습 데이터 contract
  ├── legacy baseline 비교
  ├── bounded AIDM / research loop
  └── promotion_manifest.json
                    │
                    ▼
AIDD 검증
  ├── generated/promoted_features.py
  ├── model-recipe-patch.json (해당 시)
  └── promotion-evidence.json
                    │
                    ▼
사람 검토 및 release gate
                    │
                    ▼
고객의 기존 CI/CD·registry·serving·rollback 절차
```

| 구간 | 고객이 소유하는 것 | Harness가 소유하는 것 |
| --- | --- | --- |
| 실험 환경 | 데이터 접근 승인, 기존 실행 환경, 모델 명령, 예측 export | 스킬, adapter 계약, baseline 증적 |
| 연구 | 도메인 매핑, 승인된 gate와 예산 | 제한된 proposal, 결정론적 실행, 검증 증적 |
| AIDD | 운영 코드 구조, 테스트, 배포 승인 | 승격 manifest 검증, 피처 모듈과 비실행 recipe patch |
| 운영 | registry, serving, rollout, monitoring, rollback | 직접 변경하거나 배포하지 않음 |

## 1. 고객 실험 리포에 에이전트 스킬 설치

ML Harness를 clone하고 의존성을 설치한 환경에서 고객 ML 리포를 대상으로 실행합니다.

```bash
cd /path/to/ml-harness
uv sync --all-extras
uv run power-forecast init --target /path/to/customer-ml-repository
```

설치 후 고객 리포에는 다음 항목이 추가됩니다.

```text
customer-ml-repository/
├── AGENTS.md
└── .agents/
    ├── plugin.json
    ├── adapter-template.json
    ├── skills/
    ├── scripts/
    ├── harness/
    └── legacy_adapter/
```

설치는 repository scope에만 적용됩니다. 기존 `AGENTS.md` 내용과 충돌하지 않는 사용자
`.agents/` 파일은 보존하며, 같은 경로의 관리 asset을 덮어써야 하는 경우에는 실패합니다.
고객 데이터, credential, 전역 에이전트 설정 또는 MCP 설정은 설치하지 않습니다.

### 고객 정책으로 스킬 강화

설치된 core skill을 직접 완화하지 말고 고객 리포의 `AGENTS.md`에서 다음 정책을 관리합니다.
installer가 추가한 `<!-- ml-harness:begin -->` block 바깥에 기록해야 합니다.

- 고객 데이터 contract와 prediction contract의 버전 및 소유자
- 사용할 수 있는 실험 launcher, container image 또는 가상환경
- 읽을 수 있는 데이터 위치와 쓸 수 있는 `runs/`·`outputs/` 위치
- 금지된 feature, 누수 기준, 최소 개선율, 회귀 한도와 평가 예산
- 사람이 승인해야 하는 단계와 승인 기록 형식
- 운영 리포에서 patch를 받을 담당 팀과 CI/CD·rollback 절차

고객 전용 skill을 추가한다면 core skill과 별도 디렉터리에 두고, 각 skill에 허용 입력, 출력
artifact, 읽기·쓰기 범위, 실패 시 동작, 사람 승인 지점과 중단 조건을 명시합니다. 고객 skill이
core gate를 우회하거나 `ready_for_human_review`을 배포 승인으로 해석해서는 안 됩니다.

## 2. 기존 ML 실험 환경 연결

### 2.1 연결 방식 선택

기존 학습 코드를 Harness 안으로 옮기지 않습니다. 대신 adapter가 고객 환경의 실행 명령을
호출하고 예측 결과만 표준 CSV로 export합니다.

기존 환경별 권장 연결은 다음과 같습니다.

| 기존 환경 | `legacy_command` 권장 형태 |
| --- | --- |
| Python virtualenv | `["/absolute/venv/bin/python", "/absolute/repo/run_baseline.py", "--config", "/absolute/repo/baseline.yaml"]` |
| Conda | 사전 검증한 절대 경로의 environment launcher |
| Docker | Harness 환경 변수를 받아 container에 전달하는 사전 검토된 wrapper executable |
| 사내 launcher | 절대 경로의 launcher와 고정 인자를 literal argv로 지정 |

`legacy_command`는 shell 문자열이 아니라 JSON 문자열 배열입니다. `eval`, command substitution,
pipe, redirect 또는 동적 shell interpolation을 사용하지 않습니다. adapter runner는 고객 shell
환경 전체를 전달하지 않으며 command의 working directory는 adapter manifest 디렉터리입니다.
따라서 interpreter, 고객 entrypoint와 config에는 재현 가능한 절대 경로를 사용하는 편이
안전합니다. secret은 manifest나 argv에 넣지 않습니다. 필요한 경우 승인된 launcher가 workload
identity나 고객 secret provider를 사용하도록 별도로 검토합니다.

### 2.2 Adapter manifest 작성

template을 실행 artifact 영역으로 복사합니다. 실제 고객 데이터와 실행 결과를 `.agents/` 아래에
두거나 commit하지 않습니다.

```bash
cd /path/to/customer-ml-repository
mkdir -p runs/customer-baseline/adapter
cp .agents/adapter-template.json runs/customer-baseline/adapter/adapter.json
```

예시는 다음과 같습니다.

```json
{
  "input_dataset": "input/approved-evaluation.csv",
  "legacy_command": [
    "/absolute/venv/bin/python",
    "/absolute/customer-ml-repository/run_baseline.py"
  ],
  "predictions_output": "generated/predictions.csv",
  "required_prediction_columns": [
    "plant_id",
    "timestamp",
    "prediction_mw"
  ],
  "schema_version": "1",
  "timeout_seconds": 300
}
```

`input_dataset`과 `predictions_output`은 `adapter.json`이 있는 디렉터리 기준의 상대 경로이며,
그 디렉터리 밖으로 나갈 수 없습니다. timeout은 1~3600초입니다.

runner는 고객 명령에 다음 세 환경 변수만 제공합니다.

| 변수 | 의미 |
| --- | --- |
| `HARNESS_INPUT_DATASET` | 승인된 adapter 입력 파일의 절대 경로 |
| `HARNESS_PREDICTIONS_OUTPUT` | 고객 명령이 생성해야 하는 prediction CSV 경로 |
| `HARNESS_RUN_DIR` | baseline 증적 디렉터리 |

고객 entrypoint는 `HARNESS_INPUT_DATASET`을 읽고 `HARNESS_PREDICTIONS_OUTPUT`에 UTF-8 CSV를
작성해야 합니다. CSV에는 중복 없는 header, 한 행 이상의 데이터, manifest에 선언한 모든 필수
열이 있어야 합니다.

### 2.3 Fixture-first 검증

먼저 고객 행이 없는 합성 입력과 안전한 fixture launcher로 동일한 schema와 실행 경로를 검증합니다.

```bash
.agents/scripts/run-legacy.sh \
  --adapter runs/customer-baseline/adapter/adapter.json \
  --run-dir runs/customer-baseline/evidence \
  --run-id customer-baseline-v1
```

성공 기준은 `legacy-evidence.json`의 `status: success`입니다. 증적에는 manifest·입력·예측
checksum, literal argv, 행 수와 필수 열만 남고 고객 원본 행이나 환경 변수 값은 포함되지 않습니다.

fixture가 통과한 뒤에만 사람이 실제 adapter와 데이터 실행을 승인해야 합니다. 승인 후에도
adapter와 데이터는 `runs/` 또는 `outputs/`처럼 commit되지 않는 영역에 둡니다.

## 3. 고객 데이터와 AIDM 연결

### 3.1 현재 도메인 contract 확인

현재 자동 AIDM 구현은 발전량 예측 예제에 맞춰져 있습니다. 주요 키는 `plant_id`, `timestamp`,
target은 `generation_mw`이고, prediction-time weather와 발전소 metadata 열을 사용합니다.
legacy prediction 비교에는 정확히 `plant_id`, `timestamp`, `prediction_mw`가 필요합니다.

다른 도메인에서는 adapter 연결만으로 자동 최적화가 활성화되지 않습니다. 먼저 다음 항목을 코드와
테스트로 구현해야 합니다.

1. 고객 dataset·prediction schema와 시간순 split 규칙
2. prediction 시점에 실제로 사용 가능한 입력 검증
3. 도메인별 transform과 estimator 구현
4. leakage·회귀·최소 개선 gate
5. aggregate-only 진단 및 증적 계약
6. 허용된 feature set, recipe와 bounded search space를 담은 versioned catalog

catalog는 이미 구현된 기능의 allowlist입니다. JSON catalog에 이름을 추가하는 것만으로 새로운
transform이나 estimator가 실행되지는 않습니다.

### 3.2 수동 경로로 첫 통합 검증

새 고객 연결은 자동 연구 루프보다 수동 경로를 먼저 검증하는 것을 권장합니다.

```text
legacy-intake
  -> aidm-experiment
  -> aidd-promotion
  -> human-review
  -> release-gate
```

발전량 contract에 맞는 승인 데이터의 예:

```bash
cd /path/to/ml-harness
.agents/scripts/run-aidm.sh \
  --dataset runs/customer-study/approved-dataset.csv \
  --catalog configs/optimization-catalog.v1.json \
  --proposal runs/customer-study/research-proposal.json \
  --legacy-predictions runs/customer-study/legacy-predictions.csv \
  --run-dir runs/customer-study/aidm \
  --folds 5 \
  --minimum-improvement 0.02 \
  --max-plant-regression 0.01
```

gate와 예산은 실행 전에 고객이 승인해야 합니다. 실행 중 gate, baseline, catalog, 데이터 또는
증적을 바꾸어 승격을 유도하지 않습니다.

수동 경로가 재현 가능하고 도메인 contract가 검증된 뒤에는 `research-orchestrator`를 이용해
`proposal -> AIDM -> verification` cycle을 자동화할 수 있습니다. 자동 루프도 최대 10회,
run-wide 50 evaluations 안에서만 동작하며 terminal 상태에서 멈춥니다.

## 4. AIDD 산출물을 프로덕션 변경 절차에 연결

### 4.1 AIDD 검증 실행

AIDM의 `promotion_manifest.json`이 `decision: promote`인 경우에만 같은 run directory에서
검증합니다.

```bash
.agents/scripts/verify-promotion.sh \
  --run-dir runs/customer-study/aidm
```

성공하면 다음 artifact가 생성됩니다.

| Artifact | 용도 |
| --- | --- |
| `promotion_manifest.json` | 승격된 피처·모델과 gate 결과 |
| `generated/promoted_features.py` | 결정론적인 승격 피처 정의 |
| `model-recipe-patch.json` | 선택된 모델 recipe의 비실행 patch 요청; 해당 시에만 생성 |
| `promotion-evidence.json` | manifest와 생성 artifact의 SHA-256 결속 |

`model-recipe-patch.json`의 상태는 `requires_human_review`이며 실행 코드가 아닙니다.
`promotion-evidence.json`의 성공도 사람 승인이나 배포 승인이 아닙니다.

### 4.2 운영 리포로 전달할 bundle

운영 담당자에게는 원본 고객 데이터 대신 다음 항목을 전달합니다.

- 위 artifact 중 생성된 항목과 각 SHA-256
- baseline `legacy-evidence.json`
- `experiments.db`와 `performance_report.md`
- 사용한 dataset·catalog·proposal의 checksum
- 시간순 fold, gate, aggregate metric과 알려진 입력 가용성 제약
- rollback 기준과 담당자
- 정확한 manifest checksum을 참조하는 human-review 결정

연구 루프의 `research-summary.json`, `state.json` 또는 `ready_for_human_review` 상태만으로는
release를 요청할 수 없습니다.

### 4.3 운영 코드에 반영

`generated/promoted_features.py`는 standalone 파일이 아닙니다. 현재 생성물은 pandas와
`power_forecasting.features`를 import하며, history 피처가 있으면 `power_forecasting.data`도
사용합니다. 고객은 다음 중 하나를 명시적으로 선택해 reviewed patch로 구현해야 합니다.

1. 호환되는 `power-forecasting` package version을 운영 dependency로 pin하고 생성 모듈을 통합
2. manifest의 결정론적 feature spec을 고객 feature library로 port하고 parity test로 동등성 검증

모델 recipe는 `model-recipe-patch.json`을 참고해 고객의 기존 trainer·registry 설정에 사람이
반영합니다. Harness가 고객 모델 파일, serving 설정 또는 registry를 직접 수정해서는 안 됩니다.

운영 patch에는 최소한 다음 검증이 필요합니다.

- offline AIDM과 운영 feature output의 golden-data parity
- 필수 열 누락, null, 비수치·비유한 값과 prediction-time unavailable input 거부
- `actual_*`, 현재·미래 target과 같은 leakage 입력 차단
- history 피처 사용 시 고유한 entity/time key, 정렬과 strict-prior 보장
- training-serving schema 및 timezone 일치
- 기존 모델과의 shadow 또는 canary 비교
- metric·data drift 모니터링과 즉시 rollback 경로

history feature는 임의의 단일 inference row만으로 계산할 수 없습니다. 고객 serving 계층이
동일 entity의 검증된 과거 행을 제공하고, 현재 target이나 미래 행이 섞이지 않도록 보장해야 합니다.

## 5. Release gate

release 요청은 다음 조건을 모두 만족할 때만 가능합니다.

| 조건 | 필수 증적 |
| --- | --- |
| 기존 baseline 성공 | `legacy-evidence.json`의 `status: success` |
| AIDM 승격 | `promotion_manifest.json`의 `decision: promote` |
| 실험 재현성 | `experiments.db`, `performance_report.md`, 입력 checksum |
| AIDD 성공 | `promotion-evidence.json`과 checksum이 일치하는 생성 모듈 |
| compile 성공 | `verify-promotion.sh` 검증 |
| 사람 승인 | 정확한 manifest checksum과 patch 범위를 참조하는 명시적 승인 |
| 운영 준비 | 고객 CI, staging, monitoring, rollback 검증 |

하나라도 없거나 checksum이 일치하지 않으면 fail closed입니다. release gate 통과 후에도 실제 merge,
registry 등록, rollout과 rollback은 고객의 기존 프로덕션 권한 및 change-management 절차가
수행합니다.

## 고객 도입 체크리스트

- [ ] 고객 ML 리포에 repo-scoped plugin을 설치했다.
- [ ] 고객 정책을 관리 block 바깥의 `AGENTS.md`에 명시했다.
- [ ] 합성 fixture로 adapter와 prediction schema를 검증했다.
- [ ] 실제 데이터 실행에 사람 승인을 받았고 데이터·secret을 commit하지 않았다.
- [ ] 고객 도메인 contract, 시간순 평가와 leakage gate를 구현·검증했다.
- [ ] 첫 통합은 수동 skill path로 재현했다.
- [ ] AIDM manifest와 AIDD artifact checksum이 일치한다.
- [ ] 생성 피처의 runtime dependency 또는 porting 전략을 명시했다.
- [ ] recipe patch와 운영 patch를 사람이 검토했다.
- [ ] staging, monitoring, rollback과 정확한 manifest checksum 승인이 준비됐다.
