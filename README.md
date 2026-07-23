# Legacy ML Optimization Agent Harness

이 저장소는 **기존 레거시 ML을 에이전트 스킬 프레임워크로 분석하고 개선하는 프로젝트**입니다.
에이전트는 기존 모델을 기준선으로 연결하고, 제한된 피처·모델·하이퍼파라미터 후보를 제안하며,
결정론적 runner가 실험과 검증을 수행합니다.

스킬 프레임워크의 중심은 다음 두 영역입니다.

- [`AGENTS.md`](AGENTS.md): 에이전트가 자율 연구 루프를 운영하는 순서와 중단 조건
- [`.agents/`](.agents): 스킬, 안전한 실행 스크립트, fixture, 레거시 adapter 계약
- [`.agents/skills/`](.agents/skills): 역할별 읽기·쓰기 권한과 fail-closed 규칙

현재 구현은 발전량 예측을 예제로 사용하지만, 핵심 목적은 특정 모델 하나가 아니라
**레거시 ML 개선 과정을 코딩 에이전트가 안전하고 재현 가능하게 수행하도록 만드는 하네스**입니다.

기존 상세 문서는 [`REAMDE.old.md`](REAMDE.old.md)에 보존되어 있습니다.

## 5분 시작

### 1. 환경 설치

```bash
uv sync --all-extras
```

기본 의존성은 NumPy, pandas, scikit-learn입니다. XGBoost, LightGBM, Optuna 기반 탐색에는
`model-search` extra가 필요합니다.

### 2. 다른 ML 리포에 repo-scoped plugin 설치

이 하네스를 다른 ML 리포의 workspace 스킬로 사용하려면, 이 저장소를 clone한 상태에서 대상 리포를
지정해 설치합니다. 설치 범위는 **대상 리포 한 곳**이며 전역 설정, MCP 설정, 고객 데이터는 변경하지
않습니다.

```bash
cd /path/to/ml-harness
uv run power-forecast init --target /path/to/consumer-ml-repository
```

설치 후 대상 리포에는 다음 관리 asset이 생깁니다.

```text
consumer-ml-repository/
├── AGENTS.md                         # 기존 내용 뒤에 관리 block만 추가
└── .agents/
    ├── plugin.json                    # ml-harness, scope: repo marker
    ├── adapter-template.json          # 고객 값이 없는 adapter manifest 시작점
    ├── skills/                        # Copilot workspace에서 발견하는 role skills
    ├── scripts/                       # run-legacy.sh 등 안전한 runner
    ├── harness/
    └── legacy_adapter/
```

대상 리포의 기존 `AGENTS.md`와 `.agents/` 사용자 파일은 보존합니다. 설치 asset과 같은 경로에
이미 파일이 있으면 installer는 덮어쓰지 않고 실패합니다. 첫 설치가 끝난 뒤 같은 명령을 다시
실행하는 것은 marker 검증 후 성공하므로 안전합니다.

adapter를 만들 때는 대상 리포에서 template을 별도 경로로 복사해 승인된 local input과 literal argv를
채웁니다. template 자체에는 실행 가능한 고객 경로나 secret이 없습니다.

```bash
cd /path/to/consumer-ml-repository
mkdir -p adapters/baseline
cp .agents/adapter-template.json adapters/baseline/adapter.json
# adapter.json의 placeholder를 승인된 local 값으로 교체한 뒤 fixture-first 검증
.agents/scripts/run-legacy.sh \
  --adapter adapters/baseline/adapter.json \
  --run-dir runs/legacy-baseline
```

설치된 `legacy-intake` 스킬과 black-box adapter contract는 일반적인 레거시 ML 연결에 사용할 수
있습니다. 반면 자동 AIDM research profile과 발전량 데이터 contract는 현재 power forecasting 예제에
맞춰져 있으므로, 다른 도메인에서 자동 최적화를 실행하기 전에 해당 domain contract와 catalog를
명시적으로 설계해야 합니다.

### 3. 에이전트에게 자율 연구를 요청

코딩 에이전트에서 다음처럼 요청합니다.

```text
research-orchestrator 스킬을 사용해서
.agents/fixtures/research-loop.json을 최대 3회 실행하고
terminal 상태까지 진행해줘
```

에이전트는 `AGENTS.md`에 따라 최대 반복 횟수를 확인하고, 필요한 경우
`agent_proposals: true`인 repository-local config를 `runs/` 아래에 준비합니다.
이후 진단, proposal 작성, AIDM 실험, 증적 검증을 반복합니다.

```text
legacy baseline
  -> diagnosis
  -> awaiting_proposal
  -> bounded research proposal
  -> AIDM experiment
  -> verification
  -> iterate | ready_for_human_review | exhausted | failed
```

`ready_for_human_review`은 배포 준비 완료가 아니라, 사람이 검토할 수 있는 증적이 준비됐다는 뜻입니다.

### 4. runner를 직접 확인

에이전트 오케스트레이션 아래에서 사용하는 실제 runner는 다음과 같습니다.

```bash
.agents/scripts/run-research-loop.sh \
  --config .agents/fixtures/research-loop.json
```

비터미널 실행은 동일한 config로만 재개할 수 있습니다.

```bash
.agents/scripts/run-research-loop.sh \
  --config runs/<research-config>.json \
  --resume
```

### 4. optimization catalog 확인

[`configs/optimization-catalog.v1.json`](configs/optimization-catalog.v1.json)은 profiles, feature sets, direct recipes, allowed parameter values, and bounded TPE space의 versioned external owner입니다. research-loop config의 `catalog_path`는 repository root가 아니라 research-loop config 파일이 있는 디렉터리를 기준으로 해석됩니다. 따라서 `runs/<id>/` 아래 config는 다음처럼 지정합니다.

```json
{
  "catalog_path": "../../configs/optimization-catalog.v1.json"
}
```

수동 proposal 실행도 default catalog을 `--catalog`로 명시합니다.

```bash
.agents/scripts/run-aidm.sh \
  --dataset .agents/fixtures/valid-dataset.csv \
  --catalog configs/optimization-catalog.v1.json \
  --proposal .agents/fixtures/research-proposal.json \
  --run-dir runs/manual-aidm \
  --folds 1
```

Catalog는 선언형 allowlist일 뿐입니다. Python still owns supported transforms, estimator
implementations, parameter types, leakage prevention, budgets, and gates. 따라서 catalog는
cannot add code or new estimator capabilities. Catalog SHA-256은
research config, state, journal, handoffs에 결속되며, catalog 변경은 resume 시 fail closed됩니다.

## 무엇을 최적화하는가

자율 연구 루프는 기존 ML을 기준선으로 유지하면서 다음 후보를 제한된 예산 안에서 탐색합니다.

| 영역 | 현재 지원 |
| --- | --- |
| 피처 탐색 | 예측 시점 날씨, 시간 주기, 상호작용, 발전소별 strict-prior lag·rolling feature |
| 모델 탐색 | Ridge, HistGradientBoosting, RandomForest, XGBoost, LightGBM |
| 하이퍼파라미터 최적화 | allowlist 기반 discrete parameter 비교와 bounded LightGBM Optuna TPE |
| 평가 | 시간순 fold, MAE·RMSE·NMAE, 기존 SPOT 및 선택적 legacy prediction 비교 |
| 승격 판단 | 최소 개선율, 발전소별 최대 회귀, 누수·입력 가용성 gate |
| 증적 | SQLite 실험 기록, Markdown 보고서, manifest, verification, SHA-256 checksum |

에이전트는 임의 Python estimator 코드나 callback을 작성하지 않습니다. 다음 형태의
`research-proposal.json`만 만들 수 있습니다.

```json
{
  "schema_version": "1",
  "baseline": {"model": "SPOT"},
  "feature_sets": [],
  "model_recipes": [],
  "budget": {
    "max_evaluations": 20,
    "top_feature_groups": 3
  }
}
```

runner가 schema, catalog 포함 여부, 누수, 중복 proposal, 후보 수와 평가 예산을 검증한 뒤에만
실제 학습을 수행합니다.

## 현재 지원하지 않는 범위

현재 모델 탐색은 **여러 개별 모델 후보를 비교해 하나를 선택**합니다.
RandomForest와 LightGBM을 함께 평가할 수는 있지만, 두 모델의 예측을 결합하지는 않습니다.

다음은 향후 확장 범위입니다.

- weighted blending, stacking 같은 prediction ensemble
- ensemble 후보별 독립 provenance와 평가 budget
- ensemble 구성 요소별 발전소 회귀 gate
- 모델 registry, serving, rollout, rollback
- 사람 승인 없는 자동 merge·deploy

prediction ensemble은 단순 모델 목록 추가가 아니라 별도의 증적 계약과 승격 gate가 필요합니다.

## 에이전트 스킬 프레임워크

```text
AGENTS.md
.agents/
├── skills/
│   ├── legacy-intake/
│   ├── aidm-experiment/
│   ├── aidd-promotion/
│   ├── human-review/
│   ├── release-gate/
│   ├── research-diagnostic/
│   ├── research-proposal/
│   ├── research-verification/
│   └── research-orchestrator/
├── scripts/
├── fixtures/
└── legacy_adapter/
```

| 스킬 | 책임 | 중단 경계 |
| --- | --- | --- |
| `legacy-intake` | 기존 고객 ML adapter와 baseline 증적 준비 | 승인되지 않은 고객 데이터 실행 전 |
| `research-diagnostic` | aggregate-only 품질·누수·profile 진단 | proposal 또는 실험 실행 전 |
| `research-proposal` | catalog 안에서 feature·recipe·TPE 가설 JSON 작성 | 모델 학습 전 |
| `aidm-experiment` | proposal 검증, 학습, 시간순 fold 평가, gate 판정 | AIDD 호출 전 |
| `research-verification` | manifest·보고서·DB·checksum 재검증 | promotion 또는 release 전 |
| `research-orchestrator` | proposal → AIDM → verification cycle 조정 | Stage 1 terminal 상태 |
| `aidd-promotion` | promoted manifest와 생성 모듈·patch 증적 검증 | merge·deploy 전 |
| `human-review` | 안전한 증적 요약과 다음 행동 요청 | 자동 승인 전 |
| `release-gate` | baseline부터 명시적 사람 승인까지 최종 점검 | release 실행 전 |

## 자동 실행과 수동 실행

### one-shot orchestration

사용자는 한 번 요청하고, `research-orchestrator`가 terminal 상태까지 cycle을 조정합니다.

```text
research-orchestrator
  -> run runner
  -> invoke research-proposal on awaiting_proposal
  -> resume
  -> repeat without another user request
  -> invoke human-review on ready_for_human_review
```

각 cycle은 다음 aggregate-only 진행 정보를 표시합니다.

```text
iteration 2/10 · profile history_tree · evaluations 14/50 · last result reject:<reason>
```

### manual skill-by-skill path

사람이 각 단계를 직접 통제할 수도 있습니다.

```text
research-diagnostic
  -> research-proposal
  -> aidm-experiment
  -> research-verification
  -> human-review
```

두 경로는 동일한 proposal schema, runner, gate, state machine과 checksum 증적을 사용합니다.
`research-proposal`은 무엇을 실험할지 정의하고, `aidm-experiment`는 실제 학습과 평가를 수행하므로
두 역할은 합쳐지지 않습니다.

## 레거시 ML 연결

고객의 기존 모델은 내부 구현을 변경하지 않고 black-box adapter로 연결합니다.

```bash
.agents/scripts/run-legacy.sh \
  --adapter .agents/fixtures/valid-adapter.json \
  --run-dir runs/legacy-fixture
```

성공 시 `legacy-evidence.json`과 표준 prediction CSV가 생성됩니다. adapter는 argv array,
timeout, repository-local path, 필수 prediction column을 검증하며 `shell=False`로 실행됩니다.

실제 고객 데이터 사용은 명시적 승인이 필요합니다. 고객 데이터 행, target, actual 값, secret,
credential은 에이전트 proposal이나 review 출력에 포함하지 않습니다.

## 수동 AIDM과 AIDD 예시

fixture proposal로 AIDM을 실행합니다.

```bash
.agents/scripts/run-aidm.sh \
  --dataset .agents/fixtures/valid-dataset.csv \
  --catalog configs/optimization-catalog.v1.json \
  --proposal .agents/fixtures/model-search-proposal.json \
  --legacy-predictions .agents/fixtures/legacy-predictions.csv \
  --run-dir runs/manual-aidm \
  --folds 1 \
  --minimum-improvement 0 \
  --max-plant-regression 1
```

fixture는 작기 때문에 `--folds 1`을 사용합니다. 실제 검토 증적은 기본적으로 5개의 시간순 fold를
사용해야 합니다.

manifest가 `decision: promote`일 때만 AIDD 검증을 요청합니다.

```bash
.agents/scripts/verify-promotion.sh \
  --run-dir runs/manual-aidm
```

이 과정은 `promotion-evidence.json`, 컴파일된 `generated/promoted_features.py`, 선택적으로
`model-recipe-patch.json`을 준비합니다. 성공 자체는 human approval이 아닙니다.

## 안전성과 재현성

- 모든 검증 block은 학습 block보다 시간상 뒤에 위치합니다.
- `generation_mw`, `actual_*`, 미래 행과 현재 행의 target-derived 값은 prediction feature로 금지합니다.
- history transform은 발전소별 strict-prior 값만 사용합니다.
- `max_iterations`와 `fold_count`는 각각 1~10입니다.
- proposal별 평가 예산은 1~50이고 agent-proposal run 전체 예산은 50입니다.
- config, state, journal과 role artifact는 SHA-256으로 결속됩니다.
- config 변경, stale artifact, malformed JSON, checksum mismatch는 fail closed입니다.
- gate 또는 provenance 검증이 실패하면 promotion을 거부하고 증적을 보존합니다.
- `ready_for_human_review`은 release approval이 아닙니다.
- 에이전트는 고객 시스템을 수정하거나 merge·deploy하지 않습니다.

실패도 증적으로 남습니다.

- 진단 실패: `diagnostic-failure.json`
- 실험 실패: `experiment-failure.json`
- malformed verification 입력: `verification.json`의 `status: invalid`
- verification 처리 실패: `verification-failure.json`
- 반복 소진: `exhaustion.json`

## 주요 산출물

```text
runs/<run-id>/ 또는 outputs/<run-id>/
├── research-config.json
├── state.json
├── journal.jsonl
├── diagnosis.json
├── research-summary.json
└── iterations/
    └── <iteration>-<profile>/
        ├── proposal-context.json
        ├── proposal-catalog.json
        ├── research-proposal.json
        ├── research-notes.json
        ├── experiments.db
        ├── performance_report.md
        ├── promotion_manifest.json
        ├── experiment-evidence.json
        └── verification.json
```

`runs/`와 `outputs/`는 실행 증적용 root 디렉터리이며 Git에서 무시됩니다.
`.agents/`는 재사용 가능한 framework asset을 보관하므로 실행 결과를 저장하지 않습니다.

## 데모 노트북

| 노트북 | 내용 |
| --- | --- |
| [`00_legacy_power_forecasting_models.ipynb`](notebooks/00_legacy_power_forecasting_models.ipynb) | 레거시 모델과 평가 방식 설명 |
| [`01_legacy_baseline.ipynb`](notebooks/01_legacy_baseline.ipynb) | 개선 전 baseline 생성 |
| [`02_manual_skill_path.ipynb`](notebooks/02_manual_skill_path.ipynb) | 사람이 단계별로 제어하는 AIDM/AIDD 경로 |
| [`03_auto_research_path.ipynb`](notebooks/03_auto_research_path.ipynb) | agent-proposal 자율 연구 루프 |

## Python CLI

합성 데이터에서 전체 기본 흐름을 실행하려면:

```bash
uv run python -m power_forecasting.cli all \
  --output artifacts/demo \
  --days 60 \
  --plants 3 \
  --seed 42
```

대시보드:

```bash
uv run streamlit run dashboard/app.py -- --artifacts artifacts/demo
```

## 개발과 테스트

```bash
uv run python -m pytest -q
```

성공한 demo의 winner NMAE를 확인하는 검증 스니펫:

<!-- readme-evidence-check: start -->
```bash
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/demo}" uv run python - <<'PY'
import json
import math
import os
from pathlib import Path

output = Path(os.environ["OUTPUT_DIR"])
manifest = json.loads((output / "promotion_manifest.json").read_text(encoding="utf-8"))
assert manifest["decision"] == "promote"
winner_nmae = manifest["winner"]["metrics"]["nmae"]
assert math.isfinite(winner_nmae)
print(winner_nmae)
PY
```
<!-- readme-evidence-check: end -->

## 프로젝트 경계

이 저장소는 로컬 연구·검토 하네스이며 운영 ML 플랫폼이 아닙니다.

- 실제 고객 adapter와 데이터 사용에는 명시적 승인이 필요합니다.
- 연구 결과는 자동으로 고객 저장소에 patch되지 않습니다.
- AIDD 산출물은 별도의 human-review 대상입니다.
- 승격된 history 피처(`lag`, `rolling_mean`)는 고유한 `plant_id`/`timestamp` 키를 가진 history·target 결합 frame에서 발전소별 strict-prior 행만 사용합니다.
- release-gate는 정확한 manifest checksum에 연결된 명시적 사람 승인을 요구합니다.
- chat 응답, research summary, AIDD 성공은 human approval을 대체하지 않습니다.
