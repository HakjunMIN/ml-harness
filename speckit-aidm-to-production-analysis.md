# AIDM → 프로덕션 경로를 Spec Kit(SDD)으로 전환하는 방안 분석

- 작성일: 2026-08-19
- 대상 저장소: `ml-harness` (발전량 예측 harness)
- 질문: **"AIDM의 결과가 스펙으로 기록되고, 그 스펙대로 프로덕션에 반영되게 할 수 있는가?"**
- 결론 요약: **가능하고, 권장한다. 단 "전면 대체"가 아니라 "AIDD 자리를 Spec Kit이 감싸는 하이브리드"가 정답이다.**

---

## 0. 요약 (TL;DR)

| 항목 | 판단 |
| --- | --- |
| AIDM → AIDD 구간 전체를 Spec Kit으로 대체 | ❌ 비권장. AIDD의 결정론적 검증·컴파일·checksum 결속을 LLM 생성으로 대체하면 fail-closed 보증이 깨진다. |
| AIDM 결과를 **spec.md로 기계 생성**(`promotion_manifest.json` → `spec.md`) | ✅ 강력 권장. 스펙이 사람이 읽는 계약이 되고, checksum으로 증적에 결속된다. |
| 그 스펙을 **고객 프로덕션 리포에서 `/speckit.plan` → `/speckit.tasks` → `/speckit.implement`로 반영** | ✅ 권장. 현재 `CUSTOMER_INTEGRATION_GUIDE.md` §4.3 "운영 코드에 반영" 단계가 유일하게 수작업·비구조화 구간인데, 바로 여기가 Spec Kit의 sweet spot이다. |
| AIDD를 폐기 | ❌ 유지. AIDD는 Spec Kit 워크플로에서 **spec을 생성하는 결정론적 생산자 + constitution gate 집행자**로 재배치한다. |
| 도입 형태 | Spec Kit **preset + extension + bundle** 조합. 코어를 포크하지 않는다. |

핵심 통찰: 이 저장소는 이미 SDD를 하고 있다. `docs/superpowers/specs/`에 11개의 design spec이 있고, `promotion_manifest.json`은 사실상 **기계가 쓴 스펙**이다. 다만 (1) 스펙 포맷이 표준화되어 있지 않고, (2) 스펙에서 프로덕션 코드로 가는 경로가 사람의 손에 맡겨져 있다. Spec Kit은 정확히 이 두 구멍을 메운다.

---

## 1. 현재 아키텍처 분석

### 1.1 현재 파이프라인

```text
legacy-intake            → legacy-evidence.json (baseline)
  ↓
research-diagnostic      → diagnosis.json
  ↓
research-proposal        → research-proposal.json      ← 에이전트가 쓰는 "가설"
  ↓
aidm-experiment          → promotion_manifest.json     ← ★ AIDM의 결과물
                            experiments.db
                            performance_report.md
                            experiment-evidence.json
  ↓
research-verification    → verification.json
  ↓
aidd-promotion           → generated/promoted_features.py   ← 결정론적 코드 생성
                            model-recipe-patch.json          ← 비실행 JSON 요청
                            promotion-evidence.json
  ↓
human-review             → 사람 결정
  ↓
release-gate             → fail-closed 최종 점검
  ↓
[??? 수작업 ???]         → 고객 프로덕션 리포 반영
```

### 1.2 각 산출물의 성격

`promotion_manifest.json`(fixture 기준)은 이미 스펙의 모든 구성요소를 갖고 있다.

```json
{
  "schema_version": "1",
  "decision": "promote",
  "baseline":  { "model": "SPOT", "metrics": { "nmae": 0.2 }, "run_id": "baseline-fixture" },
  "winner":    { "name": "hour_sin", "metrics": { "nmae": 0.15 }, "run_id": "winner-fixture" },
  "improvement_ratio": 0.25,
  "per_plant_deltas": { "plant-a": -0.05 },
  "thresholds": { "minimum_improvement": 0.0, "max_plant_regression": 1.0 },
  "failed_gates": [],
  "seed": 7,
  "selected_specs": [
    {
      "name": "hour_sin",
      "version": "1",
      "transform": "cyclic_hour",
      "inputs": ["timestamp"],
      "parameters": {},
      "rationale": "Synthetic fixture prediction-time hour signal."
    }
  ]
}
```

매핑해 보면 놀랍도록 SDD 구조와 일치한다.

| manifest 필드 | SDD 대응 개념 |
| --- | --- |
| `selected_specs[].name/transform/inputs/parameters` | Functional Requirements (FR) — "무엇을" |
| `selected_specs[].rationale` | Why / 근거 |
| `winner.metrics`, `baseline.metrics`, `improvement_ratio` | Success Criteria (SC) — 측정 가능한 성공 기준 |
| `thresholds`, `failed_gates` | Constitution Check (게이트 통과 증적) |
| `per_plant_deltas` | Edge Cases / 발전소별 회귀 검증 |
| `seed`, `run_id` | Assumptions / 재현성 전제 |
| `selected_model_recipe` (agentic) | Technical Context (plan 쪽 입력) |

즉 `selected_specs`의 필드명이 이미 "spec"이다. **AIDM은 이미 스펙을 만들고 있는데, 그 스펙이 JSON이라 사람이 읽고 리뷰하고 프로덕션 팀에 전달하기 어려울 뿐이다.**

### 1.3 현재 구조의 실제 병목

`CUSTOMER_INTEGRATION_GUIDE.md`의 단계를 보면:

- §1 스킬 설치 — 자동화됨 (`power-forecast init --target`)
- §2 레거시 연결 — 자동화됨 (adapter contract + `run-legacy.sh`)
- §3 AIDM 실행 — 자동화됨 (`run-aidm.sh`)
- §4.1 AIDD 검증 — 자동화됨 (`verify-promotion.sh`)
- §4.2 bundle 전달 — 반자동
- **§4.3 "운영 코드에 반영" — 완전 수작업** ← 병목
- §5 release gate — 체크리스트

문제 지점을 구체화하면:

1. **의도 손실(intent loss)**: 운영 엔지니어가 받는 것은 `generated/promoted_features.py`(기계 생성 코드) + `model-recipe-patch.json`(비실행 JSON)이다. *왜* 이 피처를 넣는지, *어떤 조건에서* 유효한지, *무엇이 깨지면 롤백해야 하는지*는 `performance_report.md`를 사람이 읽어야 알 수 있다.
2. **추적성 단절**: 운영 리포의 PR에는 harness 쪽 run-id/checksum과의 결속이 남지 않는다. 6개월 뒤 "이 피처 왜 있지?"에 답할 수 없다.
3. **프로덕션 반영의 형태가 리포마다 다름**: 고객 파이프라인은 Airflow일 수도, dbt일 수도, sklearn Pipeline일 수도 있다. `promoted_features.py`를 그대로 넣을 수 있는 리포는 드물다. 즉 **"어댑테이션 작업"이 필연적으로 존재하고, 이게 리뷰되지 않은 채 수작업으로 일어난다.**
4. **재실행 불가**: 스펙이 없으니 "같은 결정을 다른 리포에 다시 적용"이 불가능하다.

Spec Kit은 3·4번(HOW를 리포마다 다르게, 그러나 WHAT은 고정)에 정확히 대응하는 도구다.

---

## 2. Spec Kit 최신 현황 (2026-08 기준, 원문 확인)

출처: `github/spec-kit` main 브랜치 README, `templates/spec-template.md`, `templates/plan-template.md`, `templates/tasks-template.md` 원문.

### 2.1 설치와 CLI

```bash
# PyPI (권장)
uv tool install specify-cli

# 또는 릴리스 태그 고정 (재현성 필요 시 — 우리 케이스는 반드시 태그 고정)
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git@vX.Y.Z

specify init my-project --integration copilot
specify init --here --force --non-interactive --integration copilot   # 기존 리포/CI용
specify self check      # 업데이트 확인 (read-only)
specify self upgrade --tag vX.Y.Z
```

> ⚠️ **재현성 주의**: 우리 harness는 checksum fail-closed가 생명이다. `uvx ... @main` 같은 부동(floating) 참조는 금지하고 **반드시 릴리스 태그를 고정**해야 한다. `SPECIFY_UPGRADE_TIMEOUT_SECS`로 설치 subprocess 시간도 제한 가능.

30개 이상의 에이전트를 지원하며, GitHub Copilot CLI(이 저장소가 쓰는 환경)도 포함된다. 다만 **Copilot CLI는 `/speckit.*` 슬래시 명령이 아니라 `/agents`로 에이전트를 선택하거나 프롬프트에서 직접 지칭**한다. Codex CLI / Command Code skills 모드는 `$speckit-*`를 쓴다.

`--integration copilot --integration-options="--skills"`를 주면 슬래시 명령 프롬프트 파일 대신 **agent skill**로 설치된다. → **이 저장소의 `.agents/skills/` 패턴과 정확히 같은 형태**이므로 통합 마찰이 매우 낮다.

### 2.2 명령 체계 (구 `/specify` → 신 `/speckit.specify` 네임스페이스로 개명됨)

**코어 명령**

| 명령 | Agent Skill | 산출 |
| --- | --- | --- |
| `/speckit.constitution` | `speckit-constitution` | 프로젝트 지배 원칙 (`.specify/memory/constitution.md`) |
| `/speckit.specify` | `speckit-specify` | `specs/<NNN>-<feature>/spec.md` (WHAT/WHY) |
| `/speckit.plan` | `speckit-plan` | `plan.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md` (HOW) |
| `/speckit.tasks` | `speckit-tasks` | `tasks.md` (실행 가능 작업 목록) |
| `/speckit.taskstoissues` | `speckit-taskstoissues` | tasks.md → GitHub Issues |
| `/speckit.implement` | `speckit-implement` | 실제 코드 구현 |
| `/speckit.converge` | `speckit-converge` | 코드베이스를 spec/plan/tasks와 대조해 잔여 작업을 tasks에 추가 |

**선택 명령**

| 명령 | 용도 |
| --- | --- |
| `/speckit.clarify` | 미명세 영역 질의 (구 `/quizme`). `plan` 이전 권장 |
| `/speckit.analyze` | spec/plan/tasks 교차 일관성·커버리지 분석. `tasks` 이후, `implement` 이전 |
| `/speckit.checklist` | 품질 체크리스트 생성 ("영어를 위한 유닛 테스트") |

> `/speckit.converge`와 `/speckit.analyze`는 **브라운필드(기존 코드베이스)** 에서 특히 중요하다. 우리 케이스(고객의 기존 프로덕션 파이프라인에 피처를 주입)는 전형적 브라운필드다.

### 2.3 확장 메커니즘 — 포크하지 않고 커스터마이즈

템플릿 해석은 **런타임 우선순위 스택**으로 이뤄진다.

| 우선순위 | 종류 | 위치 |
| --: | --- | --- |
| ⬆ 1 | Project-Local Overrides | `.specify/templates/overrides/` |
| 2 | Presets (기존 워크플로 커스터마이즈) | `.specify/presets/templates/` |
| 3 | Extensions (새 기능 추가) | `.specify/extensions/templates/` |
| ⬇ 4 | Spec Kit Core | `.specify/templates/` |

- **Preset** = *어떻게* 동작할지 바꾼다. 스펙/플랜/태스크의 **포맷과 용어를 조직 표준으로 강제**. 규제 추적성 요구, test-first 태스크 순서 강제, 필수 보안 리뷰 게이트 추가 등. → **우리의 "ML 승격 스펙 포맷"은 preset이다.**
- **Extension** = *무엇을 할 수 있는지* 늘린다. 새 명령/템플릿 추가, 외부 도구 연동. → **우리의 `/speckit.aidm-import` 같은 신규 명령은 extension이다.**
- **Bundle** = preset + extension + workflow를 **버전 고정된 역할 기반 세트**로 묶어 한 명령으로 프로비저닝. `bundle.yml` 매니페스트로 각 컴포넌트 버전을 pin. → **고객 리포에 배포할 단위는 bundle이다.**

```bash
specify preset add <name>
specify extension add <name>
specify bundle info <id>      # install이 추가할 컴포넌트를 정확히 보여줌 (투명성 보장)
specify bundle install <id>
specify bundle validate --path ./my-bundle
specify bundle build --path ./my-bundle    # 버전 있는 .zip 아티팩트 생성
```

번들의 보장 사항이 우리 요구와 잘 맞는다: `info`가 `install`과 동일한 내용을 보여주고, 설치는 **멱등적이며 프로젝트 루트에 한정**되고, `remove`는 다른 번들이 쓰는 컴포넌트를 건드리지 않으며, **모든 명령이 로컬/핀된 소스에 대해 오프라인 동작**한다. (고객 환경 네트워크 제약 대응 가능)

### 2.4 템플릿 실제 구조 (원문 확인)

**`spec-template.md`** — 섹션:
- `## User Scenarios & Testing *(mandatory)*` — 우선순위(P1/P2/P3)가 붙은 **독립 테스트 가능한** 스토리, `**Given** … **When** … **Then** …` 수용 시나리오
- `### Edge Cases`
- `## Requirements *(mandatory)*` → `### Functional Requirements` (**FR-001** 형식, `System MUST …`), `### Key Entities`
- `## Success Criteria *(mandatory)*` → `### Measurable Outcomes` (**SC-001** 형식, **기술 중립적이고 측정 가능해야 함**)
- `## Assumptions`
- 미확정 항목은 `[NEEDS CLARIFICATION: …]` 마커로 표시

**`plan-template.md`** — 섹션:
- `## Summary`, `## Technical Context` (Language/Version, Primary Dependencies, Storage, Testing, Target Platform, Project Type, Performance Goals, Constraints, Scale/Scope)
- `## Constitution Check` — *"GATE: Must pass before Phase 0 research. Re-check after Phase 1 design."*
- `## Project Structure` (문서 트리 + 소스 트리, `**Structure Decision**`)
- `## Complexity Tracking` — 헌법 위반이 있을 때만 채우는 정당화 표 (`Violation | Why Needed | Simpler Alternative Rejected Because`)

**`tasks-template.md`** — 구조:
- 포맷: `[ID] [P?] [Story] Description` (`[P]` = 병렬 가능, `[Story]` = US1/US2 추적성, **정확한 파일 경로 포함 필수**)
- Phase 1 Setup → Phase 2 Foundational(모든 스토리 차단) → Phase 3+ 스토리별 → Phase N Polish
- `**Checkpoint**:` 마커로 각 스토리 독립 검증 지점 명시
- *"Tests (if included) MUST be written and FAIL before implementation"* — test-first
- `## Dependencies & Execution Order`, `## Parallel Opportunities`, `## Implementation Strategy` (MVP First / Incremental / Parallel Team)

### 2.5 브라운필드 지원

README는 3개 개발 단계를 정의한다: 0-to-1(그린필드), Creative Exploration(병렬 구현 탐색), **Iterative Enhancement(브라운필드 — 기능 반복 추가, 레거시 현대화, 프로세스 적응)**.

기존 프로젝트에 대한 명시적 지침: **"Spec Kit 툴링 업데이트와 피처 아티팩트 진화를 분리하라 — 업그레이드 시 관리 파일을 갱신하고, 의도된 동작이 바뀔 때 `specs/` 아티팩트를 갱신하라."** (`docs/guides/evolving-specs.md`)

→ 우리 케이스에 그대로 적용된다. **AIDM 실행 결과가 바뀔 때만 `specs/`가 갱신되고, Spec Kit 버전 업그레이드는 별도 트랙**으로 관리한다.

---

## 3. 적합성 평가 — 왜 맞고, 어디가 안 맞는가

### 3.1 강하게 맞는 지점 ✅

| 우리 요구 | Spec Kit 대응 |
| --- | --- |
| AIDM 결정을 사람이 리뷰 가능한 계약으로 기록 | `spec.md` (FR/SC/Assumptions/Edge Cases) |
| 고객 리포마다 다른 반영 방식 | **WHAT(spec)은 고정, HOW(plan)만 리포별로 다르게** — SDD 핵심 원리 그대로 |
| 누수 금지·시간순 검증·사람 승인 같은 불변 규칙 | `constitution.md` + plan의 `## Constitution Check` **게이트** |
| 미확정 사항을 조용히 추측하지 않기 | `[NEEDS CLARIFICATION]` 마커 + `/speckit.clarify` |
| 증적 일관성 재검증 | `/speckit.analyze` (spec↔plan↔tasks 교차 검증) |
| 기존 고객 코드베이스에 점진 반영 | `/speckit.converge`, 브라운필드 트랙 |
| 조직 표준 포맷 강제 | **Preset** |
| 도메인 전용 명령 추가 | **Extension** |
| 고객사에 한 번에 배포 | **Bundle** (버전 pin, 멱등, 오프라인) |
| 스킬 기반 에이전트 프레임워크와의 궁합 | `--integration-options="--skills"` → `.agents/skills/`와 동형 |
| 작업 추적성 | `tasks.md`의 `[Story]` 라벨 + `/speckit.taskstoissues` |

### 3.2 맞지 않거나 위험한 지점 ⚠️

| 위험 | 설명 | 완화 |
| --- | --- | --- |
| **LLM이 스펙을 "쓴다"** | `/speckit.specify`는 자연어 → 스펙 생성이다. AIDM 결과 스펙을 LLM이 쓰면 **환각으로 지표·게이트가 왜곡**될 수 있다. | **`spec.md`를 LLM에게 쓰게 하지 않는다.** `promotion_manifest.json` → `spec.md`를 **결정론적 렌더러(Python)** 로 생성하고, 그 spec에 `manifest_sha256`을 프론트매터로 박는다. `/speckit.specify`는 사용하지 않거나, 사람 서술 섹션에만 제한한다. |
| **`/speckit.implement`의 자유도** | 임의 코드 생성은 현재 harness의 "에이전트는 임의 estimator 코드를 쓰지 않는다" 원칙과 충돌 | `implement`는 **harness 리포가 아니라 고객 프로덕션 리포에서만**, 그리고 **사람 승인 이후에만** 허용. harness 리포 안에서는 `plan`/`tasks`/`analyze`까지만. |
| **템플릿이 웹/CRUD 편향** | "User Story", "endpoint", "frontend/backend" 등 | **Preset으로 전면 재작성.** ML 도메인 용어("Feature Requirement", "Evaluation Criteria", "Data Contract")로 치환. README가 명시적으로 지원하는 사용 사례다. |
| **`specs/` 번호 자동 증가 + feature 브랜치** | AIDM run-id와 이중 식별자 | spec 디렉터리명에 run-id를 인코딩: `specs/007-aidm-<run-id-short>/` |
| **Spec Kit 버전 드리프트** | 코어 템플릿이 바뀌면 렌더러 출력과 어긋남 | 릴리스 태그 pin + bundle.yml 버전 pin + CI에서 템플릿 SHA 검증 |
| **`.specify/`가 또 하나의 관리 표면** | `.agents/`, `.specify/` 이중 관리 | `.agents/`는 **실행·게이트**, `.specify/`는 **스펙·반영**으로 책임 분리 명문화 |
| **release-gate 우회 가능성** | `spec.md`나 `tasks.md`가 "승인처럼" 보임 | `release-gate` SKILL.md에 명시 행 추가: *"spec.md / plan.md / tasks.md / implement 성공은 human approval을 대체하지 않는다"* — 기존 `research-summary.json` 거부 규칙과 동일 패턴 |

### 3.3 "AIDM → AIDM"이 아니라 "AIDM → AIDD" 경계

질문의 "aidm에서 aidm으로"는 **AIDM → AIDD → 프로덕션** 경계로 해석했다. 이 경계에서 Spec Kit의 역할을 명확히 하면:

```text
AIDM   = 무엇이 더 좋은가를 실험으로 결정한다   (증거 생산)
spec   = 그 결정을 사람이 읽는 계약으로 고정한다 (의도 고정)   ← Spec Kit
AIDD   = 그 계약을 결정론적으로 코드화·검증한다  (안전 집행)
plan   = 그 계약을 이 리포에서 어떻게 실현할지 정한다 (적응)  ← Spec Kit
tasks/implement = 실제 프로덕션 변경                        ← Spec Kit
```

**AIDD는 없어지지 않는다. Spec Kit이 AIDD의 앞(spec 고정)과 뒤(프로덕션 적응)를 감싼다.**

---

## 4. 제안 아키텍처

### 4.1 전체 흐름

```text
┌───────────────────────── ml-harness (연구·검증 리포) ─────────────────────────┐
│                                                                              │
│  legacy-intake → research-* → aidm-experiment                                │
│                                    │                                         │
│                                    ▼                                         │
│                       promotion_manifest.json  (decision: promote)           │
│                                    │                                         │
│                    ┌───────────────┴────────────────┐                        │
│                    ▼                                ▼                        │
│         [결정론적 렌더러]                     aidd-promotion                  │
│         manifest → spec.md                   generated/promoted_features.py  │
│         (LLM 미사용)                          model-recipe-patch.json         │
│                    │                          promotion-evidence.json        │
│                    ▼                                │                        │
│         specs/<NNN>-aidm-<run>/                     │                        │
│           spec.md            ← WHAT/WHY             │                        │
│           evidence/          ← manifest, report 사본 │                        │
│           contracts/         ← feature-spec.schema  │                        │
│                    │                                │                        │
│                    └──────────┬─────────────────────┘                        │
│                               ▼                                              │
│                    /speckit.analyze  (spec ↔ 증적 일관성)                     │
│                               ▼                                              │
│                    human-review → release-gate  (fail-closed)                │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │  spec bundle (spec.md + evidence + checksums)
                                ▼
┌──────────────────── 고객 프로덕션 리포 (브라운필드) ───────────────────────────┐
│  .specify/ (bundle install)                                                  │
│    memory/constitution.md      ← 고객 정책 + 우리 ML 불변 원칙                 │
│                                                                              │
│  specs/<NNN>-aidm-<run>/spec.md   ← 그대로 이식 (변경 금지, checksum 검증)     │
│         │                                                                    │
│         ▼                                                                    │
│  /speckit.plan     → plan.md, data-model.md, contracts/, quickstart.md       │
│         │            (고객 스택: Airflow / dbt / sklearn Pipeline / …)        │
│         ▼                                                                    │
│  /speckit.tasks    → tasks.md  (test-first, 파일 경로 명시)                   │
│         ▼                                                                    │
│  /speckit.analyze  → 커버리지·일관성 (FR/SC가 태스크로 전부 매핑되었는가)       │
│         ▼                                                                    │
│  /speckit.taskstoissues → GitHub Issues (추적성)                             │
│         ▼                                                                    │
│  /speckit.implement → PR  (사람 리뷰 → merge)                                │
│         ▼                                                                    │
│  /speckit.converge → 배포 후 실제 코드 ↔ spec 재대조                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 책임 분리 원칙

| 표면 | 소유 | 절대 하지 않는 것 |
| --- | --- | --- |
| `.agents/` | 실행 runner, 게이트, checksum, fail-closed | 스펙 포맷을 정의하지 않는다 |
| `.specify/` | 스펙 템플릿, constitution, plan/tasks 워크플로 | 게이트를 집행하지 않는다(참조만) |
| `runs/`, `outputs/` | 실행 증적 (git ignore) | 커밋되지 않는다 |
| `specs/` | **커밋되는 스펙** (사람이 리뷰하는 계약) | 원시 고객 데이터·secret을 담지 않는다 |

---

## 5. 구체 설계

### 5.1 매핑 명세: `promotion_manifest.json` → `spec.md`

렌더러는 순수 함수여야 한다: `render_spec(manifest, run_id) -> str`. 동일 입력 → 바이트 동일 출력(정렬된 키, 고정 소수 자리, 타임스탬프 미포함 또는 manifest 내 값만 사용).

| spec.md 섹션 | 소스 | 규칙 |
| --- | --- | --- |
| Front matter `manifest_sha256` | `sha256(canonical_json(manifest))` | 결속 키. 불일치 시 fail closed |
| Front matter `run_id`, `seed`, `decision` | `winner.run_id`, `seed`, `decision` | `decision != promote`면 렌더 거부 |
| `## Feature Requirements` (FR) | `selected_specs[]` | 스펙당 1개 FR. 아래 포맷 |
| FR rationale | `selected_specs[].rationale` | 원문 그대로, 편집 금지 |
| `## Success Criteria` (SC) | `baseline.metrics`, `winner.metrics`, `improvement_ratio`, `thresholds` | 기술 중립·측정 가능 |
| `## Per-Plant Constraints` | `per_plant_deltas`, `thresholds.max_plant_regression` | 회귀 한계 명시 |
| `## Constitution Evidence` | `failed_gates`, `thresholds` | `failed_gates != []`면 렌더 거부 |
| `## Assumptions` | `seed`, `baseline.model`, dataset 식별자(경로 아님) | 원시 데이터 금지 |
| `## Out of Scope` | 고정 문구 | 앙상블·serving·자동 merge 제외 명시 |
| `## Evidence Index` | 아티팩트 상대경로 + SHA-256 | 고객 경로·secret 금지 |

**FR 렌더 포맷 (제안)**

```markdown
- **FR-001**: 예측 파이프라인은 `hour_sin` 피처를 MUST 제공한다.
  - Transform: `cyclic_hour` (deterministic)
  - Inputs: `timestamp`
  - Parameters: (none)
  - Spec version: `1`
  - Availability: 예측 시점(prediction-time)에 가용해야 하며, 미가용 시 MUST 거부한다.
  - Rationale: Synthetic fixture prediction-time hour signal.
```

**SC 렌더 포맷 (제안)**

```markdown
- **SC-001**: 시간순 fold 평가에서 NMAE가 baseline `SPOT`(0.2000) 대비 winner(0.1500)로
  25.00% 이상 개선된 상태를 MUST 유지한다.
- **SC-002**: 어떤 발전소도 NMAE 회귀가 `max_plant_regression = 1.0`을 MUST 초과하지 않는다.
  (측정된 최대 회귀: `plant-a` = -0.0500, 회귀 아님)
- **SC-003**: 검증 블록은 학습 블록보다 시간상 뒤에 MUST 위치한다.
- **SC-004**: `generation_mw`, `actual_*`, 현재/미래 행의 target 파생값은 예측 피처로
  MUST NOT 사용된다.
```

> `[NEEDS CLARIFICATION]`은 **manifest에 없는 정보를 렌더러가 지어내지 않기 위한 안전장치**로 활용한다. 예: `selected_model_recipe`가 없으면 `## Model Recipe` 섹션에 `[NEEDS CLARIFICATION: 이 승격은 피처 전용이며 모델 레시피 변경을 포함하지 않음]`을 명시적으로 남긴다.

### 5.2 Constitution 초안 (`.specify/memory/constitution.md`)

**협상 불가 조항(non-negotiable articles)** — plan의 `## Constitution Check` 게이트가 이걸 검사한다.

```markdown
# 발전량 예측 승격 헌법 (v1)

## Article I: 스펙 불변성 (Spec Immutability)
AIDM이 생성한 spec.md는 사람이 수정할 수 없다. 내용이 틀렸다면 AIDM을 다시 실행한다.
spec.md의 manifest_sha256은 promotion_manifest.json의 정규 해시와 일치해야 한다.
불일치는 fail closed다.

## Article II: 누수 금지 (No Leakage)
`generation_mw`, `actual_*`, 현재 행과 미래 행의 target 파생값은 어떤 plan·task·구현에서도
예측 피처가 될 수 없다. history transform은 발전소별 strict-prior 값만 사용한다.
plan이 이를 위반하면 Constitution Check는 FAIL이며 Complexity Tracking으로 정당화할 수 없다.

## Article III: 시간순 검증 (Temporal Validation)
모든 검증 블록은 학습 블록보다 시간상 뒤에 위치한다. 무작위 셔플 분할은 금지한다.

## Article IV: 결정론 (Determinism)
승격된 피처 변환은 결정론적이어야 한다. 난수·현재 시각·외부 네트워크 호출을 포함할 수 없다.
동일 입력은 동일 출력을 낸다.

## Article V: 사람 승인 (Human Approval)
spec.md, plan.md, tasks.md, /speckit.analyze 통과, /speckit.implement 성공 중 어느 것도
release approval이 아니다. release는 manifest checksum을 참조하는 명시적 사람 승인을 요구한다.
에이전트는 merge하지 않고 deploy하지 않으며 고객 시스템을 직접 수정하지 않는다.

## Article VI: 데이터 비노출 (No Raw Data)
spec, plan, tasks, research, contracts, issue 어디에도 고객 원시 행, target, actual 값,
secret, credential, 고객 절대경로를 기록하지 않는다. aggregate 지표·이름·checksum만 허용한다.

## Article VII: Test-First
tasks.md의 각 FR은 대응 검증 태스크를 가지며, 검증은 구현 전에 작성되어 실패해야 한다.
피처 가용성 거부(unavailable-input rejection) 테스트는 필수다.

## Article VIII: 범위 한정 (Bounded Scope)
prediction ensemble, model registry, serving, rollout/rollback, 자동 merge·deploy는
이 헌법의 범위 밖이며 별도 계약과 게이트를 요구한다.
```

### 5.3 Preset: `speckit-preset-ml-promotion`

코어 템플릿을 ML 승격 도메인으로 치환한다.

| 코어 | 우리 preset |
| --- | --- |
| `## User Scenarios & Testing` | `## Forecast Scenarios & Validation` — "User Story" 대신 **평가 시나리오**(fold, 발전소 세그먼트, 결측 입력) |
| `### Functional Requirements` (FR) | `### Feature Requirements` (FR) — transform/inputs/parameters/availability 필수 필드 |
| `### Key Entities` | `### Data Contract` — 필수 컬럼, 키(`plant_id`,`timestamp`), 가용성 시점 |
| `## Success Criteria` (SC) | 그대로. 단 **metric 기반 SC만 허용**(NMAE/MAE/RMSE + 게이트 임계값) |
| plan `## Technical Context` | `## Pipeline Context` — orchestrator(Airflow/dbt/…), 피처 스토어, 학습 주기, 서빙 지연, 데이터 신선도 |
| plan `## Project Structure` Option 1/2/3 | `Option A: 배치 파이프라인` / `Option B: 피처 스토어` / `Option C: 인라인 sklearn Pipeline` |
| tasks Phase 2 Foundational | `Phase 2: Data Contract & Availability Guards` (모든 FR 차단) |
| tasks 스토리 라벨 `[US1]` | `[FR-001]` — **FR 단위 추적성** |

이 preset은 README가 명시한 preset 용도("규제 추적성 요구 강제", "test-first 태스크 순서 강제", "도메인 용어")에 정확히 부합한다.

### 5.4 Extension: `speckit-ext-aidm`

새 명령을 추가한다.

| 명령 | 동작 | LLM 사용 |
| --- | --- | --- |
| `/speckit.aidm-import` | `promotion_manifest.json` + `promotion-evidence.json`을 읽어 **결정론적으로** `specs/<NNN>-aidm-<run>/spec.md`와 `evidence/`를 생성 | ❌ 없음 (Python 렌더러 호출) |
| `/speckit.aidm-verify` | spec.md의 `manifest_sha256` ↔ 실제 manifest, evidence 파일 checksum, `failed_gates == []`, `decision == promote` 재검증 | ❌ 없음 |
| `/speckit.aidm-adapt` | 고객 리포에서 spec.md를 읽고 `/speckit.plan` 실행 전 **Pipeline Context 질의**(orchestrator, 피처 가용 시점, 재학습 주기)를 `[NEEDS CLARIFICATION]`으로 수집 | ✅ 질의만 |
| `/speckit.aidm-rollback` | 배포된 FR을 되돌리는 역-tasks 생성 (SC 위반 감지 시) | ✅ 제한적 |

`/speckit.aidm-import`와 `/speckit.aidm-verify`가 **LLM을 쓰지 않는다**는 점이 이 설계의 핵심 안전 속성이다. 스펙의 진실성은 코드가 보장하고, LLM은 "그 스펙을 이 리포에 어떻게 녹일까"(plan/tasks)에만 관여한다.

### 5.5 Bundle: `ml-harness-promotion`

```yaml
# bundle.yml (개념)
id: ml-harness-promotion
version: 1.0.0
# integration 미지정 → agnostic (고객 리포의 기존 에이전트 상속)
components:
  presets:
    - id: ml-promotion
      version: 1.0.0
  extensions:
    - id: aidm
      version: 1.0.0
```

- 고객 리포: `specify bundle install ml-harness-promotion`
- 검증: `specify bundle info ml-harness-promotion` (install이 추가할 것과 정확히 동일)
- 제거: `specify bundle remove ml-harness-promotion` (다른 번들 컴포넌트 미침해)
- 오프라인 동작 → 폐쇄망 고객 대응

이는 현재 `power-forecast init --target <repo>` 설치자와 **같은 계층**이다. 장기적으로 `init`이 `.agents/` 설치 + `specify bundle install`을 함께 수행하도록 통합할 수 있다.

### 5.6 디렉터리 레이아웃

**ml-harness (연구 리포)**

```text
ml-harness/
├── AGENTS.md
├── .agents/                     # 변경 없음 — 실행·게이트
│   ├── skills/
│   │   ├── aidd-promotion/      # (수정) spec 렌더 단계 추가
│   │   ├── release-gate/        # (수정) spec/plan/tasks는 승인 아님 명시
│   │   └── spec-promotion/      # (신규) manifest → spec.md 렌더 스킬
│   └── scripts/
│       └── render-spec.sh       # (신규) 결정론적 렌더러 진입점
├── .specify/                    # (신규) specify init --here
│   ├── memory/constitution.md
│   ├── presets/templates/       # ml-promotion preset
│   ├── extensions/templates/    # aidm extension
│   └── templates/
├── specs/                       # (신규) 커밋되는 스펙
│   └── 001-aidm-a1b2c3d/
│       ├── spec.md              # 기계 생성. 사람 편집 금지
│       ├── contracts/
│       │   └── feature-spec.schema.json
│       └── evidence/
│           ├── promotion_manifest.json
│           ├── promotion-evidence.json
│           ├── performance_report.md
│           └── CHECKSUMS.txt
├── src/power_forecasting/
│   └── specs.py                 # (신규) render_spec(manifest) 순수 함수
├── runs/  outputs/              # 변경 없음 (git ignore)
└── docs/
```

**고객 프로덕션 리포**

```text
customer-prod/
├── .specify/                    # bundle install
│   └── memory/constitution.md   # 고객 정책 + Article I~VIII
├── specs/
│   └── 001-aidm-a1b2c3d/
│       ├── spec.md              # ★ harness에서 그대로 이식 (checksum 검증)
│       ├── evidence/
│       ├── plan.md              # ← 고객 스택 기준으로 여기서 생성
│       ├── research.md
│       ├── data-model.md
│       ├── contracts/
│       ├── quickstart.md
│       ├── tasks.md
│       └── checklists/
└── (고객 실제 파이프라인 코드)
```

### 5.7 명령 시퀀스 (실전)

**harness 쪽 (1회, 승격 확정 후)**

```bash
# 0. 최초 1회
specify init --here --force --non-interactive --integration copilot
specify preset add ml-promotion
specify extension add aidm

# 1. AIDM 승격 확인 (기존)
.agents/scripts/verify-promotion.sh --run-dir runs/<run-id>

# 2. manifest → spec 렌더 (신규, 결정론적)
.agents/scripts/render-spec.sh \
  --run-dir runs/<run-id> \
  --specs-dir specs

# 3. 검증
#    (Copilot CLI: /agents 로 speckit-aidm-verify 선택 또는 프롬프트로 직접 지칭)
/speckit.aidm-verify 001-aidm-a1b2c3d
/speckit.analyze

# 4. 사람 리뷰
#    human-review → release-gate (기존 fail-closed 유지)
```

**고객 프로덕션 리포 쪽 (승인 이후)**

```bash
specify bundle install ml-harness-promotion
# specs/001-aidm-a1b2c3d/ 를 spec bundle에서 복사 후
/speckit.aidm-verify 001-aidm-a1b2c3d          # checksum 재검증
/speckit.clarify                               # 파이프라인 미확정 사항 해소
/speckit.plan Airflow DAG 안의 기존 feature_engineering 태스크에 반영하고,
             피처 스토어는 쓰지 않으며, 예측 시점 가용성은 T-1h 기준으로 검증한다
/speckit.tasks
/speckit.analyze                               # FR/SC 커버리지 확인
/speckit.checklist                             # 승격 품질 체크리스트
/speckit.taskstoissues                         # 추적성
/speckit.implement                             # → PR (자동 merge 금지)
# 배포 후
/speckit.converge                              # 실제 코드 ↔ spec 재대조
```

---

## 6. 증적 결속(checksum chain) 유지 방법

현재 harness의 강점은 SHA-256 결속이다. Spec Kit 도입 시 이 체인이 끊기면 안 된다.

```text
research-config.json ─sha256─┐
optimization-catalog.v1.json ┤
research-proposal.json      ─┤
                             ├─► promotion_manifest.json ─sha256─┐
experiments.db              ─┘                                   │
                                                                 ├─► spec.md (front matter: manifest_sha256)
generated/promoted_features.py ─sha256─► promotion-evidence.json ┘        │
                                                                          ├─► specs/<id>/evidence/CHECKSUMS.txt
                                                                          │
                                                       고객 리포 spec.md ──┘ (동일 해시여야 함)
                                                                          │
                                                                          └─► plan.md / tasks.md (spec_sha256 참조)
```

구현 규칙:

1. `spec.md` front matter에 `manifest_sha256`, `evidence_bundle_sha256`, `spec_kit_version`, `preset_version`을 기록한다.
2. `specs/<id>/evidence/CHECKSUMS.txt`에 모든 증적 파일의 SHA-256을 기록한다.
3. `/speckit.aidm-verify`는 (a) manifest 재해시, (b) evidence 파일 재해시, (c) spec.md 재렌더 후 바이트 비교를 수행한다. **재렌더 결과가 다르면 사람이 spec을 손댄 것이므로 fail closed.**
4. `plan.md`/`tasks.md` front matter에 `spec_sha256`을 기록한다. spec이 바뀌면 plan은 stale로 표시하고 `/speckit.plan` 재실행을 강제한다.
5. `release-gate`는 기존 항목에 더해 **spec.md 재렌더 일치**를 요구한다.

---

## 7. 리스크와 완화

| # | 리스크 | 영향 | 완화 |
| --- | --- | --- | --- |
| R1 | LLM이 spec.md를 지어내거나 수정 | 치명 — 증적 신뢰 붕괴 | spec.md는 결정론적 렌더만. 재렌더 바이트 비교로 검출. Article I |
| R2 | `/speckit.implement`가 고객 프로덕션에 임의 변경 | 치명 | implement 산출은 **항상 PR**. 자동 merge 금지. Article V. release-gate 유지 |
| R3 | spec/plan/tasks가 "승인처럼" 오해됨 | 높음 | `release-gate` SKILL.md에 명시적 거부 행 추가 (기존 `research-summary.json` 패턴 재사용) |
| R4 | Spec Kit 버전 업그레이드로 템플릿 변동 | 중간 | 릴리스 태그 pin, bundle 버전 pin, CI에서 `.specify/templates/` SHA 검증. "툴링 업데이트와 아티팩트 진화 분리" 지침 준수 |
| R5 | 고객 데이터가 spec/plan/issue로 유출 | 치명 | Article VI + 렌더러가 경로·secret 패턴 거부 (`_reject_unsafe_string`, `_contains_sensitive_key` 재사용) |
| R6 | `.agents/`와 `.specify/` 이중 관리 부담 | 중간 | 책임 분리 표(§4.2) 문서화. `power-forecast init`이 둘 다 설치 |
| R7 | 웹 편향 템플릿이 잘못된 구조 유도 | 중간 | preset으로 전면 치환. `/speckit.analyze`로 잔여 웹 용어 검출 |
| R8 | spec 번호와 run-id 이중 식별 | 낮음 | 디렉터리명 `NNN-aidm-<run-short>` 규칙 고정 |
| R9 | 고객 폐쇄망에서 설치 불가 | 중간 | bundle은 오프라인 동작 보장. `specify bundle build`로 .zip 사전 배포 |
| R10 | Copilot CLI가 `/speckit.*` 슬래시를 노출하지 않음 | 낮음 | `--integration-options="--skills"`로 skill 설치 후 `/agents` 또는 프롬프트 직접 지칭 |

---

## 8. 단계적 도입 로드맵

### Phase 0 — 검증 (0.5~1주)

- `specify init --here --force --non-interactive --integration copilot`을 **worktree에서** 실행
- 코어 템플릿과 `.specify/` 레이아웃을 실제로 확인
- `.agents/fixtures/promoted-manifest.json`을 손으로 `spec.md`로 변환해 본다
- **수용 기준**: 기존 pytest 전부 통과, `.agents/` 무변경, `spec.md` 초안이 사람 리뷰 가능

### Phase 1 — 결정론적 렌더러 (1~2주)

- `src/power_forecasting/specs.py`: `render_spec(manifest, run_id) -> str` 순수 함수
- `.agents/scripts/render-spec.sh` 진입점
- fixture 기반 골든 테스트: `promoted-manifest.json` → 고정 `spec.md` 바이트 비교
- 거부 테스트: `rejected-promotion-manifest.json`, `leakage-promotion-manifest.json`,
  `malformed-promotion-manifest.json`, `missing-thresholds-promotion-manifest.json` → 전부 렌더 거부
- **수용 기준**: 동일 manifest → 바이트 동일 spec, 잘못된 manifest는 예외로 fail closed

### Phase 2 — Preset + Extension (2~3주)

- `ml-promotion` preset 작성 (`spec`/`plan`/`tasks` 템플릿 오버라이드)
- `aidm` extension 작성 (`/speckit.aidm-import`, `/speckit.aidm-verify`)
- `.specify/memory/constitution.md` v1 확정 (Article I~VIII)
- `aidd-promotion` SKILL.md에 spec 렌더 단계 추가, `release-gate` SKILL.md에 거부 행 추가
- **수용 기준**: `/speckit.plan`이 Constitution Check에서 누수 위반을 실제로 FAIL 처리

### Phase 3 — 고객 리포 반영 경로 (3~4주)

- `bundle.yml` 작성 → `specify bundle validate/build`
- `CUSTOMER_INTEGRATION_GUIDE.md` §4.3을 spec-driven 절차로 재작성
- 데모 리포 1개에서 end-to-end 실행: manifest → spec → plan → tasks → implement → PR
- `/speckit.converge`로 배포 후 재대조
- **수용 기준**: 사람이 작성한 코드 0줄로 PR이 만들어지고, PR 본문에 manifest checksum과 FR/SC가 추적 가능

### Phase 4 — 운영화 (선택)

- `/speckit.aidm-rollback` (SC 위반 감지 시 역-tasks)
- `/speckit.taskstoissues`로 고객 이슈 트래커 연동
- 다중 승격 spec 간 충돌 감지

---

## 9. 대안 비교

| 대안 | 장점 | 단점 | 평가 |
| --- | --- | --- | --- |
| **A. 현행 유지** (수작업 반영) | 변경 비용 0 | 병목·의도 손실·추적성 단절 지속 | ❌ |
| **B. Spec Kit 전면 대체** (AIDD 폐기, `/speckit.*`만 사용) | 단일 도구 | 결정론·checksum·fail-closed 상실. LLM이 지표를 쓰게 됨 | ❌ 위험 |
| **C. 하이브리드 — AIDM이 spec을 결정론적으로 생성, Spec Kit이 프로덕션 반영** | 증적 무결성 유지 + 반영 자동화 + 리포별 적응 | 두 표면 관리, 초기 구축 4~8주 | ✅ **권장** |
| **D. 자체 스펙 포맷 개발** | 완전 제어 | 생태계(30+ 에이전트, extension/preset/bundle, analyze/converge) 전부 재구현 | ❌ 비효율 |
| **E. Spec Kit을 harness에만, 고객 리포엔 미도입** | 도입 범위 작음 | 병목(§1.3)이 그대로 남음 — 목적 미달성 | ⚠️ Phase 2 중간 상태로만 |

**권고: C.** 핵심 원칙은 **"스펙의 진실성은 코드가 보장하고, 스펙의 적응은 LLM이 담당한다"**.

---

## 10. 미해결 질문 (사람 결정 필요)

1. **spec.md 커밋 위치**: harness 리포의 `specs/`에 커밋할 것인가, 아니면 증적 번들로만 전달할 것인가? (커밋하면 이력이 남지만 고객 도메인 정보가 harness 리포에 남는다 → 합성/익명 식별자만 쓰는 규칙 필요)
2. **spec 번호 체계**: Spec Kit 기본 `NNN` 자동 증가를 쓸 것인가, run-id 기반으로 완전 대체할 것인가?
3. **feature 브랜치**: Spec Kit은 spec당 브랜치를 전제한다. harness에서도 브랜치를 만들 것인가, `specs/`만 만들 것인가?
4. **고객 리포에 Spec Kit 도입 강제 여부**: 고객이 이미 다른 SDD 도구를 쓸 수 있다. spec.md를 "포맷 중립 산출물"로 두고 Spec Kit은 선택 경로로 둘 것인가?
5. **`/speckit.implement` 허용 범위**: 피처 엔지니어링 코드까지만? 모델 레시피 변경(`model-recipe-patch.json`)도 포함?
6. **Spec Kit 버전 pin 정책**: 어느 릴리스 태그를 기준으로 할 것인가, 업그레이드 주기는?
7. **preset/extension 배포 방식**: 공개 커뮤니티 카탈로그에 올릴 것인가, private catalog source로 둘 것인가? (`specify bundle catalog add`)

---

## 11. 부록: 렌더링 예시

`.agents/fixtures/promoted-manifest.json`에 `render_spec`을 적용한 결과 예상치.

```markdown
---
spec_type: aidm-promotion
schema_version: "1"
decision: promote
run_id: winner-fixture
baseline_run_id: baseline-fixture
seed: 7
manifest_sha256: <sha256(canonical_json(manifest))>
evidence_bundle_sha256: <sha256(CHECKSUMS.txt)>
spec_kit_version: vX.Y.Z
preset_version: ml-promotion@1.0.0
generated_by: power_forecasting.specs.render_spec
human_editable: false
---

# 승격 스펙: hour_sin

**상태**: Promoted (AIDM) · 사람 리뷰 대기
**입력**: `runs/<run-id>/promotion_manifest.json`

> 이 문서는 기계 생성되었습니다. 사람이 수정할 수 없습니다.
> 내용이 틀렸다면 AIDM을 다시 실행하십시오. (Constitution Article I)

## Forecast Scenarios & Validation

### 시나리오 1 — 예측 시점 피처 가용 (우선순위: P1)

**Given** 예측 대상 시각의 `timestamp`가 주어지고,
**When** 파이프라인이 `hour_sin`을 계산할 때,
**Then** `cyclic_hour` 변환 결과가 결정론적으로 산출된다.

**독립 검증**: 고정 `timestamp` 입력에 대해 동일 값이 반복 산출되는지 확인.

### 시나리오 2 — 입력 미가용 (우선순위: P1)

**Given** `timestamp`가 결측이거나 예측 시점에 가용하지 않고,
**When** 파이프라인이 `hour_sin`을 계산하려 할 때,
**Then** 파이프라인은 추정값으로 대체하지 않고 MUST 거부한다.

### Edge Cases

- 발전소별 데이터 시작 구간에서 strict-prior 이력이 부족하면 어떻게 되는가?
- DST 전환 시각에서 `cyclic_hour`가 중복/누락 시각을 어떻게 처리하는가?
- `plant_id`/`timestamp` 키가 중복되면 어떻게 되는가?

## Feature Requirements

- **FR-001**: 예측 파이프라인은 `hour_sin` 피처를 MUST 제공한다.
  - Transform: `cyclic_hour` (deterministic)
  - Inputs: `timestamp`
  - Parameters: (none)
  - Spec version: `1`
  - Availability: 예측 시점에 가용해야 하며, 미가용 시 MUST 거부한다.
  - Rationale: Synthetic fixture prediction-time hour signal.

## Data Contract

- 키: `plant_id`, `timestamp` (고유)
- 필수 입력 컬럼: `timestamp`
- 금지 입력: `generation_mw`, `actual_*`, 현재·미래 행의 target 파생값

## Success Criteria

- **SC-001**: 시간순 fold 평가에서 NMAE가 baseline `SPOT`(0.2000) 대비
  winner(0.1500)로 25.00% 이상 개선된 상태를 MUST 유지한다.
  (임계값: `minimum_improvement = 0.0`)
- **SC-002**: 어떤 발전소도 NMAE 회귀가 `max_plant_regression = 1.0`을
  MUST NOT 초과한다. (측정: `plant-a` = -0.0500 → 개선)
- **SC-003**: 검증 블록은 학습 블록보다 시간상 뒤에 MUST 위치한다.
- **SC-004**: `seed = 7` 하에서 재실행 시 동일 winner가 재현되어야 한다.

## Constitution Evidence

| 게이트 | 결과 |
| --- | --- |
| `failed_gates` | `[]` (없음) |
| `minimum_improvement` | 0.0 (측정 0.25 → PASS) |
| `max_plant_regression` | 1.0 (측정 최대 -0.05 → PASS) |
| Leakage 검사 | PASS |
| 시간순 fold | PASS |

## Model Recipe

[NEEDS CLARIFICATION: 이 승격은 피처 전용이며 `selected_model_recipe`를 포함하지 않는다.
모델 레시피 변경이 필요하면 별도 AIDM 실행과 별도 spec이 필요하다.]

## Assumptions

- Baseline 모델은 `SPOT`이다.
- 평가는 시간순 fold로 수행되었다.
- `seed = 7`.
- 데이터셋은 fixture이며 고객 원시 데이터를 포함하지 않는다.

## Out of Scope

- prediction ensemble (weighted blending, stacking)
- model registry, serving, rollout, rollback
- 사람 승인 없는 자동 merge·deploy

## Evidence Index

| 아티팩트 | SHA-256 |
| --- | --- |
| `evidence/promotion_manifest.json` | `<sha256>` |
| `evidence/promotion-evidence.json` | `<sha256>` |
| `evidence/performance_report.md` | `<sha256>` |
| `evidence/generated/promoted_features.py` | `<sha256>` |

> 이 스펙은 release approval이 아니다. release는 위 `manifest_sha256`을 참조하는
> 명시적 사람 승인을 요구한다. (Constitution Article V)
```

---

## 12. 참고 자료

- Spec Kit 저장소: <https://github.com/github/spec-kit>
- README (원문 확인): <https://raw.githubusercontent.com/github/spec-kit/main/README.md>
- `spec-template.md`: <https://raw.githubusercontent.com/github/spec-kit/main/templates/spec-template.md>
- `plan-template.md`: <https://raw.githubusercontent.com/github/spec-kit/main/templates/plan-template.md>
- `tasks-template.md`: <https://raw.githubusercontent.com/github/spec-kit/main/templates/tasks-template.md>
- 문서 사이트: <https://github.github.io/spec-kit/>
- 통합 에이전트 목록: <https://github.github.io/spec-kit/reference/integrations.html>
- Extensions / Presets / Bundles 레퍼런스: <https://github.github.io/spec-kit/reference/extensions.html>, <https://github.github.io/spec-kit/reference/presets.html>, <https://github.github.io/spec-kit/community/bundles.html>
- 브라운필드 spec 진화 가이드: <https://github.com/github/spec-kit/blob/main/docs/guides/evolving-specs.md>
- SDD 상세 문서: <https://github.com/github/spec-kit/blob/main/spec-driven.md>
- 저장소 내부: `README.md`, `AGENTS.md`, `CUSTOMER_INTEGRATION_GUIDE.md`,
  `.agents/skills/aidd-promotion/SKILL.md`, `.agents/skills/release-gate/SKILL.md`,
  `src/power_forecasting/aidd.py`, `.agents/fixtures/promoted-manifest.json`
