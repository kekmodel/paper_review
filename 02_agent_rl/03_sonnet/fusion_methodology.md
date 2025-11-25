# DreamGym-AE: AgentEvolver 기반 자율 데이터 수집을 통한 DreamGym 확장

## Executive Summary

본 방법론은 **검증된 DreamGym 프레임워크를 기본 틀로 하되, AgentEvolver의 자기진화 메커니즘을 통해 오프라인 데이터 의존성을 해결**하는 하이브리드 접근법입니다.

### 핵심 설계 원칙
1. **DreamGym의 검증된 이론과 실험 결과 보존**: Theorem 1의 정책 개선 보장, 경험 합성 프레임워크 유지
2. **AgentEvolver로 Cold-Start 문제 해결**: 오프라인 데이터 없이도 자율적으로 초기 경험 수집
3. **최소 침습적 통합**: DreamGym의 핵심 아키텍처 변경 없이 전처리 단계로 통합

### 기대 효과
- **새로운 도메인 적응 비용 감소**: 오프라인 데이터 수집 및 정제 비용 제거
- **데이터 품질 향상**: 자율적 탐색으로 더 다양하고 정보가 풍부한 초기 경험 확보
- **확장성 개선**: 새로운 환경에 빠르게 적응 가능

---

## 1. 문제 정의: DreamGym의 Offline Data 의존성

### 1.1 DreamGym의 초기화 요구사항

DreamGym 프레임워크는 **Phase 1 초기화**에서 다음을 필요로 합니다:

```python
# DreamGym Phase 1: 초기화
1. 시드 태스크 명령어 세트 τ_seeds
2. 오프라인 궤적 데이터로 경험 리플레이 버퍼 D 초기화
3. 오프라인 데이터로 추론 경험 모델 M_exp 학습
```

#### 핵심 수식 (paper2_detailed_analysis.md:190)

$$
(s_{t+1}, r_{t+1}) = M_{\text{exp}}^{R_t} \left( \{(s_i, a_i)\}_{i=0}^t, \{d_j\}_{j=1}^k, \tau \right)
$$

여기서:
- **$\{d_j\}_{j=1}^k$**: 리플레이 버퍼에서 검색한 오프라인 demonstration trajectories
- 경험 모델이 유사한 과거 경험을 참조하여 다음 상태 예측

### 1.2 Offline Data 의존성의 한계

#### 문제점 1: 초기 데이터 수집 비용
- 새로운 도메인마다 전문가 또는 기존 에이전트가 오프라인 궤적 수집 필요
- WebArena 예시: 공개 리더보드 데이터 필요 (paper2:214)
- 데이터 품질 검증에 인간 노력 소요

#### 문제점 2: 도메인 적응성 제한
- 오프라인 데이터가 없는 새로운 환경에서는 DreamGym 적용 불가
- 각 새로운 벤치마크마다 데이터셋 구축 필요

#### 문제점 3: 데이터 품질 의존성
- 오프라인 데이터의 품질이 경험 모델 성능에 직접 영향
- Figure 5 (paper2:738-767): 2K vs 40K 단계의 성능 차이 존재
  - Llama-3.1-8B (WebArena): 2K = ~8%, 10K = ~12%, 40K = ~13%

#### DreamGym의 데이터 효율성은 우수하나...
> "경험 모델은 매우 데이터 효율적: 매우 제한된 오프라인 샘플 (2K-10K)로도 경쟁력 있는 성능"

**하지만 여전히 초기 2K-10K 샘플은 필요**하며, 이는 새로운 환경에서 병목이 됩니다.

### 1.3 AgentEvolver가 해결할 수 있는 부분

AgentEvolver의 3가지 자기진화 메커니즘 (paper1_detailed_analysis.md:24-81):

#### 1) Self-Questioning (자기 질문)
**역할**: 오프라인 데이터 없이도 학습할 태스크 자율 생성
- 환경 상태 분석 → 관심 영역 식별 → 학습 목표 자동 생성
- 수동 데이터셋 구축 비용 제거
- **DreamGym 적용**: 시드 태스크 세트 $\tau_{\text{seeds}}$ 자율 생성

#### 2) Self-Navigating (자기 항해)
**역할**: 경험 재사용을 통한 효율적 초기 탐색
- 하이브리드 정책 가이던스: 탐색(exploration)과 활용(exploitation) 균형
- 무작위 탐색 대비 샘플 효율성 향상
- **DreamGym 적용**: 적은 시도로 고품질 초기 경험 수집

#### 3) Self-Attributing (자기 귀속)
**역할**: 수집된 경험의 품질 평가 및 선별
- 궤적의 각 상태-행동이 최종 결과에 기여한 정도 분석
- 기여도 기반 차등 보상으로 핵심 경험 식별
- **DreamGym 적용**: 리플레이 버퍼 초기화 시 고품질 샘플만 선별

---

## 2. 제안 방법론: DreamGym-AE 아키텍처

### 2.1 전체 프레임워크 개요

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 0: AgentEvolver 기반 자율 데이터 수집 (새로운 단계)        │
├─────────────────────────────────────────────────────────────────┤
│  1. Self-Questioning → 초기 태스크 생성                          │
│  2. Self-Navigating → 효율적 탐색으로 경험 수집                  │
│  3. Self-Attributing → 고품질 경험 선별                         │
│  4. 초기 리플레이 버퍼 D_0 구축                                  │
│  5. 경험 모델 M_exp 초기 학습                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1-N: DreamGym 프레임워크 (검증된 구조 그대로 유지)        │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: 초기화 (이미 완료됨)                                  │
│    - 시드 태스크: Self-Questioning이 생성                       │
│    - 리플레이 버퍼: D_0에서 초기화                              │
│    - 경험 모델: Phase 0에서 학습됨                              │
│                                                                  │
│  Phase 2: 합성 경험 생성 (기존 DreamGym)                        │
│    - 정책 π_θ가 행동 선택                                       │
│    - M_exp가 다음 상태 예측                                     │
│    - 커리큘럼 학습으로 태스크 증강                              │
│                                                                  │
│  Phase 3: RL 최적화 (기존 DreamGym)                             │
│    - PPO/GRPO로 정책 업데이트                                   │
│    - Theorem 1의 정책 개선 보장 유지                            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 핵심 설계 결정

#### 결정 1: DreamGym을 기본 틀로 선택한 이유 ✅
1. **검증된 이론적 기반**:
   - Theorem 1: 합성 경험을 통한 실제 정책 개선 보장
   - 수학적으로 증명된 $\epsilon_R$, $\epsilon_P$ 오차 분석

2. **우수한 실험 결과**:
   - WebArena: +82-108% 성능 향상 (Table 1, paper2:527-577)
   - WebShop/ALFWorld: 순수 합성으로 Traditional RL과 동등한 성능
   - DreamGym-S2R: 실제 데이터 10% 미만으로 40% 개선

3. **확장 가능한 프레임워크**:
   - 경험 합성 스케일링: 대규모 롤아웃 생성 가능
   - 커리큘럼 학습: 보상 엔트로피 기반 적응형 태스크 증강
   - 통합 아키텍처: Experience Model + RL 최적화

#### 결정 2: AgentEvolver를 전처리 단계로 통합 ✅
1. **최소 침습적 통합**:
   - DreamGym의 핵심 알고리즘 변경 없음
   - Theorem 1의 정책 개선 보장 영향 없음
   - Phase 0로 분리하여 모듈성 유지

2. **명확한 역할 분담**:
   - **AgentEvolver**: Cold-start 문제 해결 (Phase 0)
   - **DreamGym**: 검증된 경험 합성 및 RL 학습 (Phase 1-N)

3. **점진적 적용 가능**:
   - 오프라인 데이터가 있는 경우: Phase 0 스킵 가능
   - 새로운 도메인: Phase 0부터 시작
   - 하이브리드: Phase 0 + 기존 데이터 결합

---

## 3. Phase 0: AgentEvolver 기반 자율 데이터 수집

### 3.1 알고리즘: Bootstrap Data Collection

```python
Algorithm 1: AgentEvolver Bootstrap for DreamGym
────────────────────────────────────────────────────────────────
Input:
  - Environment E (new domain without offline data)
  - Base LLM policy π_base (e.g., Llama-3.1-8B-Instruct)
  - Target buffer size N_target (e.g., 2K-10K steps)
  - Quality threshold θ_quality

Output:
  - Initial replay buffer D_0
  - Initial task set τ_seeds
  - Trained experience model M_exp

────────────────────────────────────────────────────────────────
Phase 0.1: Self-Questioning - 초기 태스크 생성
────────────────────────────────────────────────────────────────
1: τ_seeds ← ∅
2: Observe initial environment state s_init from E
3: for i = 1 to N_tasks do:
4:   # LLM이 환경을 분석하여 학습 목표 생성
5:   goal_i ← SelfQuestioningLLM(s_init, E.description, τ_seeds)
6:   # 생성된 목표의 실행 가능성 검증
7:   if is_feasible(goal_i, E):
8:     τ_seeds.add(goal_i)
9:   end if
10: end for

────────────────────────────────────────────────────────────────
Phase 0.2: Self-Navigating - 효율적 초기 경험 수집
────────────────────────────────────────────────────────────────
11: D_raw ← ∅  # 원시 경험 풀
12: E_memory ← ∅  # 경험 메모리 (ReMe 시스템)
13:
14: for τ in τ_seeds do:
15:   for episode = 1 to K do:
16:     s_0 ← E.reset(τ)
17:     trajectory ← []
18:
19:     for t = 0 to T_max do:
20:       # 유사한 과거 경험 검색
21:       similar_exp ← E_memory.search(s_t, τ)
22:
23:       # 하이브리드 정책: 탐색 + 과거 경험 활용
24:       if similar_exp is not None and random() > ε_explore:
25:         a_t ← adapt_action(similar_exp.action, s_t)
26:       else:
27:         a_t ← π_base(s_t, τ)  # Base LLM policy
28:       end if
29:
30:       # 환경에서 실제 실행
31:       s_{t+1}, r_t ← E.step(a_t)
32:       trajectory.append((s_t, a_t, r_t, s_{t+1}))
33:
34:       if terminal(s_{t+1}) or t == T_max:
35:         break
36:       end if
37:       s_t ← s_{t+1}
38:     end for
39:
40:     # 경험 메모리에 저장 (Self-Navigating)
41:     E_memory.store(trajectory, τ, return(trajectory))
42:     D_raw.add(trajectory)
43:   end for
44: end for

────────────────────────────────────────────────────────────────
Phase 0.3: Self-Attributing - 고품질 경험 선별
────────────────────────────────────────────────────────────────
45: D_0 ← ∅  # 선별된 고품질 리플레이 버퍼
46:
47: for trajectory in D_raw do:
48:   # 각 상태-행동의 최종 결과 기여도 분석
49:   contributions ← []
50:   final_return ← sum(r for (s, a, r, s') in trajectory)
51:
52:   for t, (s_t, a_t, r_t, s_{t+1}) in enumerate(trajectory):
53:     # 결과 역추적: 이 행동이 성공에 얼마나 기여했나?
54:     contribution_t ← AttributionLLM(
55:       trajectory[:t+1],
56:       final_return,
57:       τ
58:     )
59:     contributions.append(contribution_t)
60:   end for
61:
62:   # 평균 기여도가 임계값 이상인 궤적만 선별
63:   avg_contribution ← mean(contributions)
64:   if avg_contribution > θ_quality:
65:     # 기여도 기반 가중치 부여
66:     D_0.add(trajectory, weight=avg_contribution)
67:   end if
68: end for

────────────────────────────────────────────────────────────────
Phase 0.4: 경험 모델 초기 학습
────────────────────────────────────────────────────────────────
69: # D_0를 사용하여 DreamGym의 추론 경험 모델 학습
70: M_exp ← TrainExperienceModel(
71:   data=D_0,
72:   base_model="Llama-3.1-8B-Instruct",
73:   training_objective=SFT_loss  # DreamGym과 동일
74: )
75:
76: # DreamGym 수식 (paper2:190) 지원 검증
77: # (s_{t+1}, r_{t+1}) = M_exp^{R_t}({(s_i, a_i)}, {d_j}, τ)
78: assert M_exp.supports_reasoning_trace()
79:
80: return D_0, τ_seeds, M_exp
```

### 3.2 SelfQuestioningLLM 상세 구현

#### 프롬프트 템플릿

```python
def SelfQuestioningLLM(s_init, env_description, existing_tasks):
    prompt = f"""
You are an autonomous agent exploring a new environment to generate
learning goals for reinforcement learning.

## Environment Description
{env_description}

## Current State
{s_init}

## Existing Tasks (to avoid duplication)
{existing_tasks}

## Your Task
Analyze the environment and propose a NEW learning goal that:
1. Is feasible within the environment's capabilities
2. Covers different aspects than existing tasks
3. Has clear success criteria
4. Ranges from simple to moderately challenging

## Output Format (JSON)
{{
  "goal": "Natural language description of the task",
  "success_criteria": "How to determine if the task is completed",
  "estimated_difficulty": "easy|medium|hard",
  "rationale": "Why this goal is valuable for learning"
}}

Generate a NEW goal:
"""

    response = LLM(prompt)
    goal = parse_json(response)

    # 검증: DreamGym의 태스크 형식과 호환되는지 확인
    if validate_dreamgym_task_format(goal):
        return goal
    else:
        return None  # 재시도 필요
```

#### Self-Questioning의 커리큘럼 전략

```python
class CurriculumSelfQuestioning:
    """
    점진적 난이도 증가를 통한 태스크 생성
    DreamGym의 보상 엔트로피와 유사한 개념
    """
    def __init__(self):
        self.difficulty_stages = ["easy", "medium", "hard"]
        self.current_stage = 0
        self.success_rate_threshold = 0.7

    def generate_next_task(self, success_history):
        # 현재 난이도에서 성공률이 높으면 다음 단계로
        recent_success_rate = np.mean(success_history[-10:])
        if recent_success_rate > self.success_rate_threshold:
            self.current_stage = min(
                self.current_stage + 1,
                len(self.difficulty_stages) - 1
            )

        target_difficulty = self.difficulty_stages[self.current_stage]

        # 해당 난이도의 태스크 생성 요청
        goal = SelfQuestioningLLM(
            s_init=self.current_state,
            env_description=self.env.description,
            existing_tasks=self.completed_tasks,
            target_difficulty=target_difficulty  # 추가 제약
        )

        return goal
```

### 3.3 Self-Navigating 경험 메모리 구현

#### ReMe 시스템 통합

AgentEvolver의 Self-Navigating은 ReMe 시스템을 활용 (paper1:62):
- GitHub: https://github.com/agentscope-ai/ReMe

```python
class ExperienceMemory:
    """
    Self-Navigating을 위한 경험 재사용 메모리
    유사한 상황에서 과거 성공 패턴 활용
    """
    def __init__(self, embedding_model):
        self.memory_store = []  # (state, action, outcome, task)
        self.embedding_model = embedding_model
        self.index = FaissIndex(dimension=768)  # 벡터 검색

    def store(self, trajectory, task, final_return):
        """성공한 궤적만 저장하여 질 높은 가이던스 제공"""
        if final_return > 0:  # 성공한 경험만
            for (s, a, r, s_next) in trajectory:
                # 상태-태스크 쌍을 임베딩
                state_task_emb = self.embedding_model.encode(
                    f"State: {s}\nTask: {task}"
                )

                self.memory_store.append({
                    "state": s,
                    "action": a,
                    "task": task,
                    "outcome": r,
                    "embedding": state_task_emb
                })

                self.index.add(state_task_emb)

    def search(self, current_state, current_task, k=5):
        """현재 상황과 유사한 과거 경험 검색"""
        query_emb = self.embedding_model.encode(
            f"State: {current_state}\nTask: {current_task}"
        )

        # Top-k 유사 경험 검색
        distances, indices = self.index.search(query_emb, k)

        similar_experiences = [
            self.memory_store[idx] for idx in indices[0]
        ]

        # 가장 유사한 경험의 행동 반환
        if len(similar_experiences) > 0:
            return similar_experiences[0]
        else:
            return None
```

### 3.4 Self-Attributing 기여도 분석

#### 기여도 평가 LLM 프롬프트

```python
def AttributionLLM(trajectory_prefix, final_return, task):
    """
    각 행동이 최종 결과에 얼마나 기여했는지 평가
    DreamGym의 경험 품질 판정과 유사한 개념
    """
    prompt = f"""
You are an expert in analyzing agent trajectories to identify
critical actions that contribute to task success or failure.

## Task
{task}

## Trajectory (up to current step)
{format_trajectory(trajectory_prefix)}

## Final Outcome
{"Success (reward=1)" if final_return > 0 else "Failure (reward=0)"}

## Your Task
Analyze the LAST action in the trajectory and rate its contribution
to the final outcome:

1. **Positive Contribution** (0.8-1.0):
   - Action directly led to success
   - Critical step toward goal

2. **Neutral Contribution** (0.4-0.7):
   - Action maintained progress
   - Necessary but not critical

3. **Negative Contribution** (0.0-0.3):
   - Action hindered progress
   - Led away from goal

4. **Irrelevant** (-1.0):
   - Action had no impact on outcome
   - Random or exploratory move

## Output Format (JSON)
{{
  "contribution_score": <float in [-1.0, 1.0]>,
  "rationale": "Brief explanation of the action's impact",
  "criticality": "critical|helpful|neutral|harmful"
}}

Analyze:
"""

    response = LLM(prompt)
    result = parse_json(response)

    return result["contribution_score"]
```

#### 기여도 기반 샘플 필터링

```python
def filter_high_quality_trajectories(D_raw, θ_quality=0.6):
    """
    Self-Attributing 기반 고품질 궤적 선별

    목표:
    - DreamGym 경험 모델 학습에 사용할 초기 데이터 품질 최대화
    - 노이즈 궤적 제거 (무작위 탐색, 실패 패턴)
    - 정보가 풍부한 경험 우선 선별
    """
    D_filtered = []

    for traj in D_raw:
        contributions = []
        final_return = sum(r for (s, a, r, s_next) in traj)

        for t, (s_t, a_t, r_t, s_next) in enumerate(traj):
            contribution = AttributionLLM(
                traj[:t+1],
                final_return,
                traj.task
            )
            contributions.append(contribution)

        avg_contribution = np.mean(contributions)

        # 고품질 궤적만 선별
        if avg_contribution > θ_quality:
            D_filtered.append({
                "trajectory": traj,
                "quality_score": avg_contribution,
                "contribution_profile": contributions
            })

    # 품질 점수 기준 정렬
    D_filtered.sort(key=lambda x: x["quality_score"], reverse=True)

    return D_filtered
```

### 3.5 Phase 0 완료 조건

Phase 0는 다음 조건을 만족하면 종료하고 DreamGym Phase 1로 전환:

#### 조건 1: 충분한 데이터 수집
```python
# DreamGym은 2K-10K 샘플로 경쟁력 있는 성능 (paper2:764-766)
N_steps_collected = sum(len(traj) for traj in D_0)
assert N_steps_collected >= 2000  # 최소 2K 스텝
```

#### 조건 2: 태스크 다양성
```python
# 다양한 태스크 커버리지 확인
unique_tasks = set(traj.task for traj in D_0)
assert len(unique_tasks) >= 10  # 최소 10개 이상의 다양한 태스크
```

#### 조건 3: 경험 모델 품질 검증
```python
# DreamGym의 경험 모델 품질 판정 기준 적용 (paper2:682-728)
# Judge: GPT-4o
quality_metrics = evaluate_experience_model(M_exp, D_0)

assert quality_metrics["accuracy"] >= 1.5  # {0, 1, 2} 스케일
assert quality_metrics["informativeness"] >= 1.5
assert quality_metrics["diversity"] >= 1.5
```

---

## 4. Phase 1-N: DreamGym 프레임워크 (검증된 구조 유지)

Phase 0가 완료되면, **기존 DreamGym 프레임워크를 그대로 사용**합니다.

### 4.1 DreamGym 통합 지점

#### 초기화 상태 (Phase 0 → Phase 1 전환)

```python
# DreamGym Phase 1: 초기화 (paper2:908-911)
# ✅ Phase 0에서 이미 준비됨
dreamgym_config = {
    "task_seeds": τ_seeds,           # ← Self-Questioning이 생성
    "replay_buffer": D_0,            # ← Self-Attributing이 선별
    "experience_model": M_exp,       # ← Phase 0.4에서 학습
}

# DreamGym 시작
dreamgym_agent = DreamGym(**dreamgym_config)
dreamgym_agent.train(
    environment=E,
    rl_algorithm="GRPO",  # or PPO
    num_iterations=100
)
```

### 4.2 DreamGym 핵심 구성요소 (변경 없음)

#### 4.2.1 추론 경험 모델 (paper2:152-215)

**수식 (그대로 유지)**:
$$
(s_{t+1}, r_{t+1}) = M_{\text{exp}}^{R_t} \left( \{(s_i, a_i)\}_{i=0}^t, \{d_j\}_{j=1}^k, \tau \right)
$$

- $\{d_j\}_{j=1}^k$: Phase 0의 $D_0$에서 검색
- $R_t$: 추론 트레이스 (CoT reasoning)

**Phase 0의 영향**:
- AgentEvolver가 수집한 고품질 데이터로 $M_{\text{exp}}$가 학습됨
- Self-Attributing으로 정보가 풍부한 $\{d_j\}$ 확보
- DreamGym의 $\epsilon_R$, $\epsilon_P$ 오차 감소 가능성

#### 4.2.2 커리큘럼 학습 (paper2:246-283)

**보상 엔트로피 기반 태스크 증강 (그대로 유지)**:

$$
V_\tau = \frac{1}{n} \sum_{i=1}^n (r_i - \bar{r})^2
$$

- 높은 보상 분산 = 도전적이고 정보가 풍부한 태스크
- 경험 모델이 자동으로 새로운 태스크 변형 생성

**Phase 0의 영향**:
- Self-Questioning이 생성한 $\tau_{\text{seeds}}$가 다양한 출발점 제공
- 커리큘럼 학습이 더 넓은 태스크 공간 탐색 가능

#### 4.2.3 RL 최적화 (paper2:108-148)

**GRPO/PPO (그대로 유지)**:
- 합성 경험으로 정책 업데이트
- Theorem 1의 정책 개선 보장 유지

**Theorem 1 (paper2:347-380) - 영향 없음**:

$$
J_{\mathcal{M}}(\pi') - J_{\mathcal{M}}(\pi) \ge \text{합성 대리 이득} - \text{신뢰 영역 페널티} - \text{경험 모델 오차}
$$

- Phase 0는 경험 모델 오차 ($\epsilon_R$, $\epsilon_P$)를 줄이는데 기여
- 정책 개선 보장은 그대로 유지

### 4.3 DreamGym 워크플로우 (Phase 1-N)

```python
Algorithm 2: DreamGym Training Loop (Original)
────────────────────────────────────────────────────────────────
Input:
  - τ_seeds (from Phase 0)
  - D_0 (from Phase 0)
  - M_exp (from Phase 0)
  - Policy π_θ (초기화: Base LLM)
  - RL algorithm (GRPO or PPO)

────────────────────────────────────────────────────────────────
for iteration = 1 to N_iterations do:

  # Phase 2: 합성 경험 생성
  for τ in τ_curriculum do:  # 커리큘럼 태스크
    for rollout = 1 to K do:
      s_0 ← sample_initial_state(M_exp, τ)
      trajectory ← []

      for t = 0 to T_max do:
        # 정책이 행동 선택
        a_t ← π_θ(s_t, τ)

        # 경험 모델이 다음 상태 예측
        similar_exp ← D_0.search(s_t, a_t, τ)  # ← Phase 0 데이터
        R_t ← M_exp.generate_reasoning(s_t, a_t, τ)
        s_{t+1}, r_t ← M_exp.predict(s_t, a_t, similar_exp, R_t)

        trajectory.append((s_t, a_t, r_t, s_{t+1}))

        if terminal(s_{t+1}):
          break
        s_t ← s_{t+1}
      end for

      D_synthetic.add(trajectory)
    end for
  end for

  # Phase 3: RL 최적화
  π_θ ← RL_update(π_θ, D_synthetic)  # GRPO/PPO

  # 커리큘럼 업데이트: 보상 엔트로피 기반
  τ_curriculum ← M_exp.augment_tasks(
    τ_curriculum,
    reward_entropy_threshold=V_high
  )

  # Optional: Real environment validation (DreamGym-S2R)
  if iteration % 10 == 0:
    D_real ← collect_real_rollouts(π_θ, E, budget=500)
    D_0 ← D_0 ∪ D_real  # 리플레이 버퍼 업데이트
    M_exp ← finetune(M_exp, D_real)  # 경험 모델 정제
  end if
end for

return π_θ
────────────────────────────────────────────────────────────────
```

---

## 5. 이론적 분석: Phase 0가 DreamGym에 미치는 영향

### 5.1 Theorem 1 재검토

DreamGym의 Theorem 1 (paper2:368-369):

$$
J_{\mathcal{M}}(\pi') - J_{\mathcal{M}}(\pi) \ge \underbrace{\text{합성 대리 이득}}_{\text{Term A}} - \underbrace{\frac{4\gamma}{(1-\gamma)^2} V_{\max} \delta}_{\text{Term B}} - \underbrace{2\left(\frac{\epsilon_R}{1-\gamma} + \frac{2\gamma R_{\max}}{(1-\gamma)^2} \epsilon_P\right)}_{\text{Term C: 경험 모델 오차}}
$$

#### Term C 분석: 경험 모델 오차

**$\epsilon_R$ (피드백 충실성)**:
$$
\epsilon_R := \sup_{s,a} |R(s,a) - \hat{R}(s,a)|
$$

**$\epsilon_P$ (도메인 일관성)**:
$$
\epsilon_P := \sup_{s,a} \| P(\cdot | s,a) - \hat{P}(\cdot | s,a) \|_1
$$

### 5.2 Phase 0가 오차를 줄이는 메커니즘

#### 메커니즘 1: Self-Attributing → $\epsilon_R$ 감소

**문제**: 낮은 품질의 오프라인 데이터로 학습된 경험 모델은 부정확한 보상 예측

**Self-Attributing의 해결책**:
```python
# 기여도가 높은 경험만 선별 → 보상 신호 신뢰성 향상
for traj in D_raw:
    if avg_contribution(traj) > θ_quality:
        D_0.add(traj)  # 정확한 보상 신호를 가진 궤적만
```

**결과**:
- 경험 모델이 정확한 보상 패턴 학습
- $\epsilon_R \downarrow$ → Theorem 1의 오차 항 감소

#### 메커니즘 2: Self-Navigating → $\epsilon_P$ 감소

**문제**: 무작위 탐색으로 수집된 데이터는 도메인 분포를 제대로 커버하지 못함

**Self-Navigating의 해결책**:
```python
# 과거 성공 경험 재사용 → 도메인 일관된 상태 전이 수집
similar_exp ← E_memory.search(s_t, τ)
if similar_exp is not None:
    a_t ← adapt_action(similar_exp.action, s_t)
    # → 도메인 동역학과 일치하는 전이 수집 확률 증가
```

**결과**:
- 경험 모델이 실제 환경 동역학과 일치하는 전이 학습
- $\epsilon_P \downarrow$ → Theorem 1의 오차 항 감소

#### 메커니즘 3: Self-Questioning → 다양성 증가

**문제**: 제한된 태스크 세트로는 정책의 일반화 능력 제한

**Self-Questioning의 해결책**:
```python
# 환경 분석 기반 다양한 태스크 생성
τ_seeds ← SelfQuestioning(E, coverage_target="comprehensive")
# → DreamGym의 커리큘럼 학습이 더 넓은 공간 탐색
```

**결과**:
- 정책이 더 다양한 상황에 노출
- Term A (합성 대리 이득) 증가

### 5.3 오차 감소의 정량적 추정

#### 가정

- **Baseline DreamGym**: 공개 리더보드 데이터 (10K 샘플, 품질 불균일)
- **DreamGym-AE**: Phase 0로 수집한 데이터 (10K 샘플, Self-Attributing 필터링)

#### 추정 (보수적)

| 오차 항 | Baseline | DreamGym-AE | 개선 | 근거 |
|---------|----------|-------------|------|------|
| $\epsilon_R$ | 0.15 | **0.10** | ▼33% | Self-Attributing 품질 필터링 |
| $\epsilon_P$ | 0.20 | **0.15** | ▼25% | Self-Navigating 도메인 일관성 |

#### Theorem 1에 미치는 영향

가정: $\gamma = 0.99$, $R_{\max} = 1$

**Baseline 오차 항**:
$$
2\left(\frac{0.15}{1-0.99} + \frac{2 \times 0.99 \times 1}{(1-0.99)^2} \times 0.20\right) = 2(15 + 396) = 822
$$

**DreamGym-AE 오차 항**:
$$
2\left(\frac{0.10}{0.01} + \frac{1.98}{0.0001} \times 0.15\right) = 2(10 + 297) = 614
$$

**개선**:
$$
\frac{822 - 614}{822} \times 100\% = \mathbf{25.3\%} \text{ 오차 감소}
$$

**의미**: Theorem 1의 정책 개선 하한이 25% 상승 → 더 안정적인 학습

---

## 6. 실험 설계 및 예상 결과

### 6.1 평가 벤치마크

#### 6.1.1 기존 도메인 (DreamGym 논문과 직접 비교)

| 벤치마크 | 특성 | 목표 |
|---------|------|------|
| **WebArena** | Non-RL-ready, 동적 웹 환경 | Phase 0 효과 검증 |
| **WebShop** | RL-ready, 전자상거래 | 기존 성능 유지 확인 |
| **ALFWorld** | RL-ready, 가상 가정 환경 | 기존 성능 유지 확인 |

#### 6.1.2 새로운 도메인 (오프라인 데이터 부재)

| 도메인 | 오프라인 데이터 | Phase 0 역할 |
|--------|----------------|--------------|
| **새로운 웹 환경** | ❌ 없음 | Cold-start 해결 필수 |
| **로보틱스 시뮬레이션** | ❌ 없음 | 자율 데이터 수집 |
| **커스텀 비즈니스 환경** | ❌ 없음 | 실용성 검증 |

### 6.2 비교 베이스라인

#### 베이스라인 1: Vanilla DreamGym (공개 데이터 사용)
```python
baseline_1 = DreamGym(
    task_seeds=manual_seeds,          # 인간이 작성
    offline_data=public_leaderboard,  # WebArena 리더보드
    experience_model=train_from_scratch(public_leaderboard)
)
```

#### 베이스라인 2: DreamGym-AE (Phase 0 적용)
```python
baseline_2 = DreamGym_AE(
    # Phase 0: 자율 데이터 수집
    bootstrap_steps=5000,
    self_questioning=True,
    self_navigating=True,
    self_attributing=True,

    # Phase 1-N: 기존 DreamGym
    # (Phase 0 결과 자동 전달)
)
```

#### 베이스라인 3: Pure AgentEvolver (DreamGym 없음)
```python
baseline_3 = AgentEvolver(
    self_questioning=True,
    self_navigating=True,
    self_attributing=True,
    # RL 알고리즘: 전통적인 PPO (경험 합성 없음)
)
```

### 6.3 평가 지표

#### 지표 1: 최종 성능 (Success Rate %)
- **정의**: 테스트 세트에서 태스크 성공률
- **비교**: Vanilla DreamGym 대비 개선율

#### 지표 2: 데이터 효율성 (Steps to Threshold)
- **정의**: 목표 성능 (예: 10% success rate) 도달에 필요한 총 환경 스텝 수
- **비교**: Phase 0 포함 vs 오프라인 데이터 수집 비용

#### 지표 3: Zero-Shot 적응성 (New Domain Performance)
- **정의**: 오프라인 데이터 없는 새로운 도메인에서의 성능
- **비교**: DreamGym-AE만 적용 가능 (Vanilla DreamGym 불가)

#### 지표 4: 데이터 품질 (Experience Model Quality)
- **정의**: GPT-4o 판정 (Accuracy, Informativeness, Diversity)
- **비교**: Phase 0 데이터 vs 공개 리더보드 데이터

### 6.4 예상 결과

#### 시나리오 1: 기존 도메인 (오프라인 데이터 있음)

**WebArena 예상 결과**:

| 방법 | Success Rate | 환경 스텝 | 데이터 품질 (Judge 평균) |
|------|--------------|-----------|-------------------------|
| Vanilla DreamGym (10K offline) | 13.3% | 10K (offline) + 0 (synthetic) | 1.6 / 2.0 |
| **DreamGym-AE (5K Phase 0)** | **14.5%** | 5K (Phase 0) + 0 (synthetic) | **1.8 / 2.0** |
| Pure AgentEvolver | 9.2% | 80K (real RL) | N/A |

**예상 개선**:
- Success Rate: +1.2% (약 9% 상대 개선)
- 데이터 효율성: 50% 감소 (10K → 5K)
- 데이터 품질: +0.2 (판정 점수 향상)

**근거**:
- Self-Attributing으로 고품질 데이터만 선별 → 경험 모델 성능 향상
- Self-Navigating으로 효율적 탐색 → 더 적은 스텝으로 충분한 커버리지
- DreamGym의 검증된 프레임워크로 안정적 학습

#### 시나리오 2: 새로운 도메인 (오프라인 데이터 없음)

**새로운 웹 환경 예상 결과**:

| 방법 | 적용 가능 | Success Rate | 총 비용 |
|------|-----------|--------------|---------|
| Vanilla DreamGym | ❌ 불가 | - | - |
| **DreamGym-AE** | ✅ 가능 | **10-12%** | Phase 0 (5K) + 합성 (0) |
| Pure AgentEvolver | ✅ 가능 | 6-8% | Real RL (80K) |

**예상 장점**:
- **Zero-shot 적용 가능**: 오프라인 데이터 없이도 DreamGym의 장점 활용
- **비용 효율성**: 실제 환경 상호작용 5K만으로 경쟁력 있는 성능
- **확장성**: 새로운 도메인에 빠르게 적응

### 6.5 Ablation Studies

#### Ablation 1: Phase 0 컴포넌트 기여도

| 제거 컴포넌트 | Success Rate | 영향 분석 |
|--------------|--------------|-----------|
| Full DreamGym-AE | **14.5%** | (기준) |
| w/o Self-Questioning | 13.8% | ▼0.7% - 태스크 다양성 감소 |
| w/o Self-Navigating | 13.2% | ▼1.3% - 탐색 비효율 증가 |
| w/o Self-Attributing | 13.0% | ▼1.5% - 데이터 품질 저하 (가장 큰 영향!) |

**예상 인사이트**:
- Self-Attributing이 가장 중요 (DreamGym paper2 Table 2와 유사한 패턴)
- 세 메커니즘의 시너지 효과 존재

#### Ablation 2: Phase 0 데이터 크기

| Phase 0 스텝 | Success Rate | 경험 모델 품질 |
|--------------|--------------|----------------|
| 2K | 13.8% | 1.6 |
| 5K | **14.5%** | **1.8** |
| 10K | 14.6% | 1.9 |
| 20K | 14.7% | 1.9 |

**예상 인사이트**:
- **5K-10K가 sweet spot**: 추가 데이터의 한계 효용 감소
- DreamGym의 데이터 효율성 (paper2 Figure 5)과 일치

---

## 7. 구현 로드맵

### 7.1 Phase 1: 프로토타입 구현 (4-6주)

#### Week 1-2: Phase 0 기본 구현
```python
# Milestone 1.1: Self-Questioning
- ✅ LLM 기반 태스크 생성 프롬프트
- ✅ 실행 가능성 검증 로직
- ✅ 커리큘럼 난이도 조절

# Milestone 1.2: Self-Navigating
- ✅ 경험 메모리 (ReMe 통합)
- ✅ 유사도 기반 검색
- ✅ 하이브리드 정책 가이던스

# Milestone 1.3: Self-Attributing
- ✅ 기여도 평가 LLM
- ✅ 품질 필터링 로직
- ✅ 가중치 기반 샘플링
```

#### Week 3-4: DreamGym 통합
```python
# Milestone 2.1: 데이터 포맷 변환
- ✅ Phase 0 → DreamGym 리플레이 버퍼
- ✅ 태스크 형식 표준화
- ✅ 경험 모델 입력 검증

# Milestone 2.2: 경험 모델 학습
- ✅ Phase 0 데이터로 M_exp 학습
- ✅ 추론 트레이스 생성 검증
- ✅ 품질 판정 (GPT-4o)
```

#### Week 5-6: End-to-End 통합
```python
# Milestone 3.1: 전체 파이프라인
- ✅ Phase 0 → DreamGym 자동 전환
- ✅ 하이퍼파라미터 튜닝
- ✅ 로깅 및 모니터링

# Milestone 3.2: 초기 실험
- ✅ WebArena 벤치마크 (proof of concept)
- ✅ 베이스라인 비교
- ✅ 결과 분석
```

### 7.2 Phase 2: 검증 및 최적화 (6-8주)

#### Week 7-10: 벤치마크 평가
```python
# 기존 도메인
- WebArena (full evaluation)
- WebShop
- ALFWorld

# 새로운 도메인
- Custom web environment (zero-shot)
- Robotics simulation (zero-shot)
```

#### Week 11-12: Ablation Studies
```python
- Phase 0 컴포넌트별 기여도
- 데이터 크기 민감도
- 하이퍼파라미터 분석
```

#### Week 13-14: 최적화
```python
- 추론 속도 개선
- 메모리 효율성
- 병렬화 전략
```

### 7.3 Phase 3: 프로덕션 준비 (4-6주)

#### Week 15-16: 코드 정제
```python
- 모듈화 및 리팩토링
- 단위 테스트 및 통합 테스트
- 문서화
```

#### Week 17-18: 패키징
```python
- PyPI 패키지 배포
- Docker 이미지
- 튜토리얼 및 예제
```

#### Week 19-20: 논문 작성
```python
- 실험 결과 정리
- 논문 초안 작성
- 리뷰 및 수정
```

---

## 8. 기술 스택

### 8.1 핵심 라이브러리

```python
# LLM 및 에이전트
- transformers >= 4.40.0  # Llama-3.1-8B-Instruct
- vllm >= 0.5.0  # 빠른 추론
- langchain >= 0.2.0  # 프롬프트 관리

# 강화학습
- trl >= 0.8.0  # GRPO/PPO 구현
- gymnasium >= 0.29.0  # 환경 인터페이스

# 벡터 검색 (Self-Navigating)
- faiss-gpu >= 1.7.4  # 경험 메모리
- sentence-transformers >= 2.7.0  # 임베딩

# DreamGym 통합
- dreamgym  # (공식 구현 사용)
- agentscope  # ReMe 시스템

# 벤치마크
- webarena >= 1.0.0
- webshop >= 1.0.0
- alfworld >= 0.3.3

# 유틸리티
- wandb >= 0.17.0  # 실험 트래킹
- hydra-core >= 1.3.0  # 설정 관리
```

### 8.2 하드웨어 요구사항

#### Phase 0 (자율 데이터 수집)
```yaml
최소:
  - GPU: 1x NVIDIA A100 (40GB) or 2x RTX 4090
  - CPU: 16 cores
  - RAM: 64GB
  - 병렬 환경 인스턴스: 8-16개

권장:
  - GPU: 2x NVIDIA A100 (80GB)
  - CPU: 32 cores
  - RAM: 128GB
  - 병렬 환경 인스턴스: 32-64개
```

#### Phase 1-N (DreamGym 학습)
```yaml
# DreamGym paper2의 요구사항과 동일
최소:
  - GPU: 1x NVIDIA A100 (40GB)
  - CPU: 8 cores
  - RAM: 32GB

권장:
  - GPU: 1x NVIDIA H100 (80GB)
  - CPU: 16 cores
  - RAM: 64GB
```

### 8.3 예상 비용

#### 클라우드 비용 (AWS 기준)

**Phase 0 (5K 스텝 수집)**:
```
환경: p4d.24xlarge (8x A100)
시간: ~4-6 시간 (병렬 수집)
비용: $32.77/hour × 6 = ~$200

LLM API (Self-Questioning, Self-Attributing):
- 프롬프트: ~500K tokens
- 비용: ~$5 (GPT-4o-mini) or ~$50 (GPT-4o)
```

**Phase 1-N (DreamGym 학습)**:
```
환경: p4d.24xlarge
시간: ~10-20 시간 (100 iterations)
비용: $32.77/hour × 20 = ~$650

합성 경험 생성 (경량):
- GPU 추론만 사용
- 실제 환경 롤아웃 비용 없음
```

**총 예상 비용 (WebArena 1회 실험)**:
```
Phase 0: $200-250
Phase 1-N: $650
총: ~$900

vs Vanilla DreamGym:
- 오프라인 데이터 수집: $500+ (인간 주석 또는 기존 에이전트)
- DreamGym 학습: $650
총: ~$1150+

절감: ~20-30%
```

---

## 9. 한계 및 향후 연구

### 9.1 현재 한계

#### 한계 1: Phase 0 초기화 비용
**문제**:
- Phase 0는 여전히 실제 환경 상호작용 필요 (5K 스텝)
- 복잡한 환경에서는 더 많은 스텝 필요할 수 있음

**완화 방안**:
- 환경 복잡도 기반 적응형 Phase 0 길이
- 사전 학습된 경험 모델 재사용 (transfer learning)

#### 한계 2: LLM 기반 모듈의 품질 의존성
**문제**:
- Self-Questioning, Self-Attributing은 LLM 능력에 의존
- 저품질 LLM 사용 시 Phase 0 효과 감소

**완화 방안**:
- 최소 권장 모델: GPT-4o-mini 또는 Llama-3.1-70B
- 품질 검증 로직으로 낮은 품질 출력 필터링

#### 한계 3: 도메인 특화 프롬프트 필요
**문제**:
- Self-Questioning 프롬프트는 도메인마다 조정 필요
- 완전 자동화 어려움

**완화 방안**:
- 도메인별 프롬프트 템플릿 라이브러리 구축
- Few-shot 예제 자동 생성

### 9.2 향후 연구 방향

#### 방향 1: Phase 0 완전 자동화
**목표**: 인간 개입 제로로 새로운 환경 적응

**접근법**:
- 환경 자동 분석 (API 문서, 상태 공간 구조)
- 메타-학습으로 도메인 불변 Phase 0 전략 학습
- 대규모 환경 데이터셋으로 사전 학습

#### 방향 2: 멀티모달 환경 확장
**목표**: 텍스트 외 비전, 음성 환경 지원

**접근법**:
- 멀티모달 경험 모델 (vision + language)
- 픽셀 기반 상태 공간 지원
- 로보틱스 실제 환경 적용

#### 방향 3: 다중 에이전트로 확장
**목표**: 여러 에이전트가 협력하여 Phase 0 수행

**접근법**:
- 에이전트 간 경험 공유 (federated learning)
- 역할 분담 (explorer, evaluator, teacher)
- 사회적 학습 메커니즘

#### 방향 4: 이론적 분석 강화
**목표**: Phase 0의 오차 감소 효과 수학적 증명

**접근법**:
- Self-Attributing의 품질 필터링 효과 정량화
- Self-Navigating의 샘플 복잡도 분석
- 통합 정리 (Unified Theorem) 제시

---

## 10. 결론

### 10.1 핵심 기여

#### 기여 1: 검증된 프레임워크 보존
✅ **DreamGym의 이론과 실험 결과를 그대로 유지**
- Theorem 1의 정책 개선 보장 영향 없음
- 검증된 성능 (WebArena +82-108%) 기반 구축
- 최소 침습적 통합으로 안정성 확보

#### 기여 2: 오프라인 데이터 의존성 해결
✅ **AgentEvolver로 Cold-Start 문제 극복**
- 새로운 도메인에 zero-shot 적용 가능
- 오프라인 데이터 수집 비용 제거
- 자율적 초기 경험 생성

#### 기여 3: 실용적 하이브리드 방법론
✅ **명확한 역할 분담과 모듈성**
- Phase 0: AgentEvolver (초기화)
- Phase 1-N: DreamGym (경험 합성 + RL)
- 점진적 적용 가능 (오프라인 데이터 있으면 스킵)

### 10.2 예상 임팩트

#### 학술적 임팩트
- **이론**: Phase 0가 Theorem 1의 오차 항을 줄이는 메커니즘 분석
- **실험**: 새로운 벤치마크에서의 zero-shot 성능 검증
- **방법론**: 두 검증된 시스템의 효과적 통합 사례

#### 산업적 임팩트
- **비용 절감**: 오프라인 데이터 구축 비용 제거 (20-30% 절감)
- **빠른 적응**: 새로운 비즈니스 환경에 빠른 배포
- **확장성**: 다양한 도메인에 일관된 프레임워크 적용

#### 사회적 임팩트
- **접근성**: 대규모 데이터 없이도 고성능 에이전트 개발 가능
- **민주화**: 중소기업 및 연구자들의 에이전트 개발 장벽 낮춤
- **혁신**: 새로운 응용 분야 탐색 가능 (맞춤형 환경)

### 10.3 최종 요약

**DreamGym-AE**는:
1. **검증된 기반**: DreamGym의 이론과 실험 보존
2. **실용적 해결책**: AgentEvolver로 오프라인 데이터 의존성 해결
3. **확장 가능**: 새로운 도메인에 zero-shot 적용
4. **비용 효율적**: 데이터 수집 비용 20-30% 절감
5. **모듈형 설계**: 점진적 적용 및 커스터마이징 가능

**핵심 메시지**:
> DreamGym의 강력한 경험 합성 프레임워크 + AgentEvolver의 자율적 데이터 수집 = 확장 가능하고 비용 효율적인 에이전트 학습 시스템

---

## 참고문헌

### 주요 논문
1. **DreamGym**: Chen et al., "Scaling Agent Learning via Experience Synthesis", arXiv:2511.03773, 2025
2. **AgentEvolver**: Zhai et al., "AgentEvolver: Efficient Self-Evolving Agent System", arXiv:2511.10395, 2025

### 관련 시스템
- **ReMe**: https://github.com/agentscope-ai/ReMe (Self-Navigating 경험 메모리)
- **DreamGym GitHub**: (공식 구현 참조)
- **AgentEvolver GitHub**: https://github.com/modelscope/AgentEvolver

### 벤치마크
- **WebArena**: Zhou et al., 2023
- **WebShop**: Yao et al., 2022
- **ALFWorld**: Shridhar et al., 2020

---

## 부록

### A. 프롬프트 템플릿

#### A.1 Self-Questioning 프롬프트 (상세)

```python
SELF_QUESTIONING_PROMPT = """
You are an autonomous agent designing learning goals for yourself in a new environment.

## Environment Context
Domain: {domain}
Description: {env_description}
Available Actions: {action_space}
State Representation: {state_description}

## Current Understanding
- Explored States: {num_explored_states}
- Successful Actions: {successful_actions}
- Failed Actions: {failed_actions}
- Existing Goals: {existing_goals}

## Your Task
Generate a NEW learning goal that:
1. **Feasibility**: Can be achieved with available actions
2. **Novelty**: Different from existing goals
3. **Clarity**: Has unambiguous success criteria
4. **Informativeness**: Teaches useful skills for the domain
5. **Difficulty**: Matches current capability level

## Current Capability Level
Based on success rate: {success_rate:.1%}
- If < 30%: Generate EASY goals
- If 30-70%: Generate MEDIUM goals
- If > 70%: Generate HARD goals

## Output Format (JSON)
{{
  "goal": "Clear natural language description",
  "success_criteria": "Precise condition for success",
  "preconditions": ["State requirements before attempting"],
  "estimated_steps": <int>,
  "difficulty": "easy|medium|hard",
  "skills_learned": ["What this goal teaches"],
  "rationale": "Why this goal is valuable"
}}

## Examples for WebArena Domain
{{
  "goal": "Find the total number of commits in April 2023 for a GitHub repository",
  "success_criteria": "Extract and return the exact commit count",
  "preconditions": ["Repository page is loaded", "Commit history is accessible"],
  "estimated_steps": 5-8,
  "difficulty": "medium",
  "skills_learned": ["Navigating commit history", "Filtering by date", "Extracting numerical data"],
  "rationale": "Combines navigation, filtering, and extraction - core skills for code repository interaction"
}}

Now generate a NEW goal:
"""
```

#### A.2 Self-Attributing 프롬프트 (상세)

```python
SELF_ATTRIBUTING_PROMPT = """
You are analyzing an agent's trajectory to identify which actions contributed to the final outcome.

## Task Description
{task_description}

## Full Trajectory
{format_full_trajectory(trajectory)}

## Final Outcome
- Success: {is_success}
- Final Reward: {final_reward}
- Total Steps: {len(trajectory)}

## Focus: Step {current_step}
State (before action):
{state_t}

Action taken:
{action_t}

State (after action):
{state_t_plus_1}

Immediate Reward:
{reward_t}

## Your Task
Analyze this SINGLE action's contribution to the final outcome.

### Evaluation Criteria

**1. Positive Contribution (0.8-1.0)**
- Action directly achieved the goal or a critical sub-goal
- Without this action, success would be impossible
- Action revealed crucial information

**2. Helpful Contribution (0.5-0.7)**
- Action moved closer to the goal
- Action was necessary but not critical
- Action provided useful information

**3. Neutral Contribution (0.2-0.4)**
- Action maintained current state without harm
- Action was exploratory but not directly helpful
- Action could be replaced with alternatives

**4. Negative Contribution (0.0-0.1)**
- Action moved away from the goal
- Action wasted time or resources
- Action created obstacles

**5. Irrelevant (-1.0)**
- Action had no causal relationship to outcome
- Action was random or uninformed
- Action's effect was overridden by later actions

## Output Format (JSON)
{{
  "contribution_score": <float in [-1.0, 1.0]>,
  "criticality": "critical|helpful|neutral|harmful|irrelevant",
  "causal_chain": "How this action led to (or hindered) final outcome",
  "counterfactual": "What would happen if this action was different",
  "information_gain": "What the agent learned from this action",
  "recommendation": "Should this pattern be reinforced or avoided"
}}

## Example (WebArena - GitHub Commits Task)

Step 3 of 7:
State: [Overview page with "Total Commits" button visible]
Action: Click("Total Commits" button)
Next State: [Commit history page loaded, showing all commits by date]
Immediate Reward: 0

Analysis:
{{
  "contribution_score": 0.9,
  "criticality": "critical",
  "causal_chain": "This action transitioned from overview to commit history, which is REQUIRED to access April 2023 commits. Without this, the filtering step (Step 4) and final extraction (Step 6) would be impossible.",
  "counterfactual": "If agent clicked a different button, it would need to navigate back, wasting 2-3 steps or failing entirely.",
  "information_gain": "Agent learned that 'Total Commits' button is the entry point for commit history - reusable knowledge for similar tasks.",
  "recommendation": "STRONGLY reinforce this pattern: identify and click navigation elements that expose target information."
}}

Now analyze Step {current_step}:
"""
```

### B. 하이퍼파라미터 설정

#### B.1 Phase 0 기본 설정

```yaml
phase_0:
  # Self-Questioning
  self_questioning:
    num_initial_tasks: 15
    difficulty_stages: ["easy", "medium", "hard"]
    success_rate_threshold: 0.7  # 다음 난이도로 진행
    max_retries: 3  # 실행 불가능한 목표 재생성

  # Self-Navigating
  self_navigating:
    experience_memory_size: 10000
    similarity_threshold: 0.75  # 유사 경험 사용 최소 유사도
    exploration_epsilon: 0.3  # 탐색 확률
    adaptation_temperature: 0.8  # 과거 행동 적응 정도

  # Self-Attributing
  self_attributing:
    quality_threshold: 0.6  # 리플레이 버퍼 추가 최소 점수
    batch_size: 50  # 기여도 평가 배치
    judge_model: "gpt-4o-mini"  # 또는 "gpt-4o"

  # 데이터 수집
  collection:
    target_steps: 5000  # 목표 스텝 수
    episodes_per_task: 10
    max_episode_length: 50
    early_stopping: true  # 목표 달성 시 조기 종료

  # 경험 모델 학습
  experience_model:
    base_model: "meta-llama/Llama-3.1-8B-Instruct"
    num_epochs: 3
    learning_rate: 1e-5
    batch_size: 16
    gradient_accumulation_steps: 4
```

#### B.2 DreamGym 설정 (Phase 1-N)

```yaml
dreamgym:
  # 커리큘럼 학습
  curriculum:
    initial_task_pool_size: 15  # Phase 0에서 제공
    reward_entropy_threshold: 0.5  # 도전적 태스크 기준
    task_augmentation_rate: 5  # 반복당 추가 태스크 수
    max_task_pool_size: 100

  # 경험 합성
  synthesis:
    rollouts_per_task: 50
    max_rollout_length: 50
    reasoning_temperature: 0.7
    diversity_penalty: 0.1  # 중복 경험 페널티

  # RL 최적화
  rl:
    algorithm: "GRPO"  # 또는 "PPO"
    num_iterations: 100
    learning_rate: 1e-6
    kl_penalty: 0.1
    value_loss_coef: 0.5  # PPO only

  # DreamGym-S2R (optional)
  sim_to_real:
    enabled: false
    real_data_ratio: 0.1  # 전체 데이터 중 실제 비율
    finetune_frequency: 10  # 반복마다 경험 모델 정제
```

### C. 구현 체크리스트

#### C.1 Phase 0 구현

- [ ] **Self-Questioning**
  - [ ] LLM 프롬프트 구현
  - [ ] 목표 실행 가능성 검증
  - [ ] 커리큘럼 난이도 조절
  - [ ] 중복 목표 제거

- [ ] **Self-Navigating**
  - [ ] 경험 메모리 (FAISS 인덱스)
  - [ ] 유사도 기반 검색
  - [ ] 하이브리드 정책 구현
  - [ ] ReMe 시스템 통합

- [ ] **Self-Attributing**
  - [ ] 기여도 평가 LLM
  - [ ] 배치 처리 파이프라인
  - [ ] 품질 필터링
  - [ ] 가중치 샘플링

- [ ] **통합**
  - [ ] 데이터 수집 루프
  - [ ] 경험 모델 학습
  - [ ] DreamGym 포맷 변환

#### C.2 DreamGym 통합

- [ ] **포맷 변환**
  - [ ] 리플레이 버퍼 변환
  - [ ] 태스크 형식 표준화
  - [ ] 경험 모델 입력 검증

- [ ] **자동 전환**
  - [ ] Phase 0 → Phase 1 파이프라인
  - [ ] 완료 조건 체크
  - [ ] 오류 처리

- [ ] **모니터링**
  - [ ] WandB 로깅
  - [ ] 품질 메트릭 트래킹
  - [ ] 중간 체크포인트

#### C.3 평가

- [ ] **벤치마크**
  - [ ] WebArena 환경 설정
  - [ ] WebShop 환경 설정
  - [ ] ALFWorld 환경 설정
  - [ ] 새로운 도메인 환경 준비

- [ ] **메트릭**
  - [ ] Success Rate 계산
  - [ ] 데이터 효율성 측정
  - [ ] 경험 모델 품질 판정
  - [ ] 비용 분석

- [ ] **Ablation Studies**
  - [ ] 컴포넌트별 제거 실험
  - [ ] 데이터 크기 변화
  - [ ] 하이퍼파라미터 민감도

---

**문서 버전**: 1.0
**작성일**: 2025-11-22
**상태**: 초안 완성, 구현 대기
