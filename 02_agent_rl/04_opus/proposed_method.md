# LRMs are World Models
### Agent RLAIF without World Model Training

---

## 제안 방법론 설계 문서

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   Title:     LRMs are World Models                                          │
│   Subtitle:  Agent RLAIF without World Model Training                       │
│                                                                             │
│   Core Thesis:                                                              │
│   "Large Reasoning Models already possess sufficient world knowledge        │
│    to simulate text environments — no world model training needed."         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 개요

### 1.1 목표
이미 학습된 오픈소스 Agentic LRM(Large Reasoning Model)을 post-training RL로 특정 도메인/환경에서 더욱 강화하는 방법론.

### 1.2 핵심 가정 (Title Thesis)

> **"LRMs are World Models"**
>
> Large Reasoning Model은 이미 충분한 세계 지식과 추론 능력을 갖추고 있어,
> 별도의 학습 없이 Text World Simulator로 활용할 수 있다.

이 가정을 기반으로:
- World Model 학습 비용 제거
- Frontier LRM을 환경 시뮬레이터로 직접 활용
- RLAIF 방식으로 Agent Post-training 수행

### 1.3 핵심 아이디어
- **World Model**: Frontier LRM을 학습 없이 환경 시뮬레이터로 활용
- **Task Generation**: 호기심 기반 + Reward Entropy 필터링으로 효과적인 학습 태스크 선택
- **Reward Design**: Step-level 이진 판단 + AND 연산으로 명확한 학습 신호
- **Training**: Offline GRPO로 단순하고 효율적인 학습

---

## 2. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Post-training RL Framework                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    1. Task Generation                                │   │
│  │  ┌───────────────────────┐    ┌───────────────────────┐             │   │
│  │  │   Curiosity-based     │    │   Reward Entropy      │             │   │
│  │  │   Task Proposal       │ ──►│   Filtering           │             │   │
│  │  │   (다양한 후보 생성)   │    │   (학습 효과 최적화)   │             │   │
│  │  └───────────────────────┘    └───────────────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    2. Data Generation                                │   │
│  │                                                                      │   │
│  │   Actor (학습 대상)              World Model (Frozen)                │   │
│  │  ┌─────────────────┐           ┌─────────────────────────┐          │   │
│  │  │ 오픈소스        │  action   │ Frontier LRM            │          │   │
│  │  │ Agentic LRM     │ ────────► │ + Domain Context        │          │   │
│  │  │                 │           │                         │          │   │
│  │  │                 │ ◄──────── │ • Next State 예측       │          │   │
│  │  │                 │  state,   │ • Step Reward 판단 (0/1)│          │   │
│  │  └─────────────────┘  reward   └─────────────────────────┘          │   │
│  │                                                                      │   │
│  │                         ▼                                            │   │
│  │                  Trajectory Dataset                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    3. Policy Training                                │   │
│  │                                                                      │   │
│  │                    Offline GRPO                                      │   │
│  │                    (상대적 trajectory 비교)                          │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 핵심 구성 요소

### 3.1 Actor: 오픈소스 Agentic LRM

**역할**: 학습 대상 정책 (Policy)

**특징**:
- 이미 기본적인 agent 능력 보유 (SFT 완료)
- Post-training RL로 특정 도메인 성능 강화
- 예: Llama-3-Agent, Qwen-Agent 등

**왜 중요한가**:
- Random policy가 아닌 reasonable baseline에서 시작
- 생성되는 trajectory 품질 보장
- OOD(Out-of-Distribution) 문제 완화

---

### 3.2 World Model: Frozen Frontier LRM

**역할**: 환경 시뮬레이터 (학습 없음)

**특징**:
- Frontier LRM을 API 호출로 사용 (Claude, GPT-4 등)
- 학습/파인튜닝 없이 고정된 상태로 사용
- 두 가지 역할 수행:
  1. Next state 예측
  2. Step-level reward 판단

**왜 Frontier LRM인가**:
```
장점:
├── 학습 비용 제거 (World Model 훈련 불필요)
├── 풍부한 세계 지식 내장
├── 자동 개선 (Frontier 모델 발전에 따라)
└── Zero-shot 일반화 능력

비용:
└── API 호출 비용 (데이터 생성 단계에 집중)
```

**경험 메모리 검색 제거**:
```
기존 논문들: 경험 메모리에서 유사 사례 검색
문제점:
├── 강한 Inductive Bias (표면적 유사성 함정)
├── Frontier LRM이 이미 충분한 지식 보유
└── 검색 오버헤드 > 정확도 향상

결정: 경험 메모리 제거, Domain Context만 직접 주입
```

**Domain Context 설계**:
```python
def create_world_model_prompt(state, action, task, domain_context):
    return f"""
    [Domain Context]
    {domain_context}

    [Current Situation]
    Task: {task.description}
    Available Tools: {format_tools(task.available_tools)}
    State: {state}
    Action: {action}

    [Instructions]
    1. 위 행동의 결과로 나타날 다음 상태를 예측하시오.
    2. 이 행동이 태스크 목표에 부합하는 올바른 행동인지 판단하시오. (0 또는 1)
       - 사용 가능한 도구 외의 도구를 사용했다면 0
       - 도구를 올바르게 사용했지만 목표와 무관하면 0
       - 목표 달성에 기여하는 올바른 도구 사용이면 1

    [Output Format]
    Next State: ...
    Step Reward: 0 or 1
    Reasoning: ...
    """
```

---

### 3.3 Task Generation: 호기심 + Reward Entropy

**목표**: 학습 효과를 극대화하는 태스크 선택

#### 3.3.1 호기심 기반 태스크 제안 (Curiosity-based Proposal)

**역할**: 다양한 태스크 후보 생성 (Task + Tool Set)

**핵심**: Agent 학습에서는 Task와 함께 사용 가능한 Tool Set도 정의되어야 함
```
Task 정의 = {
    description: "무엇을 해야 하는가",
    available_tools: ["어떤 도구를 쓸 수 있는가"],
    initial_state: "시작 상태"
}

→ Agent: Tool Set 기반으로 action 선택
→ World Model: Tool Set 기반으로 next state 시뮬레이션
```

```python
@dataclass
class Task:
    description: str           # 태스크 설명
    available_tools: list[Tool]  # 사용 가능한 도구들
    initial_state: str         # 초기 상태
    max_steps: int = 20        # 최대 스텝 수

@dataclass
class Tool:
    name: str                  # 도구 이름 (e.g., "Read", "Edit", "Bash")
    description: str           # 도구 설명
    parameters: dict           # 파라미터 스키마

def generate_task_candidates(domain_context, tool_library, num_candidates=100):
    """
    도메인 context 기반으로 다양한 태스크 후보 생성
    각 태스크는 사용 가능한 tool set과 함께 정의됨
    """
    prompt = f"""
    [Domain]
    {domain_context}

    [Available Tool Library]
    {format_tools(tool_library)}

    [Instructions]
    이 도메인에서 agent가 학습하면 좋을 다양한 태스크를 생성하시오.
    각 태스크에 대해 필요한 도구들을 tool library에서 선택하시오.

    다양성 기준:
    - 난이도: 쉬움, 중간, 어려움 균형
    - 상황: 일반적 케이스 + 엣지 케이스
    - 목표 유형: 탐색, 조작, 복합 목표 등
    - 도구 조합: 단일 도구 ~ 다중 도구 조합

    [Output Format]
    Task 1:
      description: ...
      tools: [tool1, tool2, ...]
      initial_state: ...

    Task 2:
      ...
    """
    return task_generator.generate(prompt, n=num_candidates)
```

**Tool Set 설계 전략**:
```
1. 태스크 난이도와 도구 수 연동
   ├── Easy: 1-2개 도구 (e.g., Read만)
   ├── Medium: 3-5개 도구 (e.g., Read, Edit, Grep)
   └── Hard: 5개+ 도구 (e.g., 전체 도구 세트)

2. 도구 조합 다양성
   ├── 읽기 전용: [Read, Grep, Glob]
   ├── 수정 포함: [Read, Edit, Write]
   ├── 실행 포함: [Bash, Read, Write]
   └── 전체: [Read, Edit, Write, Bash, Grep, Glob, ...]

3. 도메인별 특화 도구
   ├── 코딩: [Read, Edit, Bash, Grep]
   ├── 데이터 분석: [Read, Write, Python]
   └── 시스템 관리: [Bash, Read, Write]
```

#### 3.3.2 Task Pool 관리

**문제**: Task + Tool Set을 한 번 쓰고 버리면 낭비
```
Bad: Task 생성 → 1회 사용 → 폐기 → 다시 생성 (비효율)
Good: Task Pool 유지 → 반복 탐험 → 통계 축적 → Curriculum 형성
```

**Task Pool 구조**:
```python
@dataclass
class TaskEntry:
    task: Task                    # Task + Tool Set
    created_at: int               # 생성 iteration
    attempts: int = 0             # 시도 횟수
    successes: int = 0            # 성공 횟수
    recent_rewards: list = None   # 최근 K번 reward (entropy 계산용)
    entropy: float = 1.0          # 현재 entropy 추정치
    status: str = "active"        # active / mastered / too_hard

class TaskPool:
    def __init__(self, max_size=500):
        self.pool: dict[str, TaskEntry] = {}
        self.max_size = max_size

    def add_tasks(self, tasks: list[Task]):
        """새 태스크 추가"""
        for task in tasks:
            task_id = hash_task(task)
            if task_id not in self.pool:
                self.pool[task_id] = TaskEntry(task=task, created_at=self.iteration)

    def update_stats(self, task_id: str, reward: float):
        """태스크 통계 업데이트"""
        entry = self.pool[task_id]
        entry.attempts += 1
        entry.successes += int(reward > 0)
        entry.recent_rewards.append(reward)
        entry.recent_rewards = entry.recent_rewards[-10:]  # 최근 10개만 유지

        # Entropy 재계산
        entry.entropy = compute_entropy(entry.recent_rewards)

        # 상태 업데이트
        if entry.entropy < 0.1 and entry.successes / entry.attempts > 0.9:
            entry.status = "mastered"
        elif entry.entropy < 0.1 and entry.successes / entry.attempts < 0.1:
            entry.status = "too_hard"

    def sample_tasks(self, n: int, strategy: str = "entropy") -> list[Task]:
        """학습할 태스크 샘플링"""
        active_tasks = [e for e in self.pool.values() if e.status == "active"]

        if strategy == "entropy":
            # Entropy 높은 순 (학습 효과 최대화)
            active_tasks.sort(key=lambda e: e.entropy, reverse=True)
        elif strategy == "balanced":
            # Entropy 기반 확률적 샘플링
            probs = softmax([e.entropy for e in active_tasks])
            return np.random.choice(active_tasks, n, p=probs, replace=False)

        return [e.task for e in active_tasks[:n]]

    def get_curriculum_stats(self):
        """현재 curriculum 상태"""
        return {
            "total": len(self.pool),
            "active": sum(1 for e in self.pool.values() if e.status == "active"),
            "mastered": sum(1 for e in self.pool.values() if e.status == "mastered"),
            "too_hard": sum(1 for e in self.pool.values() if e.status == "too_hard"),
            "avg_entropy": np.mean([e.entropy for e in self.pool.values()])
        }
```

**Task Pool 생명주기**:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Task Pool Lifecycle                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   [Task Generator] ──► [Task Pool] ◄── 새 태스크 주기적 추가                │
│                             │                                               │
│                             ▼                                               │
│                    ┌────────────────┐                                       │
│                    │  Active Tasks  │ ◄─── Entropy 높음, 학습 중            │
│                    └───────┬────────┘                                       │
│                            │                                                │
│              ┌─────────────┼─────────────┐                                  │
│              ▼             ▼             ▼                                  │
│     ┌──────────────┐ ┌──────────┐ ┌────────────┐                           │
│     │   Mastered   │ │  Active  │ │  Too Hard  │                           │
│     │ (entropy↓    │ │ (계속    │ │ (entropy↓  │                           │
│     │  success↑)   │ │  학습)   │ │  success↓) │                           │
│     └──────────────┘ └──────────┘ └─────┬──────┘                           │
│            │                            │                                   │
│            │         Actor 성장 후      │                                   │
│            │         ◄─────────────────┘                                   │
│            │         재활성화 가능                                          │
│            ▼                                                                │
│     [Mastered Pool] ── 가끔 재테스트 (forgetting 체크)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**주기적 Task Pool 관리**:
```python
def manage_task_pool(pool: TaskPool, task_generator, iteration: int):
    """매 N iteration마다 실행"""

    # 1. 새 태스크 추가 (탐험)
    if pool.get_curriculum_stats()["active"] < 100:
        new_tasks = task_generator.generate_candidates(n=20)
        pool.add_tasks(new_tasks)

    # 2. Too Hard → Active 재활성화 (Actor 성장 반영)
    if iteration % 50 == 0:
        for entry in pool.pool.values():
            if entry.status == "too_hard":
                entry.status = "active"
                entry.recent_rewards = []  # 리셋

    # 3. Mastered 태스크 재검증 (Catastrophic Forgetting 체크)
    if iteration % 100 == 0:
        mastered = [e for e in pool.pool.values() if e.status == "mastered"]
        sample = random.sample(mastered, min(5, len(mastered)))
        for entry in sample:
            # 재테스트 후 성공률 떨어지면 재활성화
            pass

    # 4. 오래된 미사용 태스크 정리
    pool.cleanup_old_tasks(max_age=500)
```

#### 3.3.3 Reward Entropy 기반 필터링

**역할**: Task Pool에서 학습 효과가 높은 태스크 선택

**핵심 아이디어**:
```
Task A: 항상 성공 → Entropy ≈ 0 → "이미 마스터함, 스킵"
Task B: 항상 실패 → Entropy ≈ 0 → "아직 너무 어려움, 스킵"
Task C: 반반      → Entropy = 1 → "지금 배우기 딱 좋음" ✓
```

**시각화**:
```
                    Reward Entropy
                         │
                     1.0 ┤      ●──● 학습 최적 구간
                         │     /    \
                         │    /      \
                         │   /        \
                     0.0 ┤──●          ●──
                         └──┴──────────┴──► Task 난이도
                          쉬움        어려움
```

**구현**:
```python
def select_tasks_by_entropy(task_candidates, actor, world_model,
                            k_rollouts=5, top_n=20):
    """
    Reward entropy 기반 태스크 선택
    """
    task_scores = []

    for task in task_candidates:
        # K번 rollout으로 reward 분포 추정
        rewards = []
        for _ in range(k_rollouts):
            trajectory = rollout(actor, world_model, task)
            reward = compute_trajectory_reward(trajectory)
            rewards.append(reward)

        # Entropy 계산
        p_success = sum(rewards) / len(rewards)
        p_failure = 1 - p_success

        if p_success > 0 and p_failure > 0:
            entropy = -p_success * log(p_success) - p_failure * log(p_failure)
        else:
            entropy = 0

        task_scores.append((task, entropy))

    # Entropy 높은 순으로 정렬 후 선택
    task_scores.sort(key=lambda x: x[1], reverse=True)
    return [task for task, _ in task_scores[:top_n]]
```

**자동 Curriculum 효과**:
```
Iteration 1: 쉬운 task entropy 높음 → 선택 → 학습 → 마스터
Iteration 2: 쉬운 task entropy 낮아짐, 중간 task entropy 높아짐 → 선택
Iteration 3: 중간 task 마스터, 어려운 task entropy 높아짐 → 선택

→ 자동으로 easy → medium → hard 커리큘럼 형성
```

---

### 3.4 Reward Design: Step-level Binary + AND

**설계 철학**: 명확하고 해석 가능한 학습 신호

#### 3.4.1 Step-level Reward

```
World Model이 각 step마다 판단:
- 1: 올바른 행동 (태스크 목표 방향으로 진행)
- 0: 잘못된 행동 (태스크 목표에서 벗어남)
```

#### 3.4.2 Trajectory Reward

```
AND 연산으로 결정:
R_trajectory = r_0 ∧ r_1 ∧ r_2 ∧ ... ∧ r_T

예시:
[a0, a1, a2, a3, a4] → [1, 1, 1, 0, 1]
Trajectory Reward = 1 ∧ 1 ∧ 1 ∧ 0 ∧ 1 = 0 (실패)
                              ↑
                         실패 지점 명확
```

**장점**:
```
1. 실패 지점 명확
   └── "3번째 행동에서 틀림" 바로 식별 가능

2. Credit Assignment 자동
   └── 별도의 Self-Attributing 불필요
   └── 0 받은 step = 잘못된 행동

3. 부분 성공 정보 보존
   └── 실패 trajectory도 "여기까진 잘했다" 정보 있음

4. 해석 가능
   └── World Model이 왜 0/1 줬는지 reasoning 제공
```

**구현**:
```python
def compute_rewards(trajectory, world_model, task):
    """
    Step-level 및 trajectory-level reward 계산
    """
    step_rewards = []
    reasonings = []

    for (state, action, next_state) in trajectory:
        # World Model이 이 행동의 적절성 판단
        result = world_model.judge(
            state=state,
            action=action,
            next_state=next_state,
            task=task
        )
        step_rewards.append(result.reward)  # 0 or 1
        reasonings.append(result.reasoning)

    # Trajectory reward: AND 연산
    trajectory_reward = 1 if all(r == 1 for r in step_rewards) else 0

    # 첫 번째 실패 지점
    first_failure_idx = next(
        (i for i, r in enumerate(step_rewards) if r == 0),
        None
    )

    return {
        'step_rewards': step_rewards,
        'trajectory_reward': trajectory_reward,
        'first_failure_idx': first_failure_idx,
        'reasonings': reasonings
    }
```

---

### 3.5 Training: Offline GRPO

**선택 이유**:
```
GRPO (Group Relative Policy Optimization):
├── Value function 학습 불필요
├── Reward scale에 무관 (상대 비교)
├── Offline 데이터와 자연스럽게 호환
├── 구현 간단, 안정적
└── DeepSeek 등에서 검증됨
```

**알고리즘**:
```python
def grpo_training(actor, dataset, num_epochs):
    """
    Offline GRPO 학습
    """
    for epoch in range(num_epochs):
        # 같은 task의 trajectory들을 그룹으로 묶음
        for task, trajectories in group_by_task(dataset):
            # 각 trajectory의 reward
            rewards = [t['trajectory_reward'] for t in trajectories]

            # 그룹 내 상대적 advantage 계산
            baseline = mean(rewards)
            advantages = [r - baseline for r in rewards]

            # 정책 업데이트
            for traj, adv in zip(trajectories, advantages):
                if adv > 0:
                    # 좋은 trajectory 강화
                    actor.reinforce(traj, weight=adv)
                else:
                    # 나쁜 trajectory 약화
                    actor.penalize(traj, weight=-adv)

        # Step-level 정보 활용 (선택적)
        for traj in dataset:
            if traj['first_failure_idx'] is not None:
                # 실패 이전 행동들은 강화
                good_steps = traj['steps'][:traj['first_failure_idx']]
                actor.reinforce_steps(good_steps)

                # 실패 행동은 약화
                bad_step = traj['steps'][traj['first_failure_idx']]
                actor.penalize_step(bad_step)

    return actor
```

---

### 3.6 Training: Online GRPO (확장)

**Offline의 한계**:
```
Offline:
├── 고정된 데이터셋으로 학습
├── Actor가 개선되어도 데이터는 그대로
├── Distribution shift 발생 가능
└── 탐색 범위 제한

Online이 필요한 이유:
├── 최신 Actor로 계속 데이터 생성
├── 분포 일치 (on-policy)
├── 더 넓은 탐색 가능
└── 일반적으로 더 좋은 최종 성능
```

**Online vs Offline 비교**:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Offline:                                                                   │
│  ─────────────────────────────────────────────────────────                  │
│  [Data Gen] ──► [Dataset] ──► [Train] ──► [Train] ──► ... ──► [Done]       │
│      │              │                                                       │
│      └── 1회 ───────┘                                                       │
│                                                                             │
│  Online (On-policy):                                                        │
│  ─────────────────────────────────────────────────────────                  │
│  [Data Gen] ──► [Train] ──► [Data Gen] ──► [Train] ──► ... ──► [Done]      │
│       │            │             │             │                            │
│       └── Actor ───┘             └── Actor ────┘                            │
│           v1                         v2                                     │
│                                                                             │
│  * 각 iteration에서 현재 정책으로 생성한 데이터만 사용 (no replay buffer)    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Online GRPO 알고리즘 (Pure On-policy)**:
```python
def online_grpo_training(actor, world_model, task_generator, task_pool,
                         num_iterations, rollouts_per_iter):
    """
    Online GRPO 학습 (Pure On-policy)

    GRPO는 on-policy 알고리즘이므로 replay buffer를 사용하지 않음.
    매 iteration마다 현재 정책으로 새 데이터를 생성하고 바로 학습.
    Task Pool을 통해 curriculum 유지.
    """
    for iteration in range(num_iterations):
        # ===== Phase 0: Task Pool 관리 =====
        manage_task_pool(task_pool, task_generator, iteration)

        # ===== Phase 1: Task Selection =====
        # Task Pool에서 entropy 기반 샘플링
        selected_tasks = task_pool.sample_tasks(n=10, strategy="entropy")

        # ===== Phase 2: On-policy Data Generation =====
        # 현재 Actor로 새 데이터 생성
        trajectories = []
        for task in selected_tasks:
            task_id = hash_task(task)
            for _ in range(rollouts_per_iter // len(selected_tasks)):
                traj = rollout(actor, world_model, task)
                rewards = compute_rewards(traj, task)
                trajectories.append({
                    'task': task,
                    'task_id': task_id,
                    'trajectory': traj,
                    **rewards
                })

                # Task Pool 통계 업데이트
                task_pool.update_stats(task_id, rewards['trajectory_reward'])

        # ===== Phase 3: Policy Update =====
        # 현재 iteration에서 생성한 데이터만으로 업데이트 (on-policy)
        actor = grpo_update(actor, trajectories)

        # ===== Logging =====
        log_metrics(iteration, trajectories)
        log_curriculum_stats(task_pool.get_curriculum_stats())

    return actor
```

**GRPO가 On-policy인 이유**:
```
GRPO (Group Relative Policy Optimization):
├── 같은 task에 대해 여러 trajectory를 현재 정책으로 생성
├── 그룹 내 상대적 비교로 advantage 계산
├── 현재 정책의 행동 분포에서 직접 학습
└── Importance sampling 없이 gradient 계산

Off-policy (replay buffer) 사용 시 문제:
├── 과거 정책의 데이터 → 현재 정책과 분포 불일치
├── Importance weight 필요 → GRPO 장점 상실
└── 이론적 보장 깨짐
```

**On-policy의 Trade-off**:
```
장점:
├── 이론적으로 올바름 (분포 일치)
├── 구현 단순 (importance weight 불필요)
└── 안정적 학습

단점:
├── Sample efficiency 낮음 (데이터 1회만 사용)
├── 매 iteration API 호출 필요
└── 비용 증가
```

#### 3.6.1 Async Rollout (고급)

```
동기 방식 (Simple):
┌─────────────────────────────────────────────────┐
│  Rollout → Train → Rollout → Train → ...       │
│                                                 │
│  단점: GPU idle time 발생                        │
└─────────────────────────────────────────────────┘

비동기 방식 (Advanced):
┌─────────────────────────────────────────────────┐
│  Rollout Workers ──► Queue ──► Train            │
│       │                          │              │
│       └──── 동기화 (주기적) ──────┘              │
│                                                 │
│  * Actor 버전 동기화로 on-policy 유지           │
│  * 장점: GPU 활용도 최대화                       │
└─────────────────────────────────────────────────┘
```

```python
# 비동기 Online 학습 (On-policy 유지)
class AsyncOnlineTrainer:
    def __init__(self, actor, world_model, num_workers=4):
        self.actor = actor
        self.world_model = world_model
        self.num_workers = num_workers
        self.trajectory_queue = Queue()

    async def rollout_worker(self, worker_id):
        """비동기 rollout 워커 (동기화된 actor 사용)"""
        while not self.done:
            # 현재 actor 스냅샷 가져오기 (on-policy 보장)
            current_actor = self.get_synced_actor()
            task = self.task_generator.sample()
            traj = await self.async_rollout(current_actor, task)
            self.trajectory_queue.put(traj)

    def train_iteration(self):
        """학습 iteration (큐에서 배치 수집 후 학습)"""
        # 충분한 trajectory 수집 대기
        batch = []
        while len(batch) < batch_size:
            batch.append(self.trajectory_queue.get())

        # On-policy 업데이트
        self.actor = grpo_update(self.actor, batch)

        # 워커들에게 새 actor 동기화
        self.sync_actor_to_workers()
```

**Online 구현 시 고려사항**:
```
1. API Rate Limiting
   └── World Model API 호출 제한 고려
   └── 배치 요청, 캐싱 등 최적화

2. Actor 버전 동기화
   └── Async 시 워커들이 최신 actor 사용하도록
   └── 주기적 동기화 또는 iteration 단위 동기화

3. 비용 관리
   └── Online = 지속적 API 호출
   └── 호출량 모니터링, 예산 설정

4. 수렴 판단
   └── 언제 학습을 멈출지
   └── Reward entropy 기반 또는 성능 plateau 감지
```

**Offline → Online 전환 전략**:
```
추천: Offline으로 시작 → Online으로 fine-tune

Phase 1 (Offline):
├── 대량 데이터 생성 (1회)
├── 빠르게 reasonable policy 확보
└── Baseline 성능 달성

Phase 2 (Online):
├── Offline policy로 초기화
├── Online으로 추가 학습 (on-policy)
└── 더 높은 최종 성능 달성

장점:
├── Offline으로 빠른 부트스트랩
├── Online으로 ceiling 높이기
└── 비용 효율적
```

---

## 4. 전체 학습 파이프라인

```python
class PostTrainingRL:
    def __init__(self,
                 actor_model_path,
                 world_model_api,
                 domain_context,
                 tool_library):
        # Actor: 학습 대상 (오픈소스 Agentic LRM)
        self.actor = load_model(actor_model_path)

        # World Model: Frozen Frontier LRM
        self.world_model = FrozenWorldModel(
            api=world_model_api,
            domain_context=domain_context
        )

        # Task Generator & Pool
        self.task_generator = TaskGenerator(domain_context, tool_library)
        self.task_pool = TaskPool(max_size=500)

        # 초기 Task Pool 구성
        initial_tasks = self.task_generator.generate_candidates(n=100)
        self.task_pool.add_tasks(initial_tasks)

    def train(self, num_iterations, tasks_per_iter, rollouts_per_task):
        for iteration in range(num_iterations):
            # ===== Phase 0: Task Pool 관리 =====
            manage_task_pool(self.task_pool, self.task_generator, iteration)

            # ===== Phase 1: Task Selection =====
            # Task Pool에서 entropy 기반 샘플링
            selected_tasks = self.task_pool.sample_tasks(
                n=tasks_per_iter,
                strategy="entropy"
            )

            # ===== Phase 2: Data Generation =====
            dataset = []
            for task in selected_tasks:
                task_id = hash_task(task)
                for _ in range(rollouts_per_task):
                    # Actor로 rollout
                    trajectory = self.rollout(task)

                    # Reward 계산
                    rewards = self.compute_rewards(trajectory, task)

                    dataset.append({
                        'task': task,
                        'task_id': task_id,
                        'trajectory': trajectory,
                        **rewards
                    })

                    # Task Pool 통계 업데이트
                    self.task_pool.update_stats(task_id, rewards['trajectory_reward'])

            # ===== Phase 3: Policy Training =====
            self.actor = self.grpo_update(dataset)

            # ===== Logging =====
            self.log_metrics(iteration, dataset)
            self.log_curriculum_stats(self.task_pool.get_curriculum_stats())

    def rollout(self, task):
        """Actor와 World Model로 trajectory 생성"""
        trajectory = []
        state = task.initial_state

        for step in range(task.max_steps):
            # Actor가 행동 선택 (task의 available_tools 기반)
            action = self.actor.generate_action(
                state=state,
                task_description=task.description,
                available_tools=task.available_tools  # Tool set 주입
            )

            # World Model이 다음 상태 예측 (동일한 tool set 공유)
            next_state = self.world_model.predict_next_state(
                state=state,
                action=action,
                task=task  # task.available_tools 포함
            )

            trajectory.append({
                'state': state,
                'action': action,
                'next_state': next_state
            })

            state = next_state

            if self.is_terminal(state, task):
                break

        return trajectory

    def select_tasks_by_entropy(self, candidates, k_rollouts, top_n):
        """Reward entropy 기반 태스크 선택"""
        # (구현 상세는 3.3.2 참조)
        ...

    def compute_rewards(self, trajectory, task):
        """Step-level 및 trajectory-level reward 계산"""
        # (구현 상세는 3.4 참조)
        ...

    def grpo_update(self, dataset):
        """Offline GRPO 정책 업데이트"""
        # (구현 상세는 3.5 참조)
        ...
```

---

## 5. 실험 계획

### 5.1 기본 실험

| 실험 | 목적 |
|------|------|
| Baseline | Post-training 없는 Actor 성능 |
| + Curiosity Task Gen | 호기심 기반 태스크 생성 효과 |
| + Reward Entropy | Entropy 필터링 추가 효과 |
| + Step Reward | Step-level reward 활용 효과 |
| Full Method | 전체 방법론 성능 |

### 5.2 Ablation Study

```
1. Task Generation
   ├── Random task vs Curiosity-based
   ├── w/o Entropy filtering vs w/ Entropy filtering
   └── Entropy threshold 민감도

2. Reward Design
   ├── Trajectory-level only vs Step-level
   ├── AND 연산 vs 부분 점수 (sum/len)
   └── World Model 판단 정확도

3. Training
   ├── GRPO vs PPO vs REINFORCE
   └── Step-level 정보 활용 여부
```

### 5.3 Offline vs Online 비교 실험

**실험 목표**: Online이 추가 복잡도와 비용을 justify하는가?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        실험 설계                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  고정 변수:                                                                  │
│  ├── 동일한 Actor 초기 모델                                                  │
│  ├── 동일한 World Model (Frontier LRM)                                      │
│  ├── 동일한 Task Distribution                                               │
│  └── 동일한 Test Set                                                        │
│                                                                             │
│  비교 조건:                                                                  │
│  ├── Offline: 데이터 N개 생성 → K epochs 학습                               │
│  ├── Online:  iteration당 데이터 n개 × M iterations                         │
│  └── 총 API 호출량 동일하게 맞춤 (N ≈ n × M)                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**측정 지표**:
```
1. Test Set Reward (주요 지표)
   └── 동일 test task에서 성공률

2. Sample Efficiency
   └── 목표 성능 도달까지 필요한 trajectory 수

3. Wall-clock Time
   └── 총 학습 시간 (API 대기 포함)

4. API Cost
   └── World Model 호출 비용

5. 학습 안정성
   └── Reward variance across runs
```

**예상 결과**:
```
Test Set Reward
     │
     │                              ●─── Online
     │                         ●───●
     │                    ●───●
     │               ●───●
     │          ●───●───────────────●─── Offline
     │     ●───●
     │●───●
     └──────────────────────────────────► Training Progress

Gap 분석:
├── 초기: 비슷 (데이터 부족)
├── 중기: Online 우위 시작 (on-policy 이점)
└── 후기: Gap 유지 또는 확대
```

**세부 실험**:

| 실험 | 설명 | 목적 |
|------|------|------|
| 5.3.1 | Pure Offline vs Pure Online | 기본 비교 |
| 5.3.2 | Offline → Online (2단계) | 하이브리드 효과 |
| 5.3.3 | Iteration당 rollout 수 변화 | On-policy 배치 크기 최적화 |
| 5.3.4 | Task 난이도별 Gap | 쉬운/어려운 task에서 차이 |
| 5.3.5 | 데이터 규모별 Gap | 데이터 많을수록 차이 감소? |

**분석 질문**:
```
1. Online이 항상 더 좋은가?
   └── 어떤 조건에서 Offline으로 충분한가?

2. Gap은 어느 정도인가?
   └── +5%? +10%? +20%?

3. 비용 대비 가치가 있는가?
   └── Online 추가 비용 vs 성능 향상

4. Offline의 ceiling은 어디인가?
   └── 데이터 늘려도 더 이상 안 오르는 지점
```

---

## 6. 기대 효과 및 Contribution

### 6.1 기술적 Contribution

```
1. Frozen Frontier LRM as World Model
   └── 학습 비용 제거
   └── 자동 성능 개선 (frontier 발전 따라)

2. 경험 메모리 제거 + Domain Context 주입
   └── Inductive bias 문제 회피
   └── 시스템 단순화

3. 호기심 + Reward Entropy 태스크 선택
   └── 자동 curriculum 형성
   └── 학습 효율 극대화

4. Step-level Binary Reward + AND
   └── 명확한 실패 지점 식별
   └── 자동 credit assignment
```

### 6.2 실용적 장점

```
1. 구현 단순
   └── 복잡한 World Model 학습 불필요
   └── Value function 학습 불필요

2. 비용 효율
   └── API 호출 = 데이터 생성 단계에 집중
   └── 학습 = GPU만 사용

3. 확장 용이
   └── Domain context만 바꾸면 새 도메인 적용
   └── Actor 모델 교체 용이
```

---

## 7. 참고 문헌 및 관련 연구

### 7.1 핵심 참고 논문

| 논문 | 채용한 아이디어 |
|------|----------------|
| **DreamGym** (2511.03773) | Reasoning-based World Model, Reward Entropy Curriculum |
| **AgentEvolver** (2511.10395) | Curiosity-based Task Generation (Self-Questioning) |
| **DeepSeek GRPO** | Group Relative Policy Optimization |

### 7.2 관련 연구

- **RLHF**: Post-training RL 패러다임
- **World Models**: Environment simulation
- **Curriculum Learning**: 난이도 기반 학습
- **Process Reward Models**: Step-level reward

---

## 8. 요약

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Method Summary                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Title:    LRMs are World Models                                            │
│  Subtitle: Agent RLAIF without World Model Training                         │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  목표:     오픈소스 Agentic LRM의 post-training 강화                         │
│                                                                             │
│  Actor:    오픈소스 Agentic LRM (학습 대상)                                  │
│                                                                             │
│  World:    Frontier LRM (frozen) + Domain Context                           │
│            • 학습 없음, API 호출만                                           │
│            • 경험 메모리 검색 제거                                           │
│                                                                             │
│  Task:     호기심 기반 생성 + Reward Entropy 필터링                          │
│            • 다양성 확보 + 학습 효과 최적화                                   │
│            • 자동 curriculum 형성                                            │
│                                                                             │
│  Reward:   Step-level 0/1 판단 + AND 연산                                   │
│            • 실패 지점 명확                                                  │
│            • 자동 credit assignment                                         │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Training Strategy:                                                         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │   Phase 1: Offline GRPO (먼저 구현)                                 │   │
│  │   ├── 데이터 생성 → 학습 분리                                       │   │
│  │   ├── 구현 간단, 빠른 iteration                                     │   │
│  │   └── Baseline 성능 확립                                            │   │
│  │                                                                     │   │
│  │                         ▼                                           │   │
│  │                                                                     │   │
│  │   Phase 2: Online GRPO (확장)                                       │   │
│  │   ├── 실시간 rollout + 학습 (on-policy)                             │   │
│  │   ├── 매 iteration 현재 정책 데이터만 사용                           │   │
│  │   ├── (선택) Async rollout workers                                  │   │
│  │   └── 더 높은 최종 성능 기대                                         │   │
│  │                                                                     │   │
│  │                         ▼                                           │   │
│  │                                                                     │   │
│  │   비교 실험: Offline vs Online                                      │   │
│  │   ├── 동일 API budget에서 성능 비교                                 │   │
│  │   ├── Gap 정량화                                                    │   │
│  │   └── 비용 대비 가치 분석                                            │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**핵심 Insight**:
```
"LRMs are World Models"

Large Reasoning Model은 이미 충분한 세계 지식과 추론 능력을 갖추고 있어,
별도의 World Model 학습 없이 Text Environment Simulator로 활용할 수 있다.

이를 통해 Agent RLAIF를 단순하고 효율적으로 수행할 수 있다.
```

---

*문서 작성일: 2024-11*
*버전: 1.1 (Online 방법 추가)*
