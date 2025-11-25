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
    Task: {task}
    State: {state}
    Action: {action}

    [Instructions]
    1. 위 행동의 결과로 나타날 다음 상태를 예측하시오.
    2. 이 행동이 태스크 목표에 부합하는 올바른 행동인지 판단하시오. (0 또는 1)

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

**역할**: 다양한 태스크 후보 생성

```python
def generate_task_candidates(domain_context, num_candidates=100):
    """
    도메인 context 기반으로 다양한 태스크 후보 생성
    """
    prompt = f"""
    [Domain]
    {domain_context}

    [Instructions]
    이 도메인에서 agent가 학습하면 좋을 다양한 태스크를 생성하시오.

    다양성 기준:
    - 난이도: 쉬움, 중간, 어려움 균형
    - 상황: 일반적 케이스 + 엣지 케이스
    - 목표 유형: 탐색, 조작, 복합 목표 등

    기존에 생성된 태스크와 겹치지 않는 새로운 상황을 탐색하시오.
    """
    return task_generator.generate(prompt, n=num_candidates)
```

#### 3.3.2 Reward Entropy 기반 필터링

**역할**: 학습 효과가 높은 태스크 선택

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
│  Online:                                                                    │
│  ─────────────────────────────────────────────────────────                  │
│  [Data Gen] ──► [Train] ──► [Data Gen] ──► [Train] ──► ... ──► [Done]      │
│       │            │             │             │                            │
│       └── Actor ───┘             └── Actor ────┘                            │
│           v1                         v2                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Online GRPO 알고리즘**:
```python
def online_grpo_training(actor, world_model, task_generator,
                         num_iterations, rollouts_per_iter):
    """
    Online GRPO 학습
    """
    replay_buffer = ReplayBuffer(max_size=10000)

    for iteration in range(num_iterations):
        # ===== Phase 1: Task Selection =====
        task_candidates = task_generator.generate_candidates(n=50)
        selected_tasks = select_tasks_by_entropy(
            task_candidates, actor, world_model,
            k_rollouts=3, top_n=10
        )

        # ===== Phase 2: Online Data Generation =====
        # 현재 Actor로 새 데이터 생성
        new_trajectories = []
        for task in selected_tasks:
            for _ in range(rollouts_per_iter // len(selected_tasks)):
                traj = rollout(actor, world_model, task)
                rewards = compute_rewards(traj, task)
                new_trajectories.append({
                    'task': task,
                    'trajectory': traj,
                    **rewards
                })

        # Replay buffer에 추가
        replay_buffer.add(new_trajectories)

        # ===== Phase 3: Policy Update =====
        # 새 데이터 + 버퍼에서 샘플링 (혼합 비율 조절 가능)
        batch = sample_batch(
            new_data=new_trajectories,
            buffer=replay_buffer,
            new_ratio=0.7,  # 새 데이터 70%
            buffer_ratio=0.3  # 버퍼 30%
        )

        # GRPO 업데이트
        actor = grpo_update(actor, batch)

        # ===== Logging =====
        log_metrics(iteration, new_trajectories)

    return actor
```

**핵심 구성 요소**:

#### 3.6.1 Replay Buffer

```python
class ReplayBuffer:
    """
    Online 학습을 위한 경험 버퍼
    """
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, trajectories):
        for traj in trajectories:
            self.buffer.append(traj)

    def sample(self, n):
        return random.sample(self.buffer, min(n, len(self.buffer)))

    def sample_by_task(self, task, n):
        """같은 task의 trajectory만 샘플링 (GRPO 비교용)"""
        same_task = [t for t in self.buffer if t['task'] == task]
        return random.sample(same_task, min(n, len(same_task)))
```

#### 3.6.2 혼합 샘플링 전략

```
새 데이터 vs 버퍼 데이터 비율:

Early training:   new 90% / buffer 10%
  → 빠른 탐색, 다양성 확보

Mid training:     new 70% / buffer 30%
  → 균형 잡힌 학습

Late training:    new 50% / buffer 50%
  → 안정적 수렴, 과거 좋은 경험 활용
```

```python
def sample_batch(new_data, buffer, new_ratio, buffer_ratio, batch_size=64):
    """
    새 데이터와 버퍼를 혼합하여 배치 구성
    """
    n_new = int(batch_size * new_ratio)
    n_buffer = int(batch_size * buffer_ratio)

    batch = []
    batch += random.sample(new_data, min(n_new, len(new_data)))
    batch += buffer.sample(n_buffer)

    return batch
```

#### 3.6.3 Async Rollout (고급)

```
동기 방식 (Simple):
┌─────────────────────────────────────────────────┐
│  Rollout → Train → Rollout → Train → ...       │
│                                                 │
│  단점: GPU idle time 발생                        │
└─────────────────────────────────────────────────┘

비동기 방식 (Advanced):
┌─────────────────────────────────────────────────┐
│  Rollout Worker 1 ──────────┐                   │
│  Rollout Worker 2 ──────────┼──► Buffer ──► Train
│  Rollout Worker 3 ──────────┘       ↑          │
│                                     │          │
│                              continuously      │
│                                                 │
│  장점: GPU 활용도 최대화                         │
└─────────────────────────────────────────────────┘
```

```python
# 비동기 Online 학습 (개념)
class AsyncOnlineTrainer:
    def __init__(self, actor, world_model, num_workers=4):
        self.actor = actor
        self.world_model = world_model
        self.buffer = SharedReplayBuffer()
        self.num_workers = num_workers

    async def rollout_worker(self, worker_id):
        """비동기 rollout 워커"""
        while not self.done:
            task = self.task_generator.sample()
            traj = await self.async_rollout(task)
            self.buffer.add(traj)

    async def train_loop(self):
        """학습 루프 (버퍼에서 지속적으로 샘플링)"""
        while not self.done:
            if len(self.buffer) >= min_buffer_size:
                batch = self.buffer.sample(batch_size)
                self.actor = grpo_update(self.actor, batch)

    def train(self):
        """메인 학습 함수"""
        # 워커들과 학습 루프 동시 실행
        asyncio.gather(
            *[self.rollout_worker(i) for i in range(self.num_workers)],
            self.train_loop()
        )
```

**Online 구현 시 고려사항**:
```
1. API Rate Limiting
   └── World Model API 호출 제한 고려
   └── 배치 요청, 캐싱 등 최적화

2. Actor 버전 관리
   └── Rollout 중 Actor 업데이트 시 버전 불일치
   └── 버전 태깅 또는 주기적 동기화

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
├── Online으로 추가 학습
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
                 domain_context):
        # Actor: 학습 대상 (오픈소스 Agentic LRM)
        self.actor = load_model(actor_model_path)

        # World Model: Frozen Frontier LRM
        self.world_model = FrozenWorldModel(
            api=world_model_api,
            domain_context=domain_context
        )

        # Task Generator
        self.task_generator = TaskGenerator(domain_context)

    def train(self, num_iterations, tasks_per_iter, rollouts_per_task):
        for iteration in range(num_iterations):
            # ===== Phase 1: Task Generation =====
            # 호기심 기반 태스크 후보 생성
            task_candidates = self.task_generator.generate_candidates(n=100)

            # Reward entropy로 필터링
            selected_tasks = self.select_tasks_by_entropy(
                task_candidates,
                k_rollouts=5,
                top_n=tasks_per_iter
            )

            # ===== Phase 2: Data Generation =====
            dataset = []
            for task in selected_tasks:
                for _ in range(rollouts_per_task):
                    # Actor로 rollout
                    trajectory = self.rollout(task)

                    # Reward 계산
                    rewards = self.compute_rewards(trajectory, task)

                    dataset.append({
                        'task': task,
                        'trajectory': trajectory,
                        **rewards
                    })

            # ===== Phase 3: Policy Training =====
            self.actor = self.grpo_update(dataset)

            # Logging
            self.log_metrics(iteration, dataset)

    def rollout(self, task):
        """Actor와 World Model로 trajectory 생성"""
        trajectory = []
        state = task.initial_state

        for step in range(task.max_steps):
            # Actor가 행동 선택
            action = self.actor.generate_action(state, task)

            # World Model이 다음 상태 예측
            next_state = self.world_model.predict_next_state(state, action, task)

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
| 5.3.3 | 다양한 혼합 비율 | 최적 new/buffer 비율 탐색 |
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
│  │   ├── 실시간 rollout + 학습                                         │   │
│  │   ├── Replay buffer + 혼합 샘플링                                   │   │
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
