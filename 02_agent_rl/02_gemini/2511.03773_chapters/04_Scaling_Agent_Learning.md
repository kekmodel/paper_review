# 4 Scaling Agent Learning via Experience Synthesis

To synthesize diverse agent experiences for RL training, DreamGym is built around three key components: (1) a scalable reasoning experience model that encodes the meta-dynamics of the target domain to efficiently generate informative trajectories; (2) an experience replay buffer that integrates offline environment knowledge with online synthetic transitions, co-evolving with the agent to stay aligned with its updated policy; (3) a curriculum task generator that produces progressively challenging variations of high-value tasks selected via a reward-entropy heuristic. We elaborate each component in the following sections.

## 4.1 Building Reasoning Experience Models for Agent Learning
For effective RL training, instead of relying on heterogeneous external environments that are costly to interact with and difficult to control, DreamGym adopts a more adaptive and controllable approach by building a LLM-based experience model that can efficiently interact with the agent over multiple turns to generate diverse experiences with consistent outcomes and rich feedback signals for learning.

Unlike prior data-hungry and costly approaches that build world models to replicate the real world in raw pixel spaces, we design an efficient reasoning experience model, denoted as $\mathcal{M}_{\text{exp}}$, that operates in an abstract, meta-representational textual space $\mathcal{S}$. The key insight is that synthesizing transitions in this abstract state space can reduce irrelevant dimensions and produce trajectories that are more informative and token-efficient than those derived from raw observations. For example, in a web shopping task, instead of processing raw HTML code, the experience model directly synthesizes clean element listings while discarding irrelevant structural artifacts such as headers and tags. This state-space design makes training the experience model highly sample-efficient, requiring only small pubic trajectory datasets in our experiments, while also enhancing the effectiveness of agent learning.

### 4.1.1 Inference for Experience Rollout Collection
Notably, we find that beyond the current state-action pair, three additional contexts are important for improving state quality: (1) interaction history {$(s_i, a_i)$}$_{t=0}^{T}$, which incorporates the past trajectory in the context window to help maintain state consistency across multiple turns; (2) task instruction $\tau$, which conditions the experience model on the current goal, enabling it to interpret actions w.r.t. task objectives and thereby predict both state transitions and rewards more accurately; (3) past experiences, which are top-k demonstrations {$(d_j)$}$_{j=1}^k$ retrieved from the replay buffer based on semantic similarity with the state-action pair, i.e., {$(d_j)$}$_{j=1}^k = \text{Top}_k(\text{cos}(\phi(s_t, a_t), \phi(s_i, a_i)))$, where $\phi(\cdot)$ denotes an arbitrary semantic encoder. Leveraging knowledge this way reduces hallucinations and improves factuality for knowledge-intensive state predictions. Therefore, given these inputs, the experience model predicts the next state $s_{t+1}$ and reward $r_{t+1}$ via chain-of-thought (CoT) (Wei et al., 2022):

$$
(s_{t+1}, r_{t+1}) = \mathcal{M}_{\text{exp}}\left(R_t \mid \{(s_i, a_i)\}_{i=0}^t, \{d_j\}_{j=1}^k, \tau \right).
$$

where $R_t$ is an explicit reasoning trace produced by the experience model that guides the state transition. With such reasoning, it predicts the most consistent and informative transition and feedback that reflects the consequence of the agent action. For example, if the action is invalid, it transitions to a failure state and assigns a zero reward to signal the error, and vice versa. In our experiments, following (Feng et al., 2025), we adopt an outcome-based reward scheme, assigning $r = 1$ only at the final step when the task is successfully completed and $r = 0$ in all other cases.

### 4.1.2 Training Experience Models to Reason
Benefiting from the abstract state-space design, training the experience model is highly sample-efficient and requires only limited data from the real environment. In practice, abundant offline trajectory datasets from public benchmarks such as the WebArena Leaderboard are sufficient for training. Our experience model distills such offline knowledge and then serves as a bridge to interact with the agent online for RL training.

Concretely, given a trajectory dataset $\mathcal{D} = \{(s_t, a_t, s_{t+1}, r_{t+1})\}$, each transition is annotated with an explicit reasoning trace $R^*_t$ by LLM (prompt shown in Appendix C.1), which explains why the action $a_t$ taken in state $s_t$ consequently leads to the next state $s_{t+1}$ and reward $r_{t+1}$ given the available contexts. To distill this knowledge, we train $\mathcal{M}_{\text{exp}}$ via SFT with a joint objective over reasoning generation and next-state prediction:

$$ 
\mathcal{L}_{\text{SFT}} = \mathbb{E}_{(s_t, a_t, s_{t+1}, R^*_t) \sim \mathcal{D}} \left[ - \log P_\theta(R^*_t \mid s_t, a_t, \mathcal{H}_t, \mathcal{D}_k) - \log P_\theta(s_{t+1} \mid s_t, a_t, R^*_t, \mathcal{H}_t, \mathcal{D}_k) \right],
$$ 

where $\mathcal{H}_t$ denotes the interaction history, $\mathcal{D}_k$ denotes the retrieved top-k demonstrations, and $\theta$ denotes the parameters of $\mathcal{M}_{\text{exp}}$. This objective ensures that the model (i) learns to generate faithful reasoning traces that explain the causal effect of an action, and (ii) leverages these traces to predict consistent and informative next states. By doing so, the experience model not only imitates expert trajectories but also acquires the ability to generalize reasoning for novel rollouts during RL training.

## 4.2 Curriculum-based Task Generation
Diverse, curriculum-aligned task instructions are important for RL agents to acquire knowledge (Zhou et al., 2025). However, scaling task collections is costly, as it requires significant human effort to verify the feasibility of each task in the target environment. DreamGym inherently alleviates this burden by adapting to arbitrary new tasks within the target domain through synthetic multi-turn transitions. Building on this capability, we propose curriculum-based task generation, where the same experience model actively generates new tasks as variations of a set of $m$ seed tasks:

$$ 
\tau_t = \mathcal{M}_{\text{task}}(\{\tau_{t-1}^i\}_{i=1}^m),
$$ 

where $\mathcal{M}_{\text{task}}$ shares parameters with $\mathcal{M}_{\text{exp}}$. Specifically, the seed tasks are chosen based on two criteria: (1) they are sufficiently challenging for the current agent policy, thereby maximizing information gain; (2) they are well-defined, such that unrealistic or malformed tasks can be discarded.

To satisfy both conditions, we introduce a group-based reward entropy as a criteria for selecting high-quality and challenging tasks. Formally, for a task $\tau$, we define its value

$$ 
\mathcal{V}_\tau = \frac{1}{n} 
\sum_{i=1}^n (r^i - \bar{r})^2, \quad \text{where } \bar{r} = 
\frac{1}{n} 
\sum_{i=1}^n r^i,
$$ 

where $r^i$ are the outcome rewards from $n$ rollouts of task $\tau$ within the group $\mathcal{G}$. For GRPO, $\mathcal{G}$ can simply be the training group, while for PPO, tasks can be first clustered using a semantic embedder, and each cluster essentially forms a group $\mathcal{G}$ from which task variations can be generated. Notably, a non-zero variance in $\mathcal{G}$ indicates that the agent observes both successes and failures on the task, signaling that the task is feasible yet challenging. A task reaches maximum entropy when successes and failures are evenly balanced in $\mathcal{G}$, providing the greatest information gain for credit assignment. This observation is consistent with recent findings that LLMs learn most effectively from tasks of intermediate difficulty (Gao et al., 2025). Thus, by feeding such high-entropy tasks into $\mathcal{M}_{\text{task}}$, we generate progressively more challenging variations to enhance agent exploration and knowledge acquisition.

To stabilize training, we introduce a hyperparameter $\lambda$ that bounds the proportion of synthetic tasks sampled per iteration. This preserves sufficient coverage of the original task distribution while directing exploration toward the current policy’s weakness regions for curriculum-based improvement.

## 4.3 Learning from Synthetic Experiences
**Policy training in synthetic environments.** As shown in Fig. 2, DreamGym begins with a seed task set and generates multi-turn rollouts for each task by alternating between the agent policy, which selects actions from states, and the experience model, which predicts next states conditioned on the agent action, history, and task context (as in §4.1.1). The collected rollouts are used with standard RL algorithms (as in §3.2) to update the policy. After each iteration, the experience model augments the task set by generating variations of challenging tasks with high reward entropy (as in §4.2). This cycle of interaction, training, and curriculum expansion continues until convergence or a predefined training budget is reached. Furthermore, we provide an analytical lower bound of the policy improvement in real environments when training with purely synthetic experiences from DreamGym under trust-region assumptions, as detailed in Appendix B.1.
