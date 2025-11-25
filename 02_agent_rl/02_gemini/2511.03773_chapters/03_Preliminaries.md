# 3 Preliminaries

## 3.1 Notations
We formalize the agent learning problem as a Markov Decision Process (MDP) (Bellman, 1957), defined by the tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, T, R, \gamma, \rho_0)$, where $\mathcal{S}$ denotes the state space and $\mathcal{A}$ denotes the action space. The transition function $T: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{S})$ governs the environment dynamics, where $\Delta(\mathcal{S})$ denotes the probability simplex over $\mathcal{S}$. The reward function $R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ provides feedback signals for the agent’s actions. $\gamma \in [0, 1]$ is the discount factor, and $\rho_0 \in \Delta(\mathcal{S})$ specifies the initial state distribution that includes the task instruction $\tau_0$.

In LLM agent environments, $\tau_0$ is usually a task specified by the user in natural language, and states $s \in \mathcal{S}$ encode the environment configuration visible to the agent, such as webpage content, tool outputs, or textual environment descriptions. Actions $a \in \mathcal{A}$ represent discrete operations, including clicking UI elements, invoking external tools, or generating textual responses. The agent maintains a policy $\pi_\theta: \mathcal{S} \rightarrow \Delta(\mathcal{A})$, parameterized by $\theta$, which maps states to distributions over actions.

## 3.2 Agent Learning from Experience
Given a set of online experiences where each experience $\epsilon = \{\tau_0 \mid s_0, a_0, \dots\}$ consists of a task $\tau_0$ and state-action rollout {$s_0, a_0, \dots, s_t, a_t$}, the goal of RL is to train an agent policy $\pi_\theta$ to maximize the expected cumulative reward, which typically optimized $\theta$ via policy gradient as follows:

$$ 
\nabla J(\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi_\theta} \left[ \nabla \log \pi_\theta(a_t \mid s_t) \cdot \hat{A}(s_t, a_t) \right], 
$$ 

where $\hat{A}(s_t, a_t)$ is the advantage function, estimating how favorable an action is compared to others.

**Proximal Policy Optimization (PPO).** PPO (Schulman et al., 2017) is a popular policy gradient method that improves stability by computing $\hat{A}$ with Generalized Advantage Estimation (GAE):

$$ 
\hat{A}^{\text{PPO}}_t = \sum_{l=0}^{K-1} (\gamma \lambda)^l [r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})], 
$$ 

where $V(\cdot)$ is a value function approximated by a LLM, and $\lambda$ controls the bias-variance tradeoff.

**Group Relative Policy Optimization (GRPO).** GRPO (Shao et al., 2024) extends PPO by discarding the value function and normalizing advantages within each group of responses $\mathcal{G}$ sampled for the same task instruction. Instead of GAE, the group-relative advantage is defined as:

$$ 
\hat{A}^{\text{GRPO}}_t = (r_t - \text{mean}_{i \in \mathcal{G}}(r_i)) / \text{std}_{i \in \mathcal{G}}(r_i) 
$$ 

where $r_t$ is the reward for output $o_t$, $\text{mean}_{i \in \mathcal{G}}(r_i)$ and $\text{std}_{i \in \mathcal{G}}(r_i)$ are mean and standard deviation of rewards from group $\mathcal{G}$. GRPO discards the value function and approximates the advantage using relative normalized rewards, making policy updates more scalable but potentially less sample-efficient. Notably, our proposed DreamGym is orthogonal to specific RL algorithms and focuses on scaling the synthesis of diverse, informative experiences, thereby amplifying the effectiveness of RL training.
