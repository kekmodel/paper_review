# 7 Conclusion

We introduced DreamGym, a framework that reduces the high cost of real-environment rollouts in RL for language agents by generating scalable, reasoning-driven synthetic experiences. DreamGym compresses environment dynamics into a reasoning-based experience model that produces state transitions and adaptive curricula, creating challenging yet solvable tasks tailored to the agent’s evolving policy. Experiments across diverse environments and model backbones show consistent gains in both synthetic and sim-to-real settings, driven by the synergy of reasoning-based modeling, replay-buffer grounding, and curriculum generation. More broadly, our results suggest that the key bottleneck in RL for LLM agents lies in the quality and structure of interaction data. By treating environments as generators of structured, reasoning-rich experiences rather than mere simulators, DreamGym enables more scalable, sample-efficient, and generalizable RL for agents.

# Limitations and Future Work
Our work primarily investigates single-environment learning setups, where DreamGym is applied to individual agentic scenarios. However, our framework can be further extended to build a universal world model that unifies multiple environment models, enabling knowledge transfer across environments. Such an extension could pave the way toward scaling foundational agentic models with purely synthetic experiences, capable of zero-shot adaptation to new environments.

# Acknowledgements
We thank Sergey Levine, Xiaohan Fu, Canyu Chen for their valuable feedback on methodological conceptualizations, and Licheng Yu, Lizhu Zhang, Ravi Agrawal, Zhihan Liu, and Xiyao Wang for insightful discussions and project support.
