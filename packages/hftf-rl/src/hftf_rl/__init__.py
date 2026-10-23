"""hftf_rl: PPO training (M2-M3).

Built (M2): config (PPOConfig), policies (MLPActorCritic, Beta pace head), buffer (rollouts + GAE),
normalize (observation / reward scaling), ppo (PPOTrainer, checkpoints, PacePolicy, load_policy).
Planned (M3, docs/ARCHITECTURE.md §5-6): FuzzyActorCritic, EvolutionCallback.
"""
