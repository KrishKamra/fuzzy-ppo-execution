"""hftf_rl: PPO training (M2-M4).

Built (M2): config (PPOConfig), policies (MLPActorCritic, Beta pace head), buffer (rollouts + GAE),
normalize (observation / reward scaling), ppo (PPOTrainer, checkpoints, PacePolicy, load_policy).
Built (M3): FuzzyActorCritic (TSK heads over hftf_fuzzy), k-means rule init and KL-guarded structure
evolution inside PPOTrainer (events.csv), interpret (rule report).
Built (M4): EnsemblePolicy (mean pace over seeds), the hard-gate ablation (FuzzyActorCritic gate="hard"),
checkpoints that store their observation scaler (leave-one-stock-out), dominant rules for rule-regime AMI.
"""
