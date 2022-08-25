game="EpistemicSprites"
seed=196
# Run random agent
sbatch -p gpu --mem=10G --gres-flags=disable-binding --gres=gpu env_training.sh agent="HMM" seed=$seed env=$game
seed=$((seed+1))
# Run DQN
sbatch -p gpu --mem=10G --gres-flags=disable-binding --gres=gpu env_training.sh agent="DQN" seed=$seed env=$game
seed=$((seed+1))
# Run CHMM[reward] and CHMM[efe]
g_values='"reward" "efe_3"'
for g_value in $g_values; do
  sbatch -p gpu --mem=10G --gres-flags=disable-binding --gres=gpu env_training.sh agent="CHMM" seed=$seed agent.g_value=$g_value env=$game
  seed=$((seed+1))
done
