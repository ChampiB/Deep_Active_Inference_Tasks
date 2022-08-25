seed=148
# Run CHMM[reward] and CHMM[efe]
g_values='"reward" "efe_3"'
for g_value in $g_values; do
  sbatch -p gpu --mem=10G --gres-flags=disable-binding --gres=gpu all_env_training.sh agent="CHMM" seed=$seed agent.g_value=$g_value
  seed=$((seed+1))
done
