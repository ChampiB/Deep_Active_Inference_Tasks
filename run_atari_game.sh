atari_games='"ALE/VideoPinball-v5" "ALE/Boxing-v5" "ALE/Breakout-v5" "ALE/StarGunner-v5" "ALE/Robotank-v5" "ALE/Atlantis-v5" "ALE/CrazyClimber-v5" "ALE/Gopher-v5" "ALE/DemonAttack-v5" "ALE/NameThisGame-v5" "ALE/Krull-v5" "ALE/Assault-v5" "ALE/RoadRunner-v5" "ALE/Kangaroo-v5" "ALE/Jamesbond-v5" "ALE/Tennis-v5" "ALE/Pong-v5" "ALE/SpaceInvaders-v5" "ALE/BeamRider-v5" "ALE/Tutankham-v5" "ALE/KungFuMaster-v5" "ALE/Freeway-v5" "ALE/TimePilot-v5" "ALE/Enduro-v5" "ALE/FishingDerby-v5" "ALE/UpNDown-v5" "ALE/IceHockey-v5" "ALE/Qbert-v5" "ALE/Hero-v5" "ALE/Asterix-v5" "ALE/BattleZone-v5" "ALE/WizardOfWor-v5" "ALE/ChopperCommand-v5" "ALE/Centipede-v5" "ALE/BankHeist-v5" "ALE/Riverraid-v5" "ALE/Zaxxon-v5" "ALE/Amidar-v5" "ALE/Alien-v5" "ALE/Venture-v5" "ALE/Seaquest-v5" "ALE/DoubleDunk-v5" "ALE/Bowling-v5" "ALE/MsPacman-v5" "ALE/Asteroids-v5" "ALE/Frostbite-v5" "ALE/Gravitar-v5" "ALE/PrivateEye-v5" "ALE/MontezumaRevenge-v5"'
seed=0
agent="CHMM"
g_values='"reward" "efe_3"'
for game in $atari_games; do
  # Run CHMM agents
  for g_value in $g_values; do
    sbatch -p gpu --mem=10G --gres-flags=disable-binding --gres=gpu env_training.sh agent=$agent seed=$seed agent.g_value=$g_value env=openai env.name=$game
    seed=$((seed+1))
  done
  # Run DQN agent
  sbatch -p gpu --mem=10G --gres-flags=disable-binding --gres=gpu env_training.sh agent="DQN" seed=$seed env=openai env.name=$game
  seed=$((seed+1))
done
