from environments import EnvFactory
from environments.wrappers.DefaultWrappers import DefaultWrappers
from singletons.Logger import Logger
import hydra
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
import numpy as np
import random
import torch
from agents.save.Checkpoint import Checkpoint


def create_wrapped_env(config, i, env, env_name):
    """
    Create an environment wrapped with default wrappers
    :param config: the training configuration
    :param i: the environment index
    :param env: the string describing the environment to create
    :param env_name: the environment name
    """
    # Update environment and its name in configuration
    with open_dict(config):
        config.env = env
        config.env.name = env_name
    # Create the environment from the configuration
    env = EnvFactory.make(config)
    # Update number of actions
    with open_dict(config):
        if i == 0:
            config.env.n_actions = env.action_space.n
        else:
            config.env.n_actions = max(config.env.n_actions, env.action_space.n)
    # Apply environment wrappers
    return DefaultWrappers.apply(env, config["images"]["shape"])


@hydra.main(config_path="config", config_name="training")
def train(config):
    # Set the seed requested by the user.
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # Create the logger and keep track of the configuration.
    Logger.get(name="Training").info("Configuration:\n{}".format(OmegaConf.to_yaml(config)))

    # Create the environment and apply standard wrappers.
    env_names = [
        ("openai", "ALE/VideoPinball-v5"),
        ("openai", "ALE/Boxing-v5"),
        ("openai", "ALE/Breakout-v5"),
        ("openai", "ALE/StarGunner-v5"),
        ("openai", "ALE/Robotank-v5"),
        ("openai", "ALE/Atlantis-v5"),
        ("openai", "ALE/CrazyClimber-v5"),
        ("openai", "ALE/Gopher-v5"),
        ("openai", "ALE/DemonAttack-v5"),
        ("openai", "ALE/NameThisGame-v5"),
        ("openai", "ALE/Krull-v5"),
        ("openai", "ALE/Assault-v5"),
        ("openai", "ALE/RoadRunner-v5"),
        ("openai", "ALE/Kangaroo-v5"),
        ("openai", "ALE/Jamesbond-v5"),
        ("openai", "ALE/Tennis-v5"),
        ("openai", "ALE/Pong-v5"),
        ("openai", "ALE/SpaceInvaders-v5"),
        ("openai", "ALE/BeamRider-v5"),
        ("openai", "ALE/Tutankham-v5"),
        ("openai", "ALE/KungFuMaster-v5"),
        ("openai", "ALE/Freeway-v5"),
        ("openai", "ALE/TimePilot-v5"),
        ("openai", "ALE/Enduro-v5"),
        ("openai", "ALE/FishingDerby-v5"),
        ("openai", "ALE/UpNDown-v5"),
        ("openai", "ALE/IceHockey-v5"),
        ("openai", "ALE/Qbert-v5"),
        ("openai", "ALE/Hero-v5"),
        ("openai", "ALE/Asterix-v5"),
        ("openai", "ALE/BattleZone-v5"),
        ("openai", "ALE/WizardOfWor-v5"),
        ("openai", "ALE/ChopperCommand-v5"),
        ("openai", "ALE/Centipede-v5"),
        ("openai", "ALE/BankHeist-v5"),
        ("openai", "ALE/Riverraid-v5"),
        ("openai", "ALE/Zaxxon-v5"),
        ("openai", "ALE/Amidar-v5"),
        ("openai", "ALE/Alien-v5"),
        ("openai", "ALE/Venture-v5"),
        ("openai", "ALE/Seaquest-v5"),
        ("openai", "ALE/DoubleDunk-v5"),
        ("openai", "ALE/Bowling-v5"),
        ("openai", "ALE/MsPacman-v5"),
        ("openai", "ALE/Asteroids-v5"),
        ("openai", "ALE/Frostbite-v5"),
        ("openai", "ALE/Gravitar-v5"),
        ("openai", "ALE/PrivateEye-v5"),
        ("openai", "ALE/MontezumaRevenge-v5"),
        ("EpistemicSprites", "EpistemicSprites"),
    ]
    envs = [create_wrapped_env(config, i, env, env_name) for i, (env, env_name) in enumerate(env_names)]

    # Create the agent and train it.
    archive = Checkpoint(config["agent"]["tensorboard_dir"], config["checkpoint"]["file"])
    agent = archive.load_model() if archive.exists() else instantiate(config["agent"])
    agent.multi_train(envs, config, [env_name for _, env_name in env_names])


if __name__ == '__main__':
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Train the agent.
    train()
