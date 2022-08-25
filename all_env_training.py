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


def create_wrapped_env(config, i, env_name):
    """
    Create an environment wrapped with default wrappers
    :param config: the training configuration
    :param i: the environment index
    :param env_name: the environment name
    """
    # Update environment and its name in configuration
    with open_dict(config):
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
        "ALE/VideoPinball-v5",
        "ALE/Boxing-v5",
        "ALE/Breakout-v5",
        "ALE/StarGunner-v5",
        "ALE/Robotank-v5",
        "ALE/Atlantis-v5",
        "ALE/CrazyClimber-v5",
        "ALE/Gopher-v5",
        "ALE/DemonAttack-v5",
        "ALE/NameThisGame-v5",
        "ALE/Krull-v5",
        "ALE/Assault-v5",
        "ALE/RoadRunner-v5",
        "ALE/Kangaroo-v5",
        "ALE/Jamesbond-v5",
        "ALE/Tennis-v5",
        "ALE/Pong-v5",
        "ALE/SpaceInvaders-v5",
        "ALE/BeamRider-v5",
        "ALE/Tutankham-v5",
        "ALE/KungFuMaster-v5",
        "ALE/Freeway-v5",
        "ALE/TimePilot-v5",
        "ALE/Enduro-v5",
        "ALE/FishingDerby-v5",
        "ALE/UpNDown-v5",
        "ALE/IceHockey-v5",
        "ALE/Qbert-v5",
        "ALE/Hero-v5",
        "ALE/Asterix-v5",
        "ALE/BattleZone-v5",
        "ALE/WizardOfWor-v5",
        "ALE/ChopperCommand-v5",
        "ALE/Centipede-v5",
        "ALE/BankHeist-v5",
        "ALE/Riverraid-v5",
        "ALE/Zaxxon-v5",
        "ALE/Amidar-v5",
        "ALE/Alien-v5",
        "ALE/Venture-v5",
        "ALE/Seaquest-v5",
        "ALE/DoubleDunk-v5",
        "ALE/Bowling-v5",
        "ALE/MsPacman-v5",
        "ALE/Asteroids-v5",
        "ALE/Frostbite-v5",
        "ALE/Gravitar-v5",
        "ALE/PrivateEye-v5",
        "ALE/MontezumaRevenge-v5",
        "EpistemicSprites",
    ]
    envs = [create_wrapped_env(config, i, env_name) for i, env_name in enumerate(env_names)]

    # Create the agent and train it.
    archive = Checkpoint(config["agent"]["tensorboard_dir"], config["checkpoint"]["file"])
    agent = archive.load_model() if archive.exists() else instantiate(config["agent"])
    agent.multi_train(envs, config, env_names)


if __name__ == '__main__':
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Train the agent.
    train()
