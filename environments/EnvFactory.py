import gym
from environments.dSpritesEnv import dSpritesEnv
from environments.RandomdSpritesEnv import RandomdSpritesEnv
from environments.EpistemicSpritesEnv import EpistemicSpritesEnv
from environments.MnistEnv import MnistEnv
from environments.MazeEnv import MazeEnv


def make(config):
    """
    Create the environment according to the configuration
    :param config: the hydra configuration
    :return: the created environment
    """

    # The list of custom environments.
    environments = {
        "RandomdSprites": RandomdSpritesEnv,
        "EpistemicSprites": EpistemicSpritesEnv,
        "dSprites": dSpritesEnv,
        "MNIST": MnistEnv,
        "Maze": MazeEnv
    }

    # Instantiate the environment requested by the user.
    env_name = config["env"]["name"]
    if env_name in environments.keys():
        # Custom environments
        if config["display_gui"]:
            return environments[env_name](config, render_mode="human")
        else:
            return environments[env_name](config)
    else:
        # OpenAI environments
        if config["display_gui"]:
            return gym.make(env_name, render_mode='human')
        else:
            return gym.make(env_name)
