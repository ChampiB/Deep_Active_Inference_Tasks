import time

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from singletons.Logger import Logger


#
# Implement an agent playing the EpistemicSprite environment perfectly.
#
class PerfectAgent:

    def __init__(self, tensorboard_dir, **_):
        """
        Constructor
        :param tensorboard_dir: the directory in which tensorboard's files will be written
        """

        # Miscellaneous.
        self.total_rewards = 0.0
        self.tensorboard_dir = tensorboard_dir
        self.writer = SummaryWriter(tensorboard_dir)
        self.steps_done = 0

    @staticmethod
    def step(env):
        """
        Select a random action based on the critic output
        :param env: the EpistemicSprites environment
        :return: the random action
        """

        # If no hint available, go up
        if not env.display_reward_hint:
            return 1  # up

        # If hint is available
        pos_x = env.state[4]
        if env.reward_on_the_right:
            if pos_x != 31:
                return 3  # right
            else:
                return 0  # down
        else:
            if pos_x != 0:
                return 2  # left
            else:
                return 0  # down

    def train(self, env, config):
        """
        Train the agent in the gym environment passed as parameters
        :param env: the gym environment
        :param config: the hydra configuration
        :return: nothing
        """

        # Retrieve the initial observation from the environment.
        env.reset()

        # Train the agent.
        Logger.get().info("Start the training at {time}".format(time=datetime.now()))
        while self.steps_done < config["n_training_steps"]:

            # Select an action.
            action = self.step(env)

            # Execute the action in the environment.
            _, reward, done, _ = env.step(action)

            # Monitor total rewards (if needed).
            if config["enable_tensorboard"]:
                self.total_rewards += reward
                self.writer.add_scalar("Rewards", self.total_rewards, self.steps_done)

            # Reset the environment when a trial ends.
            if done:
                env.reset()

            # Increase the number of steps done.
            self.steps_done += 1

        # Close the environment.
        env.close()
