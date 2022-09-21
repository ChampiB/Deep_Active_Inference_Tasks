from pynput import keyboard
import hydra
from omegaconf import OmegaConf, open_dict
from environments import EnvFactory
from environments.wrappers.DefaultWrappers import DefaultWrappers

action = -1


def on_press(key):
    global action
    try:
        if key.char == 'w':  # if key 'q' is pressed
            action = 1
        if key.char == 's':
            action = 0
        if key.char == 'd':
            action = 3
        if key.char == 'a':
            action = 2
    except AttributeError:
        print('special key pressed: {0}'.format(key))


def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False
    return None


@hydra.main(config_path="config", config_name="training")
def main(config):
    global action
    # Create the environment and apply standard wrappers.
    env = EnvFactory.make(config)
    with open_dict(config):
        config.env.n_actions = env.action_space.n
    env = DefaultWrappers.apply(env, config["images"]["shape"])

    # Retrieve the initial observation from the environment.
    env.reset()

    # Play.
    steps_done = 0
    while steps_done < 1000:

        # Collect events until released
        action = -1
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
        if action == -1:
            continue

        # Execute the action in the environment.
        _, reward, done, _ = env.step(action)

        # Reset the environment when a trial ends.
        if done:
            env.reset()

        # Increase the number of steps done.
        steps_done += 1

    # Close the environment.
    env.close()


if __name__ == '__main__':
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))
    main()
