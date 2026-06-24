import gymnasium as gym
import ale_py


def create_environment():

    env = gym.make(
        "ALE/Breakout-v5",
        render_mode="human"
    )

    return env