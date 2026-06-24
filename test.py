import torch
import time


from environment import create_environment
from dqn import DQN



env = create_environment()



action_size = env.action_space.n



model = DQN(
    input_channels=12,
    num_actions=action_size
)



model.load_state_dict(
    torch.load("dqn_model.pth")
)



model.eval()



state, info = env.reset()



done = False



while not done:


    action = env.action_space.sample()


    state, reward, done, truncated, info = env.step(action)


    print(
        "Action:",
        action,
        "Reward:",
        reward
    )


    time.sleep(0.05)



env.close()