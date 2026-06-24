import torch
import time
import numpy as np

from environment import create_environment
from dqn import DQN
from frame_stack import FrameStack


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = create_environment()

action_size = env.action_space.n


model = DQN(
    input_channels=12,
    num_actions=action_size
).to(device)

model.load_state_dict(torch.load("dqn_model.pth", map_location=device))
model.eval()


def preprocess(frame):
    import cv2
    frame = cv2.resize(frame, (84, 84))
    frame = frame / 255.0
    return frame


state, info = env.reset()

stack = FrameStack(4)
state = preprocess(state)
state = stack.reset(state)

done = False

while not done:

    state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

    next_state, reward, done, truncated, info = env.step(action)

    next_state = preprocess(next_state)
    state = stack.add(next_state)

    print("Action:", action, "Reward:", reward)

    time.sleep(0.03)

env.close()