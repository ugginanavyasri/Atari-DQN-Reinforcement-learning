import torch
from dqn import DQN


model = DQN(
    input_channels=12,
    num_actions=4
)


dummy_input = torch.randn(
    1,
    12,
    84,
    84
)


output = model(dummy_input)


print(output)
print("Output shape:", output.shape)