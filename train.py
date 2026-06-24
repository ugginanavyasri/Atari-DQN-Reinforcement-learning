import torch
import torch.nn as nn
import torch.optim as optim
import random
import cv2
import numpy as np


from environment import create_environment
from frame_stack import FrameStack
from replay_buffer import ReplayBuffer
from dqn import DQN



def preprocess(frame):

    frame = cv2.resize(
        frame,
        (84,84)
    )

    frame = frame / 255.0

    return frame



def select_action(state, model, epsilon, action_size):

    if random.random() < epsilon:

        return random.randint(
            0,
            action_size-1
        )

    else:

        state = torch.tensor(
            state,
            dtype=torch.float32
        )


        state = state.permute(
            2,0,1
        )


        state = state.unsqueeze(0)


        with torch.no_grad():

            q_values = model(state)


        return torch.argmax(q_values).item()



env = create_environment()



device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



action_size = env.action_space.n



policy_net = DQN(
    input_channels=12,
    num_actions=action_size
).to(device)



target_net = DQN(
    input_channels=12,
    num_actions=action_size
).to(device)



target_net.load_state_dict(
    policy_net.state_dict()
)


target_net.eval()


optimizer = optim.Adam(
    policy_net.parameters(),
    lr=0.0001
)



loss_function = nn.MSELoss()



memory = ReplayBuffer(
    50000
)



episodes = 100



batch_size = 32



gamma = 0.99



epsilon = 1.0



epsilon_min = 0.1



epsilon_decay = 0.995




for episode in range(episodes):


    state, info = env.reset()


    state = preprocess(state)



    stacker = FrameStack(4)


    state = stacker.reset(state)



    done = False


    total_reward = 0



    while not done:



        action = select_action(
    state,
    policy_net,
    epsilon,
    action_size
)



        next_state, reward, done, truncated, info = env.step(action)



        next_state = preprocess(next_state)


        next_state = stacker.add(next_state)



        memory.add(
            state,
            action,
            reward,
            next_state,
            done
        )



        state = next_state


        total_reward += reward




        if memory.size() > batch_size:



            states, actions, rewards, next_states, dones = memory.sample(batch_size)



            states = torch.as_tensor(states, dtype=torch.float32, device=device)
            next_states = torch.as_tensor(next_states, dtype=torch.float32, device=device)

            actions = torch.as_tensor(actions, dtype=torch.long, device=device)
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            dones = torch.as_tensor(dones, dtype=torch.float32, device=device)

            states = states.permute(0, 3, 1, 2)
            next_states = next_states.permute(0, 3, 1, 2)

            q_values = policy_net(states)


            current_q = q_values.gather(
                1,
                actions.unsqueeze(1)
            ).squeeze()


            with torch.no_grad():


                next_actions = policy_net(next_states).argmax(1)


                next_q = target_net(next_states).gather(
                    1,
                    next_actions.unsqueeze(1)
                ).squeeze()



                target_q = rewards + gamma * next_q * (1 - dones)



            loss = loss_function(
                current_q,
                target_q
            )



            optimizer.zero_grad()


            loss.backward()


            optimizer.step()




    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    target_net.load_state_dict(
    policy_net.state_dict()
)

    print(
        "Episode:",
        episode,
        "Reward:",
        total_reward,
        "Epsilon:",
        epsilon,
        "Memory:",
        memory.size()
    )

    if episode % 10 == 0:
        print("Saving checkpoint...")
        torch.save(policy_net.state_dict(), "dqn_model.pth")

torch.save(
    policy_net.state_dict(),
    "dqn_model.pth"
)

env.close()