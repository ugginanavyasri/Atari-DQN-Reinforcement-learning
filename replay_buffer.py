import random
from collections import deque
import numpy as np


class ReplayBuffer:


    def __init__(self, capacity):

        self.memory = deque(
            maxlen=capacity
        )


    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )


    def sample(self, batch_size):

        batch = random.sample(
            self.memory,
            batch_size
        )


        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []


        for experience in batch:

            state, action, reward, next_state, done = experience


            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)


        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )


    def size(self):

        return len(self.memory)