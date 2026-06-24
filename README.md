# Atari Reinforcement Learning Agent using DQN

## Overview

Implemented an AI agent that learns Atari games using Deep Q Network (DQN).

The agent learns by interacting with the environment and improving actions based on rewards.

## Technologies Used

- Python
- PyTorch
- Gymnasium
- Atari ALE
- OpenCV
- NumPy

## Reinforcement Learning Concepts Implemented

- Deep Q Network (DQN)
- Double DQN
- Experience Replay Buffer
- Epsilon Greedy Exploration
- Target Network
- Frame Stacking

## Project Structure

train.py
- Trains the AI agent

test_ai.py
- Runs the trained AI agent

dqn.py
- Neural network architecture

environment.py
- Atari environment setup

replay_buffer.py
- Stores experiences

frame_stack.py
- Combines multiple frames


## How to Run

Install dependencies:

pip install -r requirements.txt


Train the model:

python train.py


Test the trained agent:

python test_ai.py





## Demo

AI agent playing Atari game:

![AI Gameplay](assets/reinforcement_learning_demo_video.gif)