# Project 2: Continuous Control

## Problem Description

In this project, we will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. A double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. The goal is to maintain the agent's position at the target location for as many time steps as possible.

## State & Action Space

- **State Space:** 33 variables, including position, rotation, velocity, and angular velocities of the arm.
- **Action Space:** Each action is a vector of four numbers, corresponding to torque applicable to two joints. Each entry in the action vector should be a number between -1 and 1.

## Criteria for Solving

- **Single Agent Version:** The environment is considered solved when the agent achieves an average score of +30 over 100 consecutive episodes.

## Solution Algorithm

This project uses the **Deep Deterministic Policy Gradient (DDPG)** algorithm to solve the continuous control task. DDPG is an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/harikri2007/Reinforcement-Learning.git
   cd Reinforcement-Learning/p2_continuous-control
   ```
2. **Set up the Python environment:**
   ```bash
   python3 -m venv drlnd-env
   source drlnd-env/bin/activate
   pip install -r requirements.txt
   ```

## Download the Environment

Download the Reacher environment for your operating system:

- **Version 1: One (1) Agent**

  - Linux: [Reacher_Linux.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
  - Mac OSX: [Reacher.app.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
  - Windows (32-bit): [Reacher_Windows_x86.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
  - Windows (64-bit): [Reacher_Windows_x86_64.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

- **Version 2: Twenty (20) Agents**
  - Linux: [Reacher_Linux.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
  - Mac OSX: [Reacher.app.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
  - Windows (32-bit): [Reacher_Windows_x86.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
  - Windows (64-bit): [Reacher_Windows_x86_64.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

(_For Windows users_) [How to determine your Windows version](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)

(_For AWS_) If training on AWS (without a virtual screen), use [Reacher_Linux_NoVis.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [Reacher_Linux_NoVis.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2).

Place the downloaded file in the `p2_continuous-control/` folder and unzip it.

## Running the Notebook

1. Activate your Python environment:
   ```bash
   source drlnd-env/bin/activate
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Continuous_Control.ipynb
   ```
3. Follow the instructions in `Continuous_Control.ipynb` to train and evaluate your agent.
