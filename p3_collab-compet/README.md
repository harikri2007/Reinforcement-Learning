# Project 3: Collaboration and Competition

## Problem Description

In this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. Two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. The goal is to keep the ball in play as long as possible.

## State & Action Space

- **State Space:** 8 variables per agent, corresponding to the position and velocity of the ball and racket. Each agent receives its own local observation.
- **Action Space:** Each action is a vector of two continuous values, corresponding to movement toward/away from the net and jumping. Each entry in the action vector should be a number between -1 and 1.

## Criteria for Solving

The environment is considered solved when the agents achieve an average score of +0.5 over 100 consecutive episodes (taking the maximum score over both agents for each episode).

## Solution Algorithm

This project uses the **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** algorithm to solve the collaboration and competition task. MADDPG extends DDPG to multi-agent environments, allowing agents to learn coordinated policies using centralized critics and decentralized actors.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/harikri2007/Reinforcement-Learning.git
   cd Reinforcement-Learning/p3_collab-compet
   ```
2. **Set up the Python environment:**
   ```bash
   python3 -m venv drlnd-env
   source drlnd-env/bin/activate
   pip install -r requirements.txt
   ```

## Download the Environment

Download the Tennis environment for your operating system:

- **Linux:** [Tennis_Linux.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- **Mac OSX:** [Tennis.app.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- **Windows (32-bit):** [Tennis_Windows_x86.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- **Windows (64-bit):** [Tennis_Windows_x86_64.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

(_For Windows users_) [How to determine your Windows version](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)

(_For AWS_) If training on AWS (without a virtual screen), use [Tennis_Linux_NoVis.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip).

Place the downloaded file in the `p3_collab-compet/` folder and unzip it.

## Running the Notebook

1. Activate your Python environment:
   ```bash
   source drlnd-env/bin/activate
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Tennis.ipynb
   ```
3. Follow the instructions in `Tennis.ipynb` to train and evaluate your agents.
