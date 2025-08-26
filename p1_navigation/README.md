[//]: # "Image References"
[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

## Problem Description

Train an agent to navigate a large, square world and collect yellow bananas while avoiding blue bananas. The agent receives a reward of +1 for collecting a yellow banana and -1 for collecting a blue banana. The goal is to maximize the number of yellow bananas collected.

## State & Action Space

- **State Space:** 37 dimensions, including the agent's velocity and ray-based perception of objects around its forward direction.
- **Action Space:** 4 discrete actions:
  - 0: Move forward
  - 1: Move backward
  - 2: Turn left
  - 3: Turn right

## Criteria for Solving

The environment is considered solved when the agent achieves an average score of +13 over 100 consecutive episodes.

## Solution Algorithm

This project uses the **Deep Q-Network (DQN)** algorithm to solve the navigation task. DQN leverages experience replay and a target network to stabilize learning in high-dimensional state spaces.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/harikri2007/Reinforcement-Learning.git
   cd Reinforcement-Learning/p1_navigation
   ```
2. **Set up the Python environment:**
   ```bash
   python3 -m venv drlnd-env
   source drlnd-env/bin/activate
   pip install -r requirements.txt
   ```

## Download the Environment

Download the Banana environment for your operating system:

- **Linux:** [Banana_Linux.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- **Mac OSX:** [Banana.app.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- **Windows (32-bit):** [Banana_Windows_x86.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- **Windows (64-bit):** [Banana_Windows_x86_64.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

(_For Windows users_) [How to determine your Windows version](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)

(_For AWS_) If training on AWS (without a virtual screen), use [Banana_Linux_NoVis.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip).

Place the downloaded file in the `p1_navigation/` folder and unzip it.

## Running the Notebook

1. Activate your Python environment:
   ```bash
   source drlnd-env/bin/activate
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Navigation.ipynb
   ```
3. Follow the instructions in `Navigation.ipynb` to train and evaluate your agent.
