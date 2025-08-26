# Continuous Control

---

In this notebook, you will learn how to use the Unity ML-Agents environment for the second project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.

### 1. Start the Environment

We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).


```python
from unityagents import UnityEnvironment
import numpy as np
import torch 
from torch import nn
from collections import deque
import random
import matplotlib.pyplot as plt
import time
```

Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.

- **Mac**: `"path/to/Reacher.app"`
- **Windows** (x86): `"path/to/Reacher_Windows_x86/Reacher.exe"`
- **Windows** (x86_64): `"path/to/Reacher_Windows_x86_64/Reacher.exe"`
- **Linux** (x86): `"path/to/Reacher_Linux/Reacher.x86"`
- **Linux** (x86_64): `"path/to/Reacher_Linux/Reacher.x86_64"`
- **Linux** (x86, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86"`
- **Linux** (x86_64, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86_64"`

For instance, if you are using a Mac, then you downloaded `Reacher.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
```
env = UnityEnvironment(file_name="Reacher.app")
```


```python
env = UnityEnvironment(file_name='./Reacher_Linux_NoVis/Reacher.x86_64')
```

    Found path: /home/vijayah/deep-reinforcement-learning/p2_continuous-control/./Reacher_Linux_NoVis/Reacher.x86_64
    Mono path[0] = '/home/vijayah/deep-reinforcement-learning/p2_continuous-control/./Reacher_Linux_NoVis/Reacher_Data/Managed'
    Mono config path = '/home/vijayah/deep-reinforcement-learning/p2_continuous-control/./Reacher_Linux_NoVis/Reacher_Data/MonoBleedingEdge/etc'
    Preloaded 'libgrpc_csharp_ext.x64.so'
    Unable to preload the following plugins:
    	libgrpc_csharp_ext.x86.so
    Logging to /home/vijayah/.config/unity3d/Unity Technologies/Unity Environment/Player.log
    Preloaded 'libgrpc_csharp_ext.x64.so'
    Unable to preload the following plugins:
    	libgrpc_csharp_ext.x86.so
    Logging to /home/vijayah/.config/unity3d/Unity Technologies/Unity Environment/Player.log


    Failed to create secure directory (/run/user/1001/pulse): No such file or directory
    Failed to create secure directory (/run/user/1001/pulse): No such file or directory
    Failed to create secure directory (/run/user/1001/pulse): No such file or directory
    INFO:unityagents:
    'Academy' started successfully!
    Unity Academy name: Academy
            Number of Brains: 1
            Number of External Brains : 1
            Lesson number : 0
            Reset Parameters :
    		goal_speed -> 1.0
    		goal_size -> 5.0
    Unity brain name: ReacherBrain
            Number of Visual Observations (per agent): 0
            Vector Observation space type: continuous
            Vector Observation space size (per agent): 33
            Number of stacked Vector Observation: 1
            Vector Action space type: continuous
            Vector Action space size (per agent): 4
            Vector Action descriptions: , , , 
    INFO:unityagents:
    'Academy' started successfully!
    Unity Academy name: Academy
            Number of Brains: 1
            Number of External Brains : 1
            Lesson number : 0
            Reset Parameters :
    		goal_speed -> 1.0
    		goal_size -> 5.0
    Unity brain name: ReacherBrain
            Number of Visual Observations (per agent): 0
            Vector Observation space type: continuous
            Vector Observation space size (per agent): 33
            Number of stacked Vector Observation: 1
            Vector Action space type: continuous
            Vector Action space size (per agent): 4
            Vector Action descriptions: , , , 


Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.


```python
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
```




    device(type='cuda', index=0)



### 2. State and Action Spaces and condition for solving the problem

In this environment, a double-jointed arm can move to target locations. A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of `33` variables corresponding to position, rotation, velocity, and angular velocities of the arm.  Each action is a vector with four numbers, corresponding to torque applicable to two joints.  Every entry in the action vector must be a number between `-1` and `1`.

Problem considered solved if we achieve an average score of 30 over 100 consecutive episodes.


```python
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
```

    Number of agents: 1
    Size of each action: 4
    There are 1 agents. Each observes a state with length: 33
    The state for the first agent looks like: [ 0.00000000e+00 -4.00000000e+00  0.00000000e+00  1.00000000e+00
     -0.00000000e+00 -0.00000000e+00 -4.37113883e-08  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00 -1.00000000e+01  0.00000000e+00
      1.00000000e+00 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  5.75471878e+00 -1.00000000e+00
      5.55726671e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
     -1.68164849e-01]


### 3. Hyperparameter Tuning

Below are the hyperparameters used for training the agent using DDPG algorithm. These values were obtained after tuning and experimenting with different values.


```python

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.995            # discount factor
TAU = 1e-3               # for soft update of target parameters
LRA = 1e-4               # learning rate for actor
LRC = 1e-3               # learning rate for critic
UPDATE_EVERY = 4         # how often to update the network
SEED = 0                 # random seed
EPSILON = 1.0            # initial epsilon for exploration
EPSILON_DECAY = 0.995    # epsilon decay rate per episode
EPSILON_MIN = 0.01       # minimum epsilon
```


```python

```

### Network Architecture Description

The agent uses the Deep Deterministic Policy Gradient (DDPG) algorithm, which consists of two neural networks: an Actor and a Critic.

- **Actor Network:**  
  The actor maps states to actions. It consists of two hidden layers with 100 units each, using ReLU activations, followed by an output layer with a tanh activation to ensure actions are in the range [-1, 1].

- **Critic Network:**  
  The critic evaluates the value of state-action pairs. It takes both the state and action as input, concatenates them, and passes them through two hidden layers (100 units each, ReLU activations), followed by a linear output layer that predicts the Q-value.

Both networks are trained using experiences sampled from a replay buffer, and target networks are updated using soft updates for stability.


```python

import torch.nn.functional as F
class Actor(nn.Module):
    def __init__(self, nS, nA): # nS: state space size, nA: action space size
        super(Actor, self).__init__()

        self.h1 = nn.Linear(nS, 100)
        self.h2 = nn.Linear(100, 100)
        self.out = nn.Linear(100, nA)

    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.tanh(self.out(x))
        return x


class Critic(nn.Module):
    def __init__(self, nS, nA): # nS: state space size, nA: action space size
        super(Critic, self).__init__()

        self.h1 = nn.Linear(nS + nA, 100)
        self.h2 = nn.Linear(100, 100)
        self.out = nn.Linear(100, 1)

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.out(x)
        return x

```


```python
import copy


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(size=len(x))
        self.state = x + dx
        return self.state
```

Standard Replay buffer used with a size of 1,000,000 and batch size of 128.

```python


```python
from collections import namedtuple


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
```

### DDPGAgent Implementation Details

The `DDPGAgent` class implements the Deep Deterministic Policy Gradient algorithm for continuous control tasks. Key features and implementation details include:

- **Actor and Critic Networks:**  
  Both local and target networks are created for the actor and critic, using two hidden layers of 100 units each. The actor outputs actions in the range [-1, 1] using a tanh activation.

- **Replay Buffer:**  
  Experiences are stored in a replay buffer with a capacity of 1,000,000 and sampled in batches of 128 for training.

- **Ornstein-Uhlenbeck Noise:**  
  Exploration is encouraged by adding temporally correlated noise to actions using the OU process.

- **Learning and Updates:**  
  The agent updates its networks every 4 steps if enough samples are available. The critic is trained using the mean squared error between predicted and target Q-values, while the actor is trained to maximize expected Q-values. Gradients are clipped for stability.

- **Soft Target Updates:**  
  Target networks are updated using soft updates with a small interpolation factor (`tau`), ensuring stable learning.

- **Multi-Agent Support:**  
  The agent is designed to handle multiple agents by averaging rewards and handling state/action batches.

- **Solving Criterion:**  
  Training stops when the average score over 100 episodes reaches 30, and the trained actor network is saved.

This implementation balances stability and exploration, making it suitable for solving the Unity Reacher environment.


```python

class DDPGAgent:
    def __init__(self, state_size, action_size, random_seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random_seed
        
        self.agent_local = Actor(state_size, action_size).to(device)
        self.agent_target = Actor(state_size, action_size).to(device)
        
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        
        self.optimizer_actor = torch.optim.Adam(self.agent_local.parameters(), lr=LRA)
        self.optimizer_critic = torch.optim.Adam(self.critic_local.parameters(), lr=LRC)
        self.memory = ReplayBuffer(action_size, buffer_size=int(BUFFER_SIZE), batch_size=BATCH_SIZE, seed=random_seed)
        self.noise = OUNoise(action_size, random_seed)
        self.t_step = 0
        self.soft_update(self.agent_local, self.agent_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state,action,reward,next_state,done)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    def action(self, state,noise):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.agent_local.eval()
        with torch.no_grad():
            action = self.agent_local(state).cpu().data.numpy().squeeze(0)
        self.agent_local.train()
        action += self.noise.sample() *noise  # add noise for exploration
        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q
        # critic is updated using Q values 
        # critic_target is used to get the expected Q values for the next states
        
        with torch.no_grad():
            Q_target_next = self.critic_target(next_states, self.agent_target(next_states))
        Q_target = rewards + (gamma * Q_target_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer_critic.zero_grad()  
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.optimizer_critic.step()
        
        action_pred = self.agent_local(states)
        action_loss = -self.critic_local(states, action_pred).mean() 
        self.optimizer_actor.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent_local.parameters(), 1.0)
        self.optimizer_actor.step()
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.agent_local, self.agent_target, TAU)
        
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def reset(self):
        self.noise.reset()
```


```python
agent = DDPGAgent(action_size=4,state_size=33)

def ddpg(n_episodes=10000, max_t=200, print_every=100):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores                  # initialize epsilon
    noise_scale=0.2
    noise_decay=0.999
    for i_episode in range(1, n_episodes+1):
        state = env.reset(train_mode=True)[brain_name].vector_observations
        agent.reset()
        score = 0
        while True:
            action = agent.action(state,noise_scale)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations if len(env_info.vector_observations) > 0 else None
            reward = env_info.rewards                   # get the reward
            done = env_info.local_done
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += np.mean(reward)
            noise_scale *= noise_decay
            if np.any(done):
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        average_reward = sum(scores_window)/len(scores_window)
        print('Episode: {}\t Score: {}\t Avg. Reward: {}'.format(i_episode, score, average_reward))
        if average_reward >= 30:
            print("\t--> SOLVED! <--\t")
            torch.save(agent.agent_target.state_dict(), 'checkpoint.pth')
            break
    return scores


scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
```

    Episode: 1	 Score: 0.14999999664723873	 Avg. Reward: 0.14999999664723873
    Episode: 2	 Score: 0.36999999172985554	 Avg. Reward: 0.25999999418854713
    Episode: 2	 Score: 0.36999999172985554	 Avg. Reward: 0.25999999418854713
    Episode: 3	 Score: 0.6999999843537807	 Avg. Reward: 0.40666665757695836
    Episode: 3	 Score: 0.6999999843537807	 Avg. Reward: 0.40666665757695836
    Episode: 4	 Score: 0.9399999789893627	 Avg. Reward: 0.5399999879300594
    Episode: 4	 Score: 0.9399999789893627	 Avg. Reward: 0.5399999879300594
    Episode: 5	 Score: 0.24999999441206455	 Avg. Reward: 0.4819999892264605
    Episode: 5	 Score: 0.24999999441206455	 Avg. Reward: 0.4819999892264605
    Episode: 6	 Score: 0.24999999441206455	 Avg. Reward: 0.4433333234240611
    Episode: 6	 Score: 0.24999999441206455	 Avg. Reward: 0.4433333234240611
    Episode: 7	 Score: 0.0	 Avg. Reward: 0.3799999915063381
    Episode: 7	 Score: 0.0	 Avg. Reward: 0.3799999915063381
    Episode: 8	 Score: 0.2899999935179949	 Avg. Reward: 0.3687499917577952
    Episode: 8	 Score: 0.2899999935179949	 Avg. Reward: 0.3687499917577952
    Episode: 9	 Score: 0.669999985024333	 Avg. Reward: 0.402222213231855
    Episode: 9	 Score: 0.669999985024333	 Avg. Reward: 0.402222213231855
    Episode: 10	 Score: 0.3399999924004078	 Avg. Reward: 0.39599999114871026
    Episode: 10	 Score: 0.3399999924004078	 Avg. Reward: 0.39599999114871026
    Episode: 11	 Score: 1.2399999722838402	 Avg. Reward: 0.4727272621609948
    Episode: 11	 Score: 1.2399999722838402	 Avg. Reward: 0.4727272621609948
    Episode: 12	 Score: 0.46999998949468136	 Avg. Reward: 0.472499989438802
    Episode: 12	 Score: 0.46999998949468136	 Avg. Reward: 0.472499989438802
    Episode: 13	 Score: 0.8999999798834324	 Avg. Reward: 0.5053846040883889
    Episode: 13	 Score: 0.8999999798834324	 Avg. Reward: 0.5053846040883889
    Episode: 14	 Score: 0.5999999865889549	 Avg. Reward: 0.5121428456955722
    Episode: 14	 Score: 0.5999999865889549	 Avg. Reward: 0.5121428456955722
    Episode: 15	 Score: 0.5799999870359898	 Avg. Reward: 0.5166666551182667
    Episode: 15	 Score: 0.5799999870359898	 Avg. Reward: 0.5166666551182667
    Episode: 16	 Score: 1.3799999691545963	 Avg. Reward: 0.5706249872455373
    Episode: 16	 Score: 1.3799999691545963	 Avg. Reward: 0.5706249872455373
    Episode: 17	 Score: 0.7399999834597111	 Avg. Reward: 0.5805882223169593
    Episode: 17	 Score: 0.7399999834597111	 Avg. Reward: 0.5805882223169593
    Episode: 18	 Score: 0.8599999807775021	 Avg. Reward: 0.5961110977869895
    Episode: 18	 Score: 0.8599999807775021	 Avg. Reward: 0.5961110977869895
    Episode: 19	 Score: 0.8699999805539846	 Avg. Reward: 0.6105263021431471
    Episode: 19	 Score: 0.8699999805539846	 Avg. Reward: 0.6105263021431471
    Episode: 20	 Score: 1.0099999774247408	 Avg. Reward: 0.6304999859072268
    Episode: 20	 Score: 1.0099999774247408	 Avg. Reward: 0.6304999859072268
    Episode: 21	 Score: 0.9499999787658453	 Avg. Reward: 0.6457142712814468
    Episode: 21	 Score: 0.9499999787658453	 Avg. Reward: 0.6457142712814468
    Episode: 22	 Score: 0.3799999915063381	 Avg. Reward: 0.6336363494734872
    Episode: 22	 Score: 0.3799999915063381	 Avg. Reward: 0.6336363494734872
    Episode: 23	 Score: 0.11999999731779099	 Avg. Reward: 0.6113043341623701
    Episode: 23	 Score: 0.11999999731779099	 Avg. Reward: 0.6113043341623701
    Episode: 24	 Score: 0.5199999883770943	 Avg. Reward: 0.6074999864213169
    Episode: 24	 Score: 0.5199999883770943	 Avg. Reward: 0.6074999864213169
    Episode: 25	 Score: 0.6899999845772982	 Avg. Reward: 0.6107999863475562
    Episode: 25	 Score: 0.6899999845772982	 Avg. Reward: 0.6107999863475562
    Episode: 26	 Score: 0.549999987706542	 Avg. Reward: 0.6084615248613633
    Episode: 26	 Score: 0.549999987706542	 Avg. Reward: 0.6084615248613633
    Episode: 27	 Score: 0.4399999901652336	 Avg. Reward: 0.6022222087615066
    Episode: 27	 Score: 0.4399999901652336	 Avg. Reward: 0.6022222087615066
    Episode: 28	 Score: 0.3799999915063381	 Avg. Reward: 0.5942857010023934
    Episode: 28	 Score: 0.3799999915063381	 Avg. Reward: 0.5942857010023934
    Episode: 29	 Score: 0.29999999329447746	 Avg. Reward: 0.5841379179779825
    Episode: 29	 Score: 0.29999999329447746	 Avg. Reward: 0.5841379179779825
    Episode: 30	 Score: 0.9999999776482582	 Avg. Reward: 0.5979999866336584
    Episode: 30	 Score: 0.9999999776482582	 Avg. Reward: 0.5979999866336584
    Episode: 31	 Score: 0.3899999912828207	 Avg. Reward: 0.5912903093642765
    Episode: 31	 Score: 0.3899999912828207	 Avg. Reward: 0.5912903093642765
    Episode: 32	 Score: 1.169999973848462	 Avg. Reward: 0.6093749863794073
    Episode: 32	 Score: 1.169999973848462	 Avg. Reward: 0.6093749863794073
    Episode: 33	 Score: 0.8199999816715717	 Avg. Reward: 0.6157575619943214
    Episode: 33	 Score: 0.8199999816715717	 Avg. Reward: 0.6157575619943214
    Episode: 34	 Score: 0.41999999061226845	 Avg. Reward: 0.6099999863654375
    Episode: 34	 Score: 0.41999999061226845	 Avg. Reward: 0.6099999863654375
    Episode: 35	 Score: 0.8299999814480543	 Avg. Reward: 0.6162857005106551
    Episode: 35	 Score: 0.8299999814480543	 Avg. Reward: 0.6162857005106551
    Episode: 36	 Score: 2.729999938979745	 Avg. Reward: 0.6749999849125743
    Episode: 36	 Score: 2.729999938979745	 Avg. Reward: 0.6749999849125743
    Episode: 37	 Score: 0.8999999798834324	 Avg. Reward: 0.6810810658577326
    Episode: 37	 Score: 0.8999999798834324	 Avg. Reward: 0.6810810658577326
    Episode: 38	 Score: 0.8199999816715717	 Avg. Reward: 0.6847368268002021
    Episode: 38	 Score: 0.8199999816715717	 Avg. Reward: 0.6847368268002021
    Episode: 39	 Score: 0.5799999870359898	 Avg. Reward: 0.682051266806248
    Episode: 39	 Score: 0.5799999870359898	 Avg. Reward: 0.682051266806248
    Episode: 40	 Score: 1.1199999749660492	 Avg. Reward: 0.6929999845102429
    Episode: 40	 Score: 1.1199999749660492	 Avg. Reward: 0.6929999845102429
    Episode: 41	 Score: 0.05999999865889549	 Avg. Reward: 0.6775609604650881
    Episode: 41	 Score: 0.05999999865889549	 Avg. Reward: 0.6775609604650881
    Episode: 42	 Score: 0.12999999709427357	 Avg. Reward: 0.6645237946705449
    Episode: 42	 Score: 0.12999999709427357	 Avg. Reward: 0.6645237946705449
    Episode: 43	 Score: 1.0199999772012234	 Avg. Reward: 0.6727906826363746
    Episode: 43	 Score: 1.0199999772012234	 Avg. Reward: 0.6727906826363746
    Episode: 44	 Score: 0.2199999950826168	 Avg. Reward: 0.6624999851919711
    Episode: 44	 Score: 0.2199999950826168	 Avg. Reward: 0.6624999851919711
    Episode: 45	 Score: 1.3499999698251486	 Avg. Reward: 0.677777762628264
    Episode: 45	 Score: 1.3499999698251486	 Avg. Reward: 0.677777762628264
    Episode: 46	 Score: 0.3999999910593033	 Avg. Reward: 0.671739115420243
    Episode: 46	 Score: 0.3999999910593033	 Avg. Reward: 0.671739115420243
    Episode: 47	 Score: 2.149999951943755	 Avg. Reward: 0.7031914736441476
    Episode: 47	 Score: 2.149999951943755	 Avg. Reward: 0.7031914736441476
    Episode: 48	 Score: 0.06999999843537807	 Avg. Reward: 0.6899999845772982
    Episode: 48	 Score: 0.06999999843537807	 Avg. Reward: 0.6899999845772982
    Episode: 49	 Score: 0.17999999597668648	 Avg. Reward: 0.6795918215446326
    Episode: 49	 Score: 0.17999999597668648	 Avg. Reward: 0.6795918215446326
    Episode: 50	 Score: 0.09999999776482582	 Avg. Reward: 0.6679999850690365
    Episode: 50	 Score: 0.09999999776482582	 Avg. Reward: 0.6679999850690365
    Episode: 51	 Score: 0.42999999038875103	 Avg. Reward: 0.663333318506678
    Episode: 51	 Score: 0.42999999038875103	 Avg. Reward: 0.663333318506678
    Episode: 52	 Score: 0.789999982342124	 Avg. Reward: 0.6657692158881289
    Episode: 52	 Score: 0.789999982342124	 Avg. Reward: 0.6657692158881289
    Episode: 53	 Score: 0.6299999859184027	 Avg. Reward: 0.6650943247566246
    Episode: 53	 Score: 0.6299999859184027	 Avg. Reward: 0.6650943247566246
    Episode: 54	 Score: 0.9499999787658453	 Avg. Reward: 0.670370355386425
    Episode: 54	 Score: 0.9499999787658453	 Avg. Reward: 0.670370355386425
    Episode: 55	 Score: 0.0	 Avg. Reward: 0.6581818034703081
    Episode: 55	 Score: 0.0	 Avg. Reward: 0.6581818034703081
    Episode: 56	 Score: 0.7199999839067459	 Avg. Reward: 0.6592856995495302
    Episode: 56	 Score: 0.7199999839067459	 Avg. Reward: 0.6592856995495302
    Episode: 57	 Score: 0.6099999863654375	 Avg. Reward: 0.6584210379147216
    Episode: 57	 Score: 0.6099999863654375	 Avg. Reward: 0.6584210379147216
    Episode: 58	 Score: 0.6599999852478504	 Avg. Reward: 0.6584482611446031
    Episode: 58	 Score: 0.6599999852478504	 Avg. Reward: 0.6584482611446031
    Episode: 59	 Score: 0.5199999883770943	 Avg. Reward: 0.6561016802502385
    Episode: 59	 Score: 0.5199999883770943	 Avg. Reward: 0.6561016802502385
    Episode: 60	 Score: 1.149999974295497	 Avg. Reward: 0.6643333184843262
    Episode: 60	 Score: 1.149999974295497	 Avg. Reward: 0.6643333184843262
    Episode: 61	 Score: 0.7799999825656414	 Avg. Reward: 0.6662294933053313
    Episode: 61	 Score: 0.7799999825656414	 Avg. Reward: 0.6662294933053313
    Episode: 62	 Score: 1.5999999642372131	 Avg. Reward: 0.6812903073526198
    Episode: 62	 Score: 1.5999999642372131	 Avg. Reward: 0.6812903073526198
    Episode: 63	 Score: 0.6599999852478504	 Avg. Reward: 0.6809523657319092
    Episode: 63	 Score: 0.6599999852478504	 Avg. Reward: 0.6809523657319092
    Episode: 64	 Score: 0.9499999787658453	 Avg. Reward: 0.6851562346855644
    Episode: 64	 Score: 0.9499999787658453	 Avg. Reward: 0.6851562346855644
    Episode: 65	 Score: 1.3099999707192183	 Avg. Reward: 0.6947692152399283
    Episode: 65	 Score: 1.3099999707192183	 Avg. Reward: 0.6947692152399283
    Episode: 66	 Score: 0.3799999915063381	 Avg. Reward: 0.6899999845772982
    Episode: 66	 Score: 0.3799999915063381	 Avg. Reward: 0.6899999845772982
    Episode: 67	 Score: 0.0	 Avg. Reward: 0.6797014773448011
    Episode: 67	 Score: 0.0	 Avg. Reward: 0.6797014773448011
    Episode: 68	 Score: 0.9399999789893627	 Avg. Reward: 0.6835293964866329
    Episode: 68	 Score: 0.9399999789893627	 Avg. Reward: 0.6835293964866329
    Episode: 69	 Score: 0.7099999841302633	 Avg. Reward: 0.6839130281916131
    Episode: 69	 Score: 0.7099999841302633	 Avg. Reward: 0.6839130281916131
    Episode: 70	 Score: 0.7799999825656414	 Avg. Reward: 0.685285698968385
    Episode: 70	 Score: 0.7799999825656414	 Avg. Reward: 0.685285698968385
    Episode: 71	 Score: 1.4899999666959047	 Avg. Reward: 0.696619702739195
    Episode: 71	 Score: 1.4899999666959047	 Avg. Reward: 0.696619702739195
    Episode: 72	 Score: 0.8999999798834324	 Avg. Reward: 0.6994444288106428
    Episode: 72	 Score: 0.8999999798834324	 Avg. Reward: 0.6994444288106428
    Episode: 73	 Score: 1.529999965801835	 Avg. Reward: 0.7108219019201112
    Episode: 73	 Score: 1.529999965801835	 Avg. Reward: 0.7108219019201112
    Episode: 74	 Score: 0.9199999794363976	 Avg. Reward: 0.7136486326973583
    Episode: 74	 Score: 0.9199999794363976	 Avg. Reward: 0.7136486326973583
    Episode: 75	 Score: 1.7299999613314867	 Avg. Reward: 0.7271999837458134
    Episode: 75	 Score: 1.7299999613314867	 Avg. Reward: 0.7271999837458134
    Episode: 76	 Score: 0.2299999948590994	 Avg. Reward: 0.720657878628883
    Episode: 76	 Score: 0.2299999948590994	 Avg. Reward: 0.720657878628883
    Episode: 77	 Score: 1.169999973848462	 Avg. Reward: 0.7264934902551112
    Episode: 77	 Score: 1.169999973848462	 Avg. Reward: 0.7264934902551112
    Episode: 78	 Score: 1.5699999649077654	 Avg. Reward: 0.7373076758275812
    Episode: 78	 Score: 1.5699999649077654	 Avg. Reward: 0.7373076758275812
    Episode: 79	 Score: 0.4499999899417162	 Avg. Reward: 0.7336708696771271
    Episode: 79	 Score: 0.4499999899417162	 Avg. Reward: 0.7336708696771271
    Episode: 80	 Score: 0.5899999868124723	 Avg. Reward: 0.731874983641319
    Episode: 80	 Score: 0.5899999868124723	 Avg. Reward: 0.731874983641319
    Episode: 81	 Score: 1.4699999671429396	 Avg. Reward: 0.740987637758623
    Episode: 81	 Score: 1.4699999671429396	 Avg. Reward: 0.740987637758623
    Episode: 82	 Score: 1.5599999651312828	 Avg. Reward: 0.7509755929704847
    Episode: 82	 Score: 1.5599999651312828	 Avg. Reward: 0.7509755929704847
    Episode: 83	 Score: 0.7499999832361937	 Avg. Reward: 0.7509638386363365
    Episode: 83	 Score: 0.7499999832361937	 Avg. Reward: 0.7509638386363365
    Episode: 84	 Score: 0.4499999899417162	 Avg. Reward: 0.7473809356756863
    Episode: 84	 Score: 0.4499999899417162	 Avg. Reward: 0.7473809356756863
    Episode: 85	 Score: 1.6199999637901783	 Avg. Reward: 0.757647041888798
    Episode: 85	 Score: 1.6199999637901783	 Avg. Reward: 0.757647041888798
    Episode: 86	 Score: 2.0999999530613422	 Avg. Reward: 0.7732557966698741
    Episode: 86	 Score: 2.0999999530613422	 Avg. Reward: 0.7732557966698741
    Episode: 87	 Score: 1.649999963119626	 Avg. Reward: 0.783333315824469
    Episode: 87	 Score: 1.649999963119626	 Avg. Reward: 0.783333315824469
    Episode: 88	 Score: 0.8399999812245369	 Avg. Reward: 0.7839772552040152
    Episode: 88	 Score: 0.8399999812245369	 Avg. Reward: 0.7839772552040152
    Episode: 89	 Score: 1.7099999617785215	 Avg. Reward: 0.7943820047160882
    Episode: 89	 Score: 1.7099999617785215	 Avg. Reward: 0.7943820047160882
    Episode: 90	 Score: 0.2299999948590994	 Avg. Reward: 0.788111093495455
    Episode: 90	 Score: 0.2299999948590994	 Avg. Reward: 0.788111093495455
    Episode: 91	 Score: 0.08999999798834324	 Avg. Reward: 0.7804395429953769
    Episode: 91	 Score: 0.08999999798834324	 Avg. Reward: 0.7804395429953769
    Episode: 92	 Score: 1.0099999774247408	 Avg. Reward: 0.7829347651087396
    Episode: 92	 Score: 1.0099999774247408	 Avg. Reward: 0.7829347651087396
    Episode: 93	 Score: 1.2999999709427357	 Avg. Reward: 0.7884946060316858
    Episode: 93	 Score: 1.2999999709427357	 Avg. Reward: 0.7884946060316858
    Episode: 94	 Score: 0.30999999307096004	 Avg. Reward: 0.7834042378086993
    Episode: 94	 Score: 0.30999999307096004	 Avg. Reward: 0.7834042378086993
    Episode: 95	 Score: 1.269999971613288	 Avg. Reward: 0.7885262981645371
    Episode: 95	 Score: 1.269999971613288	 Avg. Reward: 0.7885262981645371
    Episode: 96	 Score: 0.019999999552965164	 Avg. Reward: 0.7805208158873332
    Episode: 96	 Score: 0.019999999552965164	 Avg. Reward: 0.7805208158873332
    Episode: 97	 Score: 0.0	 Avg. Reward: 0.7724742095379793
    Episode: 97	 Score: 0.0	 Avg. Reward: 0.7724742095379793
    Episode: 98	 Score: 0.4499999899417162	 Avg. Reward: 0.7691836562767929
    Episode: 98	 Score: 0.4499999899417162	 Avg. Reward: 0.7691836562767929
    Episode: 99	 Score: 0.06999999843537807	 Avg. Reward: 0.7621211950864756
    Episode: 99	 Score: 0.06999999843537807	 Avg. Reward: 0.7621211950864756
    Episode: 100	 Score: 1.7399999611079693	 Avg. Reward: 0.7718999827466905
    Episode: 100	 Score: 1.7399999611079693	 Avg. Reward: 0.7718999827466905
    Episode: 101	 Score: 0.35999999195337296	 Avg. Reward: 0.7739999826997519
    Episode: 101	 Score: 0.35999999195337296	 Avg. Reward: 0.7739999826997519
    Episode: 102	 Score: 0.8399999812245369	 Avg. Reward: 0.7786999825946986
    Episode: 102	 Score: 0.8399999812245369	 Avg. Reward: 0.7786999825946986
    Episode: 103	 Score: 0.36999999172985554	 Avg. Reward: 0.7753999826684594
    Episode: 103	 Score: 0.36999999172985554	 Avg. Reward: 0.7753999826684594
    Episode: 104	 Score: 0.9199999794363976	 Avg. Reward: 0.7751999826729298
    Episode: 104	 Score: 0.9199999794363976	 Avg. Reward: 0.7751999826729298
    Episode: 105	 Score: 0.8099999818950891	 Avg. Reward: 0.7807999825477601
    Episode: 105	 Score: 0.8099999818950891	 Avg. Reward: 0.7807999825477601
    Episode: 106	 Score: 0.8099999818950891	 Avg. Reward: 0.7863999824225902
    Episode: 106	 Score: 0.8099999818950891	 Avg. Reward: 0.7863999824225902
    Episode: 107	 Score: 0.9299999792128801	 Avg. Reward: 0.795699982214719
    Episode: 107	 Score: 0.9299999792128801	 Avg. Reward: 0.795699982214719
    Episode: 108	 Score: 0.9299999792128801	 Avg. Reward: 0.8020999820716679
    Episode: 108	 Score: 0.9299999792128801	 Avg. Reward: 0.8020999820716679
    Episode: 109	 Score: 0.6499999854713678	 Avg. Reward: 0.8018999820761382
    Episode: 109	 Score: 0.6499999854713678	 Avg. Reward: 0.8018999820761382
    Episode: 110	 Score: 0.9199999794363976	 Avg. Reward: 0.8076999819464982
    Episode: 110	 Score: 0.9199999794363976	 Avg. Reward: 0.8076999819464982
    Episode: 111	 Score: 0.1599999964237213	 Avg. Reward: 0.7968999821878969
    Episode: 111	 Score: 0.1599999964237213	 Avg. Reward: 0.7968999821878969
    Episode: 112	 Score: 0.29999999329447746	 Avg. Reward: 0.7951999822258949
    Episode: 112	 Score: 0.29999999329447746	 Avg. Reward: 0.7951999822258949
    Episode: 113	 Score: 0.7999999821186066	 Avg. Reward: 0.7941999822482466
    Episode: 113	 Score: 0.7999999821186066	 Avg. Reward: 0.7941999822482466
    Episode: 114	 Score: 0.9199999794363976	 Avg. Reward: 0.7973999821767211
    Episode: 114	 Score: 0.9199999794363976	 Avg. Reward: 0.7973999821767211
    Episode: 115	 Score: 1.7899999599903822	 Avg. Reward: 0.809499981906265
    Episode: 115	 Score: 1.7899999599903822	 Avg. Reward: 0.809499981906265
    Episode: 116	 Score: 0.7099999841302633	 Avg. Reward: 0.8027999820560217
    Episode: 116	 Score: 0.7099999841302633	 Avg. Reward: 0.8027999820560217
    Episode: 117	 Score: 1.2299999725073576	 Avg. Reward: 0.8076999819464982
    Episode: 117	 Score: 1.2299999725073576	 Avg. Reward: 0.8076999819464982
    Episode: 118	 Score: 0.46999998949468136	 Avg. Reward: 0.80379998203367
    Episode: 118	 Score: 0.46999998949468136	 Avg. Reward: 0.80379998203367
    Episode: 119	 Score: 0.3299999926239252	 Avg. Reward: 0.7983999821543694
    Episode: 119	 Score: 0.3299999926239252	 Avg. Reward: 0.7983999821543694
    Episode: 120	 Score: 0.7999999821186066	 Avg. Reward: 0.796299982201308
    Episode: 120	 Score: 0.7999999821186066	 Avg. Reward: 0.796299982201308
    Episode: 121	 Score: 0.8899999801069498	 Avg. Reward: 0.795699982214719
    Episode: 121	 Score: 0.8899999801069498	 Avg. Reward: 0.795699982214719
    Episode: 122	 Score: 0.6299999859184027	 Avg. Reward: 0.7981999821588397
    Episode: 122	 Score: 0.6299999859184027	 Avg. Reward: 0.7981999821588397
    Episode: 123	 Score: 0.30999999307096004	 Avg. Reward: 0.8000999821163713
    Episode: 123	 Score: 0.30999999307096004	 Avg. Reward: 0.8000999821163713
    Episode: 124	 Score: 1.7899999599903822	 Avg. Reward: 0.8127999818325042
    Episode: 124	 Score: 1.7899999599903822	 Avg. Reward: 0.8127999818325042
    Episode: 125	 Score: 1.6799999624490738	 Avg. Reward: 0.822699981611222
    Episode: 125	 Score: 1.6799999624490738	 Avg. Reward: 0.822699981611222
    Episode: 126	 Score: 0.7299999836832285	 Avg. Reward: 0.8244999815709889
    Episode: 126	 Score: 0.7299999836832285	 Avg. Reward: 0.8244999815709889
    Episode: 127	 Score: 0.36999999172985554	 Avg. Reward: 0.8237999815866351
    Episode: 127	 Score: 0.36999999172985554	 Avg. Reward: 0.8237999815866351
    Episode: 128	 Score: 1.9099999573081732	 Avg. Reward: 0.8390999812446535
    Episode: 128	 Score: 1.9099999573081732	 Avg. Reward: 0.8390999812446535
    Episode: 129	 Score: 1.529999965801835	 Avg. Reward: 0.851399980969727
    Episode: 129	 Score: 1.529999965801835	 Avg. Reward: 0.851399980969727
    Episode: 130	 Score: 2.029999954625964	 Avg. Reward: 0.861699980739504
    Episode: 130	 Score: 2.029999954625964	 Avg. Reward: 0.861699980739504
    Episode: 131	 Score: 2.6199999414384365	 Avg. Reward: 0.8839999802410603
    Episode: 131	 Score: 2.6199999414384365	 Avg. Reward: 0.8839999802410603
    Episode: 132	 Score: 1.2399999722838402	 Avg. Reward: 0.8846999802254141
    Episode: 132	 Score: 1.2399999722838402	 Avg. Reward: 0.8846999802254141
    Episode: 133	 Score: 5.349999880418181	 Avg. Reward: 0.9299999792128801
    Episode: 133	 Score: 5.349999880418181	 Avg. Reward: 0.9299999792128801
    Episode: 134	 Score: 2.4299999456852674	 Avg. Reward: 0.9500999787636101
    Episode: 134	 Score: 2.4299999456852674	 Avg. Reward: 0.9500999787636101
    Episode: 135	 Score: 1.6599999628961086	 Avg. Reward: 0.9583999785780907
    Episode: 135	 Score: 1.6599999628961086	 Avg. Reward: 0.9583999785780907
    Episode: 136	 Score: 0.41999999061226845	 Avg. Reward: 0.9352999790944159
    Episode: 136	 Score: 0.41999999061226845	 Avg. Reward: 0.9352999790944159
    Episode: 137	 Score: 3.179999928921461	 Avg. Reward: 0.9580999785847962
    Episode: 137	 Score: 3.179999928921461	 Avg. Reward: 0.9580999785847962
    Episode: 138	 Score: 4.239999905228615	 Avg. Reward: 0.9922999778203666
    Episode: 138	 Score: 4.239999905228615	 Avg. Reward: 0.9922999778203666
    Episode: 139	 Score: 3.3199999257922173	 Avg. Reward: 1.019699977207929
    Episode: 139	 Score: 3.3199999257922173	 Avg. Reward: 1.019699977207929
    Episode: 140	 Score: 1.4899999666959047	 Avg. Reward: 1.0233999771252273
    Episode: 140	 Score: 1.4899999666959047	 Avg. Reward: 1.0233999771252273
    Episode: 141	 Score: 4.099999908357859	 Avg. Reward: 1.063799976222217
    Episode: 141	 Score: 4.099999908357859	 Avg. Reward: 1.063799976222217
    Episode: 142	 Score: 1.7999999597668648	 Avg. Reward: 1.080499975848943
    Episode: 142	 Score: 1.7999999597668648	 Avg. Reward: 1.080499975848943
    Episode: 143	 Score: 0.0	 Avg. Reward: 1.0702999760769307
    Episode: 143	 Score: 0.0	 Avg. Reward: 1.0702999760769307
    Episode: 144	 Score: 0.8999999798834324	 Avg. Reward: 1.077099975924939
    Episode: 144	 Score: 0.8999999798834324	 Avg. Reward: 1.077099975924939
    Episode: 145	 Score: 1.2099999729543924	 Avg. Reward: 1.0756999759562313
    Episode: 145	 Score: 1.2099999729543924	 Avg. Reward: 1.0756999759562313
    Episode: 146	 Score: 1.7099999617785215	 Avg. Reward: 1.0887999756634235
    Episode: 146	 Score: 1.7099999617785215	 Avg. Reward: 1.0887999756634235
    Episode: 147	 Score: 3.3599999248981476	 Avg. Reward: 1.1008999753929674
    Episode: 147	 Score: 3.3599999248981476	 Avg. Reward: 1.1008999753929674
    Episode: 148	 Score: 4.229999905452132	 Avg. Reward: 1.142499974463135
    Episode: 148	 Score: 4.229999905452132	 Avg. Reward: 1.142499974463135
    Episode: 149	 Score: 2.7199999392032623	 Avg. Reward: 1.1678999738954008
    Episode: 149	 Score: 2.7199999392032623	 Avg. Reward: 1.1678999738954008
    Episode: 150	 Score: 3.3299999255687	 Avg. Reward: 1.2001999731734394
    Episode: 150	 Score: 3.3299999255687	 Avg. Reward: 1.2001999731734394
    Episode: 151	 Score: 2.079999953508377	 Avg. Reward: 1.2166999728046357
    Episode: 151	 Score: 2.079999953508377	 Avg. Reward: 1.2166999728046357
    Episode: 152	 Score: 2.079999953508377	 Avg. Reward: 1.2295999725162983
    Episode: 152	 Score: 2.079999953508377	 Avg. Reward: 1.2295999725162983
    Episode: 153	 Score: 2.4499999452382326	 Avg. Reward: 1.2477999721094966
    Episode: 153	 Score: 2.4499999452382326	 Avg. Reward: 1.2477999721094966
    Episode: 154	 Score: 1.769999960437417	 Avg. Reward: 1.2559999719262123
    Episode: 154	 Score: 1.769999960437417	 Avg. Reward: 1.2559999719262123
    Episode: 155	 Score: 1.529999965801835	 Avg. Reward: 1.2712999715842306
    Episode: 155	 Score: 1.529999965801835	 Avg. Reward: 1.2712999715842306
    Episode: 156	 Score: 2.4299999456852674	 Avg. Reward: 1.2883999712020158
    Episode: 156	 Score: 2.4299999456852674	 Avg. Reward: 1.2883999712020158
    Episode: 157	 Score: 1.0899999756366014	 Avg. Reward: 1.2931999710947275
    Episode: 157	 Score: 1.0899999756366014	 Avg. Reward: 1.2931999710947275
    Episode: 158	 Score: 0.13999999687075615	 Avg. Reward: 1.2879999712109567
    Episode: 158	 Score: 0.13999999687075615	 Avg. Reward: 1.2879999712109567
    Episode: 159	 Score: 7.079999841749668	 Avg. Reward: 1.3535999697446823
    Episode: 159	 Score: 7.079999841749668	 Avg. Reward: 1.3535999697446823
    Episode: 160	 Score: 2.199999950826168	 Avg. Reward: 1.3640999695099891
    Episode: 160	 Score: 2.199999950826168	 Avg. Reward: 1.3640999695099891
    Episode: 161	 Score: 1.7599999606609344	 Avg. Reward: 1.373899969290942
    Episode: 161	 Score: 1.7599999606609344	 Avg. Reward: 1.373899969290942
    Episode: 162	 Score: 0.6599999852478504	 Avg. Reward: 1.3644999695010482
    Episode: 162	 Score: 0.6599999852478504	 Avg. Reward: 1.3644999695010482
    Episode: 163	 Score: 0.0	 Avg. Reward: 1.3578999696485698
    Episode: 163	 Score: 0.0	 Avg. Reward: 1.3578999696485698
    Episode: 164	 Score: 3.2599999271333218	 Avg. Reward: 1.3809999691322445
    Episode: 164	 Score: 3.2599999271333218	 Avg. Reward: 1.3809999691322445
    Episode: 165	 Score: 1.8799999579787254	 Avg. Reward: 1.3866999690048396
    Episode: 165	 Score: 1.8799999579787254	 Avg. Reward: 1.3866999690048396
    Episode: 166	 Score: 2.249999949708581	 Avg. Reward: 1.4053999685868621
    Episode: 166	 Score: 2.249999949708581	 Avg. Reward: 1.4053999685868621
    Episode: 167	 Score: 0.2799999937415123	 Avg. Reward: 1.4081999685242772
    Episode: 167	 Score: 0.2799999937415123	 Avg. Reward: 1.4081999685242772
    Episode: 168	 Score: 0.8299999814480543	 Avg. Reward: 1.407099968548864
    Episode: 168	 Score: 0.8299999814480543	 Avg. Reward: 1.407099968548864
    Episode: 169	 Score: 3.4299999233335257	 Avg. Reward: 1.4342999679408968
    Episode: 169	 Score: 3.4299999233335257	 Avg. Reward: 1.4342999679408968
    Episode: 170	 Score: 2.2599999494850636	 Avg. Reward: 1.449099967610091
    Episode: 170	 Score: 2.2599999494850636	 Avg. Reward: 1.449099967610091
    Episode: 171	 Score: 1.769999960437417	 Avg. Reward: 1.4518999675475062
    Episode: 171	 Score: 1.769999960437417	 Avg. Reward: 1.4518999675475062
    Episode: 172	 Score: 5.099999886006117	 Avg. Reward: 1.493899966608733
    Episode: 172	 Score: 5.099999886006117	 Avg. Reward: 1.493899966608733
    Episode: 173	 Score: 2.12999995239079	 Avg. Reward: 1.4998999664746224
    Episode: 173	 Score: 2.12999995239079	 Avg. Reward: 1.4998999664746224
    Episode: 174	 Score: 1.0699999760836363	 Avg. Reward: 1.5013999664410949
    Episode: 174	 Score: 1.0699999760836363	 Avg. Reward: 1.5013999664410949
    Episode: 175	 Score: 1.50999996624887	 Avg. Reward: 1.4991999664902687
    Episode: 175	 Score: 1.50999996624887	 Avg. Reward: 1.4991999664902687
    Episode: 176	 Score: 1.4199999682605267	 Avg. Reward: 1.511099966224283
    Episode: 176	 Score: 1.4199999682605267	 Avg. Reward: 1.511099966224283
    Episode: 177	 Score: 2.0899999532848597	 Avg. Reward: 1.520299966018647
    Episode: 177	 Score: 2.0899999532848597	 Avg. Reward: 1.520299966018647
    Episode: 178	 Score: 1.81999995931983	 Avg. Reward: 1.5227999659627676
    Episode: 178	 Score: 1.81999995931983	 Avg. Reward: 1.5227999659627676
    Episode: 179	 Score: 1.50999996624887	 Avg. Reward: 1.533399965725839
    Episode: 179	 Score: 1.50999996624887	 Avg. Reward: 1.533399965725839
    Episode: 180	 Score: 1.1899999734014273	 Avg. Reward: 1.5393999655917288
    Episode: 180	 Score: 1.1899999734014273	 Avg. Reward: 1.5393999655917288
    Episode: 181	 Score: 1.8999999575316906	 Avg. Reward: 1.5436999654956163
    Episode: 181	 Score: 1.8999999575316906	 Avg. Reward: 1.5436999654956163
    Episode: 182	 Score: 3.729999916628003	 Avg. Reward: 1.5653999650105834
    Episode: 182	 Score: 3.729999916628003	 Avg. Reward: 1.5653999650105834
    Episode: 183	 Score: 3.009999932721257	 Avg. Reward: 1.5879999645054341
    Episode: 183	 Score: 3.009999932721257	 Avg. Reward: 1.5879999645054341
    Episode: 184	 Score: 3.349999925121665	 Avg. Reward: 1.6169999638572334
    Episode: 184	 Score: 3.349999925121665	 Avg. Reward: 1.6169999638572334
    Episode: 185	 Score: 3.7399999164044857	 Avg. Reward: 1.6381999633833766
    Episode: 185	 Score: 3.7399999164044857	 Avg. Reward: 1.6381999633833766
    Episode: 186	 Score: 2.659999940544367	 Avg. Reward: 1.6437999632582068
    Episode: 186	 Score: 2.659999940544367	 Avg. Reward: 1.6437999632582068
    Episode: 187	 Score: 3.0499999318271875	 Avg. Reward: 1.6577999629452824
    Episode: 187	 Score: 3.0499999318271875	 Avg. Reward: 1.6577999629452824
    Episode: 188	 Score: 2.609999941661954	 Avg. Reward: 1.6754999625496567
    Episode: 188	 Score: 2.609999941661954	 Avg. Reward: 1.6754999625496567
    Episode: 189	 Score: 1.5899999644607306	 Avg. Reward: 1.6742999625764787
    Episode: 189	 Score: 1.5899999644607306	 Avg. Reward: 1.6742999625764787
    Episode: 190	 Score: 2.9499999340623617	 Avg. Reward: 1.7014999619685114
    Episode: 190	 Score: 2.9499999340623617	 Avg. Reward: 1.7014999619685114
    Episode: 191	 Score: 1.769999960437417	 Avg. Reward: 1.7182999615930021
    Episode: 191	 Score: 1.769999960437417	 Avg. Reward: 1.7182999615930021
    Episode: 192	 Score: 2.5199999436736107	 Avg. Reward: 1.7333999612554907
    Episode: 192	 Score: 2.5199999436736107	 Avg. Reward: 1.7333999612554907
    Episode: 193	 Score: 3.009999932721257	 Avg. Reward: 1.750499960873276
    Episode: 193	 Score: 3.009999932721257	 Avg. Reward: 1.750499960873276
    Episode: 194	 Score: 4.539999898523092	 Avg. Reward: 1.7927999599277973
    Episode: 194	 Score: 4.539999898523092	 Avg. Reward: 1.7927999599277973
    Episode: 195	 Score: 3.899999912828207	 Avg. Reward: 1.8190999593399466
    Episode: 195	 Score: 3.899999912828207	 Avg. Reward: 1.8190999593399466
    Episode: 196	 Score: 4.089999908581376	 Avg. Reward: 1.8597999584302307
    Episode: 196	 Score: 4.089999908581376	 Avg. Reward: 1.8597999584302307
    Episode: 197	 Score: 5.619999874383211	 Avg. Reward: 1.9159999571740627
    Episode: 197	 Score: 5.619999874383211	 Avg. Reward: 1.9159999571740627
    Episode: 198	 Score: 2.2099999506026506	 Avg. Reward: 1.933599956780672
    Episode: 198	 Score: 2.2099999506026506	 Avg. Reward: 1.933599956780672
    Episode: 199	 Score: 6.269999859854579	 Avg. Reward: 1.995599955394864
    Episode: 199	 Score: 6.269999859854579	 Avg. Reward: 1.995599955394864
    Episode: 200	 Score: 5.939999867230654	 Avg. Reward: 2.0375999544560908
    Episode: 200	 Score: 5.939999867230654	 Avg. Reward: 2.0375999544560908
    Episode: 201	 Score: 4.479999899864197	 Avg. Reward: 2.0787999535351993
    Episode: 201	 Score: 4.479999899864197	 Avg. Reward: 2.0787999535351993
    Episode: 202	 Score: 4.129999907687306	 Avg. Reward: 2.111699952799827
    Episode: 202	 Score: 4.129999907687306	 Avg. Reward: 2.111699952799827
    Episode: 203	 Score: 1.9999999552965164	 Avg. Reward: 2.1279999524354936
    Episode: 203	 Score: 1.9999999552965164	 Avg. Reward: 2.1279999524354936
    Episode: 204	 Score: 4.109999908134341	 Avg. Reward: 2.159899951722473
    Episode: 204	 Score: 4.109999908134341	 Avg. Reward: 2.159899951722473
    Episode: 205	 Score: 4.339999902993441	 Avg. Reward: 2.1951999509334565
    Episode: 205	 Score: 4.339999902993441	 Avg. Reward: 2.1951999509334565
    Episode: 206	 Score: 5.979999866336584	 Avg. Reward: 2.2468999497778714
    Episode: 206	 Score: 5.979999866336584	 Avg. Reward: 2.2468999497778714
    Episode: 207	 Score: 3.7699999157339334	 Avg. Reward: 2.275299949143082
    Episode: 207	 Score: 3.7699999157339334	 Avg. Reward: 2.275299949143082
    Episode: 208	 Score: 3.709999917075038	 Avg. Reward: 2.3030999485217034
    Episode: 208	 Score: 3.709999917075038	 Avg. Reward: 2.3030999485217034
    Episode: 209	 Score: 5.209999883547425	 Avg. Reward: 2.348699947502464
    Episode: 209	 Score: 5.209999883547425	 Avg. Reward: 2.348699947502464
    Episode: 210	 Score: 5.279999881982803	 Avg. Reward: 2.392299946527928
    Episode: 210	 Score: 5.279999881982803	 Avg. Reward: 2.392299946527928
    Episode: 211	 Score: 7.579999830573797	 Avg. Reward: 2.4664999448694287
    Episode: 211	 Score: 7.579999830573797	 Avg. Reward: 2.4664999448694287
    Episode: 212	 Score: 6.789999848231673	 Avg. Reward: 2.531399943418801
    Episode: 212	 Score: 6.789999848231673	 Avg. Reward: 2.531399943418801
    Episode: 213	 Score: 3.2399999275803566	 Avg. Reward: 2.5557999428734184
    Episode: 213	 Score: 3.2399999275803566	 Avg. Reward: 2.5557999428734184
    Episode: 214	 Score: 4.379999902099371	 Avg. Reward: 2.590399942100048
    Episode: 214	 Score: 4.379999902099371	 Avg. Reward: 2.590399942100048
    Episode: 215	 Score: 1.579999964684248	 Avg. Reward: 2.5882999421469868
    Episode: 215	 Score: 1.579999964684248	 Avg. Reward: 2.5882999421469868
    Episode: 216	 Score: 5.869999868795276	 Avg. Reward: 2.639899940993637
    Episode: 216	 Score: 5.869999868795276	 Avg. Reward: 2.639899940993637
    Episode: 217	 Score: 7.5499998312443495	 Avg. Reward: 2.7030999395810067
    Episode: 217	 Score: 7.5499998312443495	 Avg. Reward: 2.7030999395810067
    Episode: 218	 Score: 6.499999854713678	 Avg. Reward: 2.7633999382331966
    Episode: 218	 Score: 6.499999854713678	 Avg. Reward: 2.7633999382331966
    Episode: 219	 Score: 10.95999975502491	 Avg. Reward: 2.8696999358572066
    Episode: 219	 Score: 10.95999975502491	 Avg. Reward: 2.8696999358572066
    Episode: 220	 Score: 6.309999858960509	 Avg. Reward: 2.9247999346256255
    Episode: 220	 Score: 6.309999858960509	 Avg. Reward: 2.9247999346256255
    Episode: 221	 Score: 3.3799999244511127	 Avg. Reward: 2.9496999340690673
    Episode: 221	 Score: 3.3799999244511127	 Avg. Reward: 2.9496999340690673
    Episode: 222	 Score: 6.409999856725335	 Avg. Reward: 3.0074999327771366
    Episode: 222	 Score: 6.409999856725335	 Avg. Reward: 3.0074999327771366
    Episode: 223	 Score: 5.0599998869001865	 Avg. Reward: 3.054999931715429
    Episode: 223	 Score: 5.0599998869001865	 Avg. Reward: 3.054999931715429
    Episode: 224	 Score: 3.8399999141693115	 Avg. Reward: 3.075499931257218
    Episode: 224	 Score: 3.8399999141693115	 Avg. Reward: 3.075499931257218
    Episode: 225	 Score: 8.029999820515513	 Avg. Reward: 3.1389999298378823
    Episode: 225	 Score: 8.029999820515513	 Avg. Reward: 3.1389999298378823
    Episode: 226	 Score: 11.049999753013253	 Avg. Reward: 3.242199927531183
    Episode: 226	 Score: 11.049999753013253	 Avg. Reward: 3.242199927531183
    Episode: 227	 Score: 4.829999892041087	 Avg. Reward: 3.286799926534295
    Episode: 227	 Score: 4.829999892041087	 Avg. Reward: 3.286799926534295
    Episode: 228	 Score: 1.2599999718368053	 Avg. Reward: 3.2802999266795814
    Episode: 228	 Score: 1.2599999718368053	 Avg. Reward: 3.2802999266795814
    Episode: 229	 Score: 6.889999845996499	 Avg. Reward: 3.333899925481528
    Episode: 229	 Score: 6.889999845996499	 Avg. Reward: 3.333899925481528
    Episode: 230	 Score: 3.039999932050705	 Avg. Reward: 3.3439999252557753
    Episode: 230	 Score: 3.039999932050705	 Avg. Reward: 3.3439999252557753
    Episode: 231	 Score: 4.209999905899167	 Avg. Reward: 3.359899924900383
    Episode: 231	 Score: 4.209999905899167	 Avg. Reward: 3.359899924900383
    Episode: 232	 Score: 9.419999789446592	 Avg. Reward: 3.4416999230720102
    Episode: 232	 Score: 9.419999789446592	 Avg. Reward: 3.4416999230720102
    Episode: 233	 Score: 19.649999560788274	 Avg. Reward: 3.5846999198757112
    Episode: 233	 Score: 19.649999560788274	 Avg. Reward: 3.5846999198757112
    Episode: 234	 Score: 10.37999976798892	 Avg. Reward: 3.6641999180987477
    Episode: 234	 Score: 10.37999976798892	 Avg. Reward: 3.6641999180987477
    Episode: 235	 Score: 10.56999976374209	 Avg. Reward: 3.7532999161072076
    Episode: 235	 Score: 10.56999976374209	 Avg. Reward: 3.7532999161072076
    Episode: 236	 Score: 16.61999962851405	 Avg. Reward: 3.9152999124862253
    Episode: 236	 Score: 16.61999962851405	 Avg. Reward: 3.9152999124862253
    Episode: 237	 Score: 6.0899998638778925	 Avg. Reward: 3.9443999118357898
    Episode: 237	 Score: 6.0899998638778925	 Avg. Reward: 3.9443999118357898
    Episode: 238	 Score: 6.189999861642718	 Avg. Reward: 3.9638999113999307
    Episode: 238	 Score: 6.189999861642718	 Avg. Reward: 3.9638999113999307
    Episode: 239	 Score: 10.359999768435955	 Avg. Reward: 4.034299909826368
    Episode: 239	 Score: 10.359999768435955	 Avg. Reward: 4.034299909826368
    Episode: 240	 Score: 11.749999737367034	 Avg. Reward: 4.1368999075330795
    Episode: 240	 Score: 11.749999737367034	 Avg. Reward: 4.1368999075330795
    Episode: 241	 Score: 14.709999671205878	 Avg. Reward: 4.242999905161559
    Episode: 241	 Score: 14.709999671205878	 Avg. Reward: 4.242999905161559
    Episode: 242	 Score: 7.679999828338623	 Avg. Reward: 4.301799903847277
    Episode: 242	 Score: 7.679999828338623	 Avg. Reward: 4.301799903847277
    Episode: 243	 Score: 5.509999876841903	 Avg. Reward: 4.356899902615696
    Episode: 243	 Score: 5.509999876841903	 Avg. Reward: 4.356899902615696
    Episode: 244	 Score: 13.849999690428376	 Avg. Reward: 4.486399899721146
    Episode: 244	 Score: 13.849999690428376	 Avg. Reward: 4.486399899721146
    Episode: 245	 Score: 14.199999682605267	 Avg. Reward: 4.6162998968176545
    Episode: 245	 Score: 14.199999682605267	 Avg. Reward: 4.6162998968176545
    Episode: 246	 Score: 8.079999819397926	 Avg. Reward: 4.679999895393848
    Episode: 246	 Score: 8.079999819397926	 Avg. Reward: 4.679999895393848
    Episode: 247	 Score: 17.589999606832862	 Avg. Reward: 4.822299892213196
    Episode: 247	 Score: 17.589999606832862	 Avg. Reward: 4.822299892213196
    Episode: 248	 Score: 9.609999785199761	 Avg. Reward: 4.876099891010671
    Episode: 248	 Score: 9.609999785199761	 Avg. Reward: 4.876099891010671
    Episode: 249	 Score: 13.539999697357416	 Avg. Reward: 4.984299888592213
    Episode: 249	 Score: 13.539999697357416	 Avg. Reward: 4.984299888592213
    Episode: 250	 Score: 11.559999741613865	 Avg. Reward: 5.066599886752665
    Episode: 250	 Score: 11.559999741613865	 Avg. Reward: 5.066599886752665
    Episode: 251	 Score: 9.279999792575836	 Avg. Reward: 5.138599885143339
    Episode: 251	 Score: 9.279999792575836	 Avg. Reward: 5.138599885143339
    Episode: 252	 Score: 8.239999815821648	 Avg. Reward: 5.200199883766472
    Episode: 252	 Score: 8.239999815821648	 Avg. Reward: 5.200199883766472
    Episode: 253	 Score: 4.999999888241291	 Avg. Reward: 5.225699883196503
    Episode: 253	 Score: 4.999999888241291	 Avg. Reward: 5.225699883196503
    Episode: 254	 Score: 18.069999596104026	 Avg. Reward: 5.388699879553169
    Episode: 254	 Score: 18.069999596104026	 Avg. Reward: 5.388699879553169
    Episode: 255	 Score: 21.7599995136261	 Avg. Reward: 5.590999875031412
    Episode: 255	 Score: 21.7599995136261	 Avg. Reward: 5.590999875031412
    Episode: 256	 Score: 14.279999680817127	 Avg. Reward: 5.709499872382731
    Episode: 256	 Score: 14.279999680817127	 Avg. Reward: 5.709499872382731
    Episode: 257	 Score: 7.689999828115106	 Avg. Reward: 5.775499870907515
    Episode: 257	 Score: 7.689999828115106	 Avg. Reward: 5.775499870907515
    Episode: 258	 Score: 17.489999609068036	 Avg. Reward: 5.948999867029488
    Episode: 258	 Score: 17.489999609068036	 Avg. Reward: 5.948999867029488
    Episode: 259	 Score: 13.539999697357416	 Avg. Reward: 6.013599865585565
    Episode: 259	 Score: 13.539999697357416	 Avg. Reward: 6.013599865585565
    Episode: 260	 Score: 10.469999765977263	 Avg. Reward: 6.096299863737077
    Episode: 260	 Score: 10.469999765977263	 Avg. Reward: 6.096299863737077
    Episode: 261	 Score: 20.929999532178044	 Avg. Reward: 6.287999859452247
    Episode: 261	 Score: 20.929999532178044	 Avg. Reward: 6.287999859452247
    Episode: 262	 Score: 10.199999772012234	 Avg. Reward: 6.383399857319891
    Episode: 262	 Score: 10.199999772012234	 Avg. Reward: 6.383399857319891
    Episode: 263	 Score: 16.939999621361494	 Avg. Reward: 6.552799853533506
    Episode: 263	 Score: 16.939999621361494	 Avg. Reward: 6.552799853533506
    Episode: 264	 Score: 3.5099999215453863	 Avg. Reward: 6.555299853477627
    Episode: 264	 Score: 3.5099999215453863	 Avg. Reward: 6.555299853477627
    Episode: 265	 Score: 13.269999703392386	 Avg. Reward: 6.669199850931764
    Episode: 265	 Score: 13.269999703392386	 Avg. Reward: 6.669199850931764
    Episode: 266	 Score: 14.539999675005674	 Avg. Reward: 6.792099848184734
    Episode: 266	 Score: 14.539999675005674	 Avg. Reward: 6.792099848184734
    Episode: 267	 Score: 13.229999704286456	 Avg. Reward: 6.921599845290184
    Episode: 267	 Score: 13.229999704286456	 Avg. Reward: 6.921599845290184
    Episode: 268	 Score: 11.159999750554562	 Avg. Reward: 7.024899842981249
    Episode: 268	 Score: 11.159999750554562	 Avg. Reward: 7.024899842981249
    Episode: 269	 Score: 16.699999626725912	 Avg. Reward: 7.157599840015173
    Episode: 269	 Score: 16.699999626725912	 Avg. Reward: 7.157599840015173
    Episode: 270	 Score: 17.089999618008733	 Avg. Reward: 7.3058998367004095
    Episode: 270	 Score: 17.089999618008733	 Avg. Reward: 7.3058998367004095
    Episode: 271	 Score: 10.72999976016581	 Avg. Reward: 7.395499834697693
    Episode: 271	 Score: 10.72999976016581	 Avg. Reward: 7.395499834697693
    Episode: 272	 Score: 15.749999647960067	 Avg. Reward: 7.501999832317233
    Episode: 272	 Score: 15.749999647960067	 Avg. Reward: 7.501999832317233
    Episode: 273	 Score: 22.679999493062496	 Avg. Reward: 7.70749982772395
    Episode: 273	 Score: 22.679999493062496	 Avg. Reward: 7.70749982772395
    Episode: 274	 Score: 15.199999660253525	 Avg. Reward: 7.848799824565649
    Episode: 274	 Score: 15.199999660253525	 Avg. Reward: 7.848799824565649
    Episode: 275	 Score: 14.009999686852098	 Avg. Reward: 7.9737998217716815
    Episode: 275	 Score: 14.009999686852098	 Avg. Reward: 7.9737998217716815
    Episode: 276	 Score: 16.26999963633716	 Avg. Reward: 8.122299818452447
    Episode: 276	 Score: 16.26999963633716	 Avg. Reward: 8.122299818452447
    Episode: 277	 Score: 17.399999611079693	 Avg. Reward: 8.275399815030395
    Episode: 277	 Score: 17.399999611079693	 Avg. Reward: 8.275399815030395
    Episode: 278	 Score: 13.829999690875411	 Avg. Reward: 8.395499812345951
    Episode: 278	 Score: 13.829999690875411	 Avg. Reward: 8.395499812345951
    Episode: 279	 Score: 13.569999696686864	 Avg. Reward: 8.516099809650331
    Episode: 279	 Score: 13.569999696686864	 Avg. Reward: 8.516099809650331
    Episode: 280	 Score: 21.779999513179064	 Avg. Reward: 8.721999805048108
    Episode: 280	 Score: 21.779999513179064	 Avg. Reward: 8.721999805048108
    Episode: 281	 Score: 28.129999371245503	 Avg. Reward: 8.984299799185246
    Episode: 281	 Score: 28.129999371245503	 Avg. Reward: 8.984299799185246
    Episode: 282	 Score: 20.929999532178044	 Avg. Reward: 9.156299795340747
    Episode: 282	 Score: 20.929999532178044	 Avg. Reward: 9.156299795340747
    Episode: 283	 Score: 17.399999611079693	 Avg. Reward: 9.300199792124332
    Episode: 283	 Score: 17.399999611079693	 Avg. Reward: 9.300199792124332
    Episode: 284	 Score: 8.659999806433916	 Avg. Reward: 9.353299790937454
    Episode: 284	 Score: 8.659999806433916	 Avg. Reward: 9.353299790937454
    Episode: 285	 Score: 23.88999946601689	 Avg. Reward: 9.554799786433577
    Episode: 285	 Score: 23.88999946601689	 Avg. Reward: 9.554799786433577
    Episode: 286	 Score: 15.52999965287745	 Avg. Reward: 9.683499783556908
    Episode: 286	 Score: 15.52999965287745	 Avg. Reward: 9.683499783556908
    Episode: 287	 Score: 28.239999368786812	 Avg. Reward: 9.935399777926504
    Episode: 287	 Score: 28.239999368786812	 Avg. Reward: 9.935399777926504
    Episode: 288	 Score: 18.97999957576394	 Avg. Reward: 10.099099774267524
    Episode: 288	 Score: 18.97999957576394	 Avg. Reward: 10.099099774267524
    Episode: 289	 Score: 14.189999682828784	 Avg. Reward: 10.225099771451205
    Episode: 289	 Score: 14.189999682828784	 Avg. Reward: 10.225099771451205
    Episode: 290	 Score: 15.359999656677246	 Avg. Reward: 10.349199768677353
    Episode: 290	 Score: 15.359999656677246	 Avg. Reward: 10.349199768677353
    Episode: 291	 Score: 21.0699995290488	 Avg. Reward: 10.542199764363467
    Episode: 291	 Score: 21.0699995290488	 Avg. Reward: 10.542199764363467
    Episode: 292	 Score: 15.589999651536345	 Avg. Reward: 10.672899761442094
    Episode: 292	 Score: 15.589999651536345	 Avg. Reward: 10.672899761442094
    Episode: 293	 Score: 17.479999609291553	 Avg. Reward: 10.817599758207798
    Episode: 293	 Score: 17.479999609291553	 Avg. Reward: 10.817599758207798
    Episode: 294	 Score: 20.239999547600746	 Avg. Reward: 10.974599754698575
    Episode: 294	 Score: 20.239999547600746	 Avg. Reward: 10.974599754698575
    Episode: 295	 Score: 23.9399994648993	 Avg. Reward: 11.174999750219285
    Episode: 295	 Score: 23.9399994648993	 Avg. Reward: 11.174999750219285
    Episode: 296	 Score: 22.799999490380287	 Avg. Reward: 11.362099746037275
    Episode: 296	 Score: 22.799999490380287	 Avg. Reward: 11.362099746037275
    Episode: 297	 Score: 16.119999639689922	 Avg. Reward: 11.467099743690342
    Episode: 297	 Score: 16.119999639689922	 Avg. Reward: 11.467099743690342
    Episode: 298	 Score: 21.399999521672726	 Avg. Reward: 11.658999739401043
    Episode: 298	 Score: 21.399999521672726	 Avg. Reward: 11.658999739401043
    Episode: 299	 Score: 19.509999563917518	 Avg. Reward: 11.791399736441672
    Episode: 299	 Score: 19.509999563917518	 Avg. Reward: 11.791399736441672
    Episode: 300	 Score: 17.509999608621	 Avg. Reward: 11.907099733855576
    Episode: 300	 Score: 17.509999608621	 Avg. Reward: 11.907099733855576
    Episode: 301	 Score: 24.289999457076192	 Avg. Reward: 12.105199729427696
    Episode: 301	 Score: 24.289999457076192	 Avg. Reward: 12.105199729427696
    Episode: 302	 Score: 17.559999607503414	 Avg. Reward: 12.239499726425857
    Episode: 302	 Score: 17.559999607503414	 Avg. Reward: 12.239499726425857
    Episode: 303	 Score: 20.52999954111874	 Avg. Reward: 12.424799722284078
    Episode: 303	 Score: 20.52999954111874	 Avg. Reward: 12.424799722284078
    Episode: 304	 Score: 22.92999948747456	 Avg. Reward: 12.61299971807748
    Episode: 304	 Score: 22.92999948747456	 Avg. Reward: 12.61299971807748
    Episode: 305	 Score: 39.53999911621213	 Avg. Reward: 12.964999710209668
    Episode: 305	 Score: 39.53999911621213	 Avg. Reward: 12.964999710209668
    Episode: 306	 Score: 17.129999617114663	 Avg. Reward: 13.07649970771745
    Episode: 306	 Score: 17.129999617114663	 Avg. Reward: 13.07649970771745
    Episode: 307	 Score: 25.889999421313405	 Avg. Reward: 13.297699702773244
    Episode: 307	 Score: 25.889999421313405	 Avg. Reward: 13.297699702773244
    Episode: 308	 Score: 17.69999960437417	 Avg. Reward: 13.437599699646235
    Episode: 308	 Score: 17.69999960437417	 Avg. Reward: 13.437599699646235
    Episode: 309	 Score: 16.239999637007713	 Avg. Reward: 13.547899697180837
    Episode: 309	 Score: 16.239999637007713	 Avg. Reward: 13.547899697180837
    Episode: 310	 Score: 22.41999949887395	 Avg. Reward: 13.719299693349749
    Episode: 310	 Score: 22.41999949887395	 Avg. Reward: 13.719299693349749
    Episode: 311	 Score: 24.419999454170465	 Avg. Reward: 13.887699689585716
    Episode: 311	 Score: 24.419999454170465	 Avg. Reward: 13.887699689585716
    Episode: 312	 Score: 21.979999508708715	 Avg. Reward: 14.039599686190487
    Episode: 312	 Score: 21.979999508708715	 Avg. Reward: 14.039599686190487
    Episode: 313	 Score: 21.80999951250851	 Avg. Reward: 14.225299682039768
    Episode: 313	 Score: 21.80999951250851	 Avg. Reward: 14.225299682039768
    Episode: 314	 Score: 18.909999577328563	 Avg. Reward: 14.37059967879206
    Episode: 314	 Score: 18.909999577328563	 Avg. Reward: 14.37059967879206
    Episode: 315	 Score: 13.00999970920384	 Avg. Reward: 14.484899676237255
    Episode: 315	 Score: 13.00999970920384	 Avg. Reward: 14.484899676237255
    Episode: 316	 Score: 25.919999420642853	 Avg. Reward: 14.68539967175573
    Episode: 316	 Score: 25.919999420642853	 Avg. Reward: 14.68539967175573
    Episode: 317	 Score: 18.27999959141016	 Avg. Reward: 14.79269966935739
    Episode: 317	 Score: 18.27999959141016	 Avg. Reward: 14.79269966935739
    Episode: 318	 Score: 20.499999541789293	 Avg. Reward: 14.932699666228146
    Episode: 318	 Score: 20.499999541789293	 Avg. Reward: 14.932699666228146
    Episode: 319	 Score: 30.219999324530363	 Avg. Reward: 15.1252996619232
    Episode: 319	 Score: 30.219999324530363	 Avg. Reward: 15.1252996619232
    Episode: 320	 Score: 22.299999501556158	 Avg. Reward: 15.285199658349157
    Episode: 320	 Score: 22.299999501556158	 Avg. Reward: 15.285199658349157
    Episode: 321	 Score: 15.639999650418758	 Avg. Reward: 15.407799655608834
    Episode: 321	 Score: 15.639999650418758	 Avg. Reward: 15.407799655608834
    Episode: 322	 Score: 19.55999956279993	 Avg. Reward: 15.539299652669579
    Episode: 322	 Score: 19.55999956279993	 Avg. Reward: 15.539299652669579
    Episode: 323	 Score: 17.85999960079789	 Avg. Reward: 15.667299649808555
    Episode: 323	 Score: 17.85999960079789	 Avg. Reward: 15.667299649808555
    Episode: 324	 Score: 27.64999938197434	 Avg. Reward: 15.905399644486605
    Episode: 324	 Score: 27.64999938197434	 Avg. Reward: 15.905399644486605
    Episode: 325	 Score: 18.54999958537519	 Avg. Reward: 16.010599642135205
    Episode: 325	 Score: 18.54999958537519	 Avg. Reward: 16.010599642135205
    Episode: 326	 Score: 21.979999508708715	 Avg. Reward: 16.119899639692157
    Episode: 326	 Score: 21.979999508708715	 Avg. Reward: 16.119899639692157
    Episode: 327	 Score: 20.91999953240156	 Avg. Reward: 16.28079963609576
    Episode: 327	 Score: 20.91999953240156	 Avg. Reward: 16.28079963609576
    Episode: 328	 Score: 15.44999965466559	 Avg. Reward: 16.42269963292405
    Episode: 328	 Score: 15.44999965466559	 Avg. Reward: 16.42269963292405
    Episode: 329	 Score: 26.32999941147864	 Avg. Reward: 16.61709962857887
    Episode: 329	 Score: 26.32999941147864	 Avg. Reward: 16.61709962857887
    Episode: 330	 Score: 33.33999925479293	 Avg. Reward: 16.920099621806294
    Episode: 330	 Score: 33.33999925479293	 Avg. Reward: 16.920099621806294
    Episode: 331	 Score: 20.029999552294612	 Avg. Reward: 17.078299618270247
    Episode: 331	 Score: 20.029999552294612	 Avg. Reward: 17.078299618270247
    Episode: 332	 Score: 21.589999517425895	 Avg. Reward: 17.19999961555004
    Episode: 332	 Score: 21.589999517425895	 Avg. Reward: 17.19999961555004
    Episode: 333	 Score: 30.779999312013388	 Avg. Reward: 17.311299613062292
    Episode: 333	 Score: 30.779999312013388	 Avg. Reward: 17.311299613062292
    Episode: 334	 Score: 24.769999446347356	 Avg. Reward: 17.455199609845877
    Episode: 334	 Score: 24.769999446347356	 Avg. Reward: 17.455199609845877
    Episode: 335	 Score: 14.32999967969954	 Avg. Reward: 17.492799609005452
    Episode: 335	 Score: 14.32999967969954	 Avg. Reward: 17.492799609005452
    Episode: 336	 Score: 13.39999970048666	 Avg. Reward: 17.46059960972518
    Episode: 336	 Score: 13.39999970048666	 Avg. Reward: 17.46059960972518
    Episode: 337	 Score: 15.699999649077654	 Avg. Reward: 17.556699607577176
    Episode: 337	 Score: 15.699999649077654	 Avg. Reward: 17.556699607577176
    Episode: 338	 Score: 19.82999955676496	 Avg. Reward: 17.6930996045284
    Episode: 338	 Score: 19.82999955676496	 Avg. Reward: 17.6930996045284
    Episode: 339	 Score: 29.369999343529344	 Avg. Reward: 17.88319960027933
    Episode: 339	 Score: 29.369999343529344	 Avg. Reward: 17.88319960027933
    Episode: 340	 Score: 19.759999558329582	 Avg. Reward: 17.963299598488955
    Episode: 340	 Score: 19.759999558329582	 Avg. Reward: 17.963299598488955
    Episode: 341	 Score: 14.549999674782157	 Avg. Reward: 17.96169959852472
    Episode: 341	 Score: 14.549999674782157	 Avg. Reward: 17.96169959852472
    Episode: 342	 Score: 23.829999467357993	 Avg. Reward: 18.12319959491491
    Episode: 342	 Score: 23.829999467357993	 Avg. Reward: 18.12319959491491
    Episode: 343	 Score: 20.089999550953507	 Avg. Reward: 18.26899959165603
    Episode: 343	 Score: 20.089999550953507	 Avg. Reward: 18.26899959165603
    Episode: 344	 Score: 27.329999389126897	 Avg. Reward: 18.403799588643015
    Episode: 344	 Score: 27.329999389126897	 Avg. Reward: 18.403799588643015
    Episode: 345	 Score: 24.059999462217093	 Avg. Reward: 18.502399586439132
    Episode: 345	 Score: 24.059999462217093	 Avg. Reward: 18.502399586439132
    Episode: 346	 Score: 5.939999867230654	 Avg. Reward: 18.48099958691746
    Episode: 346	 Score: 5.939999867230654	 Avg. Reward: 18.48099958691746
    Episode: 347	 Score: 16.30999963544309	 Avg. Reward: 18.46819958720356
    Episode: 347	 Score: 16.30999963544309	 Avg. Reward: 18.46819958720356
    Episode: 348	 Score: 18.659999582916498	 Avg. Reward: 18.55869958518073
    Episode: 348	 Score: 18.659999582916498	 Avg. Reward: 18.55869958518073
    Episode: 349	 Score: 17.379999611526728	 Avg. Reward: 18.59709958432242
    Episode: 349	 Score: 17.379999611526728	 Avg. Reward: 18.59709958432242
    Episode: 350	 Score: 12.879999712109566	 Avg. Reward: 18.61029958402738
    Episode: 350	 Score: 12.879999712109566	 Avg. Reward: 18.61029958402738
    Episode: 351	 Score: 18.249999592080712	 Avg. Reward: 18.69999958202243
    Episode: 351	 Score: 18.249999592080712	 Avg. Reward: 18.69999958202243
    Episode: 352	 Score: 20.469999542459846	 Avg. Reward: 18.82229957928881
    Episode: 352	 Score: 20.469999542459846	 Avg. Reward: 18.82229957928881
    Episode: 353	 Score: 20.889999533072114	 Avg. Reward: 18.98119957573712
    Episode: 353	 Score: 20.889999533072114	 Avg. Reward: 18.98119957573712
    Episode: 354	 Score: 17.359999611973763	 Avg. Reward: 18.974099575895817
    Episode: 354	 Score: 17.359999611973763	 Avg. Reward: 18.974099575895817
    Episode: 355	 Score: 17.979999598115683	 Avg. Reward: 18.936299576740712
    Episode: 355	 Score: 17.979999598115683	 Avg. Reward: 18.936299576740712
    Episode: 356	 Score: 24.799999445676804	 Avg. Reward: 19.04149957438931
    Episode: 356	 Score: 24.799999445676804	 Avg. Reward: 19.04149957438931
    Episode: 357	 Score: 17.729999603703618	 Avg. Reward: 19.141899572145192
    Episode: 357	 Score: 17.729999603703618	 Avg. Reward: 19.141899572145192
    Episode: 358	 Score: 29.879999332129955	 Avg. Reward: 19.26579956937581
    Episode: 358	 Score: 29.879999332129955	 Avg. Reward: 19.26579956937581
    Episode: 359	 Score: 15.679999649524689	 Avg. Reward: 19.287199568897485
    Episode: 359	 Score: 15.679999649524689	 Avg. Reward: 19.287199568897485
    Episode: 360	 Score: 23.84999946691096	 Avg. Reward: 19.420999565906822
    Episode: 360	 Score: 23.84999946691096	 Avg. Reward: 19.420999565906822
    Episode: 361	 Score: 22.73999949172139	 Avg. Reward: 19.439099565502257
    Episode: 361	 Score: 22.73999949172139	 Avg. Reward: 19.439099565502257
    Episode: 362	 Score: 23.909999465569854	 Avg. Reward: 19.576199562437832
    Episode: 362	 Score: 23.909999465569854	 Avg. Reward: 19.576199562437832
    Episode: 363	 Score: 24.119999460875988	 Avg. Reward: 19.64799956083298
    Episode: 363	 Score: 24.119999460875988	 Avg. Reward: 19.64799956083298
    Episode: 364	 Score: 22.599999494850636	 Avg. Reward: 19.83889955656603
    Episode: 364	 Score: 22.599999494850636	 Avg. Reward: 19.83889955656603
    Episode: 365	 Score: 25.55999942868948	 Avg. Reward: 19.961799553819002
    Episode: 365	 Score: 25.55999942868948	 Avg. Reward: 19.961799553819002
    Episode: 366	 Score: 29.459999341517687	 Avg. Reward: 20.11099955048412
    Episode: 366	 Score: 29.459999341517687	 Avg. Reward: 20.11099955048412
    Episode: 367	 Score: 21.159999527037144	 Avg. Reward: 20.19029954871163
    Episode: 367	 Score: 21.159999527037144	 Avg. Reward: 20.19029954871163
    Episode: 368	 Score: 17.23999961465597	 Avg. Reward: 20.25109954735264
    Episode: 368	 Score: 17.23999961465597	 Avg. Reward: 20.25109954735264
    Episode: 369	 Score: 20.29999954625964	 Avg. Reward: 20.287099546547978
    Episode: 369	 Score: 20.29999954625964	 Avg. Reward: 20.287099546547978
    Episode: 370	 Score: 32.38999927602708	 Avg. Reward: 20.440099543128163
    Episode: 370	 Score: 32.38999927602708	 Avg. Reward: 20.440099543128163
    Episode: 371	 Score: 25.16999943740666	 Avg. Reward: 20.58449953990057
    Episode: 371	 Score: 25.16999943740666	 Avg. Reward: 20.58449953990057
    Episode: 372	 Score: 29.069999350234866	 Avg. Reward: 20.71769953692332
    Episode: 372	 Score: 29.069999350234866	 Avg. Reward: 20.71769953692332
    Episode: 373	 Score: 19.00999957509339	 Avg. Reward: 20.680999537743627
    Episode: 373	 Score: 19.00999957509339	 Avg. Reward: 20.680999537743627
    Episode: 374	 Score: 20.94999953173101	 Avg. Reward: 20.738499536458402
    Episode: 374	 Score: 20.94999953173101	 Avg. Reward: 20.738499536458402
    Episode: 375	 Score: 33.09999926015735	 Avg. Reward: 20.929399532191454
    Episode: 375	 Score: 33.09999926015735	 Avg. Reward: 20.929399532191454
    Episode: 376	 Score: 28.279999367892742	 Avg. Reward: 21.04949952950701
    Episode: 376	 Score: 28.279999367892742	 Avg. Reward: 21.04949952950701
    Episode: 377	 Score: 29.169999347999692	 Avg. Reward: 21.16719952687621
    Episode: 377	 Score: 29.169999347999692	 Avg. Reward: 21.16719952687621
    Episode: 378	 Score: 18.96999957598746	 Avg. Reward: 21.21859952572733
    Episode: 378	 Score: 18.96999957598746	 Avg. Reward: 21.21859952572733
    Episode: 379	 Score: 17.329999612644315	 Avg. Reward: 21.256199524886906
    Episode: 379	 Score: 17.329999612644315	 Avg. Reward: 21.256199524886906
    Episode: 380	 Score: 27.739999379962683	 Avg. Reward: 21.31579952355474
    Episode: 380	 Score: 27.739999379962683	 Avg. Reward: 21.31579952355474
    Episode: 381	 Score: 29.39999934285879	 Avg. Reward: 21.328499523270875
    Episode: 381	 Score: 29.39999934285879	 Avg. Reward: 21.328499523270875
    Episode: 382	 Score: 30.159999325871468	 Avg. Reward: 21.420799521207808
    Episode: 382	 Score: 30.159999325871468	 Avg. Reward: 21.420799521207808
    Episode: 383	 Score: 25.569999428465962	 Avg. Reward: 21.502499519381672
    Episode: 383	 Score: 25.569999428465962	 Avg. Reward: 21.502499519381672
    Episode: 384	 Score: 17.099999617785215	 Avg. Reward: 21.586899517495183
    Episode: 384	 Score: 17.099999617785215	 Avg. Reward: 21.586899517495183
    Episode: 385	 Score: 33.99999924004078	 Avg. Reward: 21.687999515235425
    Episode: 385	 Score: 33.99999924004078	 Avg. Reward: 21.687999515235425
    Episode: 386	 Score: 34.43999923020601	 Avg. Reward: 21.87709951100871
    Episode: 386	 Score: 34.43999923020601	 Avg. Reward: 21.87709951100871
    Episode: 387	 Score: 39.639999113976955	 Avg. Reward: 21.99109950846061
    Episode: 387	 Score: 39.639999113976955	 Avg. Reward: 21.99109950846061
    Episode: 388	 Score: 24.419999454170465	 Avg. Reward: 22.045499507244678
    Episode: 388	 Score: 24.419999454170465	 Avg. Reward: 22.045499507244678
    Episode: 389	 Score: 25.689999425783753	 Avg. Reward: 22.160499504674227
    Episode: 389	 Score: 25.689999425783753	 Avg. Reward: 22.160499504674227
    Episode: 390	 Score: 21.239999525249004	 Avg. Reward: 22.219299503359945
    Episode: 390	 Score: 21.239999525249004	 Avg. Reward: 22.219299503359945
    Episode: 391	 Score: 30.779999312013388	 Avg. Reward: 22.31639950118959
    Episode: 391	 Score: 30.779999312013388	 Avg. Reward: 22.31639950118959
    Episode: 392	 Score: 27.969999374821782	 Avg. Reward: 22.440199498422444
    Episode: 392	 Score: 27.969999374821782	 Avg. Reward: 22.440199498422444
    Episode: 393	 Score: 25.359999433159828	 Avg. Reward: 22.518999496661127
    Episode: 393	 Score: 25.359999433159828	 Avg. Reward: 22.518999496661127
    Episode: 394	 Score: 30.719999313354492	 Avg. Reward: 22.623799494318664
    Episode: 394	 Score: 30.719999313354492	 Avg. Reward: 22.623799494318664
    Episode: 395	 Score: 27.349999388679862	 Avg. Reward: 22.65789949355647
    Episode: 395	 Score: 27.349999388679862	 Avg. Reward: 22.65789949355647
    Episode: 396	 Score: 24.309999456629157	 Avg. Reward: 22.67299949321896
    Episode: 396	 Score: 24.309999456629157	 Avg. Reward: 22.67299949321896
    Episode: 397	 Score: 22.219999503344297	 Avg. Reward: 22.733999491855503
    Episode: 397	 Score: 22.219999503344297	 Avg. Reward: 22.733999491855503
    Episode: 398	 Score: 23.38999947719276	 Avg. Reward: 22.753899491410703
    Episode: 398	 Score: 23.38999947719276	 Avg. Reward: 22.753899491410703
    Episode: 399	 Score: 27.319999389350414	 Avg. Reward: 22.83199948966503
    Episode: 399	 Score: 27.319999389350414	 Avg. Reward: 22.83199948966503
    Episode: 400	 Score: 26.01999941840768	 Avg. Reward: 22.917099487762897
    Episode: 400	 Score: 26.01999941840768	 Avg. Reward: 22.917099487762897
    Episode: 401	 Score: 32.129999281838536	 Avg. Reward: 22.995499486010523
    Episode: 401	 Score: 32.129999281838536	 Avg. Reward: 22.995499486010523
    Episode: 402	 Score: 31.709999291226268	 Avg. Reward: 23.13699948284775
    Episode: 402	 Score: 31.709999291226268	 Avg. Reward: 23.13699948284775
    Episode: 403	 Score: 32.84999926574528	 Avg. Reward: 23.260199480094016
    Episode: 403	 Score: 32.84999926574528	 Avg. Reward: 23.260199480094016
    Episode: 404	 Score: 24.189999459311366	 Avg. Reward: 23.272799479812385
    Episode: 404	 Score: 24.189999459311366	 Avg. Reward: 23.272799479812385
    Episode: 405	 Score: 24.289999457076192	 Avg. Reward: 23.120299483221025
    Episode: 405	 Score: 24.289999457076192	 Avg. Reward: 23.120299483221025
    Episode: 406	 Score: 27.389999387785792	 Avg. Reward: 23.222899480927737
    Episode: 406	 Score: 27.389999387785792	 Avg. Reward: 23.222899480927737
    Episode: 407	 Score: 24.609999449923635	 Avg. Reward: 23.210099481213838
    Episode: 407	 Score: 24.609999449923635	 Avg. Reward: 23.210099481213838
    Episode: 408	 Score: 30.069999327883124	 Avg. Reward: 23.333799478448928
    Episode: 408	 Score: 30.069999327883124	 Avg. Reward: 23.333799478448928
    Episode: 409	 Score: 21.939999509602785	 Avg. Reward: 23.390799477174877
    Episode: 409	 Score: 21.939999509602785	 Avg. Reward: 23.390799477174877
    Episode: 410	 Score: 26.469999408349395	 Avg. Reward: 23.431299476269633
    Episode: 410	 Score: 26.469999408349395	 Avg. Reward: 23.431299476269633
    Episode: 411	 Score: 24.389999454841018	 Avg. Reward: 23.430999476276337
    Episode: 411	 Score: 24.389999454841018	 Avg. Reward: 23.430999476276337
    Episode: 412	 Score: 29.479999341070652	 Avg. Reward: 23.505999474599957
    Episode: 412	 Score: 29.479999341070652	 Avg. Reward: 23.505999474599957
    Episode: 413	 Score: 16.209999637678266	 Avg. Reward: 23.449999475851655
    Episode: 413	 Score: 16.209999637678266	 Avg. Reward: 23.449999475851655
    Episode: 414	 Score: 24.079999461770058	 Avg. Reward: 23.50169947469607
    Episode: 414	 Score: 24.079999461770058	 Avg. Reward: 23.50169947469607
    Episode: 415	 Score: 24.369999455288053	 Avg. Reward: 23.615299472156913
    Episode: 415	 Score: 24.369999455288053	 Avg. Reward: 23.615299472156913
    Episode: 416	 Score: 26.68999940343201	 Avg. Reward: 23.622999471984805
    Episode: 416	 Score: 26.68999940343201	 Avg. Reward: 23.622999471984805
    Episode: 417	 Score: 26.579999405890703	 Avg. Reward: 23.70599947012961
    Episode: 417	 Score: 26.579999405890703	 Avg. Reward: 23.70599947012961
    Episode: 418	 Score: 34.39999923110008	 Avg. Reward: 23.844999467022717
    Episode: 418	 Score: 34.39999923110008	 Avg. Reward: 23.844999467022717
    Episode: 419	 Score: 21.929999509826303	 Avg. Reward: 23.762099468875675
    Episode: 419	 Score: 21.929999509826303	 Avg. Reward: 23.762099468875675
    Episode: 420	 Score: 39.46999911777675	 Avg. Reward: 23.933799465037882
    Episode: 420	 Score: 39.46999911777675	 Avg. Reward: 23.933799465037882
    Episode: 421	 Score: 29.93999933078885	 Avg. Reward: 24.076799461841585
    Episode: 421	 Score: 29.93999933078885	 Avg. Reward: 24.076799461841585
    Episode: 422	 Score: 16.209999637678266	 Avg. Reward: 24.043299462590365
    Episode: 422	 Score: 16.209999637678266	 Avg. Reward: 24.043299462590365
    Episode: 423	 Score: 32.52999927289784	 Avg. Reward: 24.189999459311366
    Episode: 423	 Score: 32.52999927289784	 Avg. Reward: 24.189999459311366
    Episode: 424	 Score: 16.61999962851405	 Avg. Reward: 24.079699461776762
    Episode: 424	 Score: 16.61999962851405	 Avg. Reward: 24.079699461776762
    Episode: 425	 Score: 21.10999952815473	 Avg. Reward: 24.10529946120456
    Episode: 425	 Score: 21.10999952815473	 Avg. Reward: 24.10529946120456
    Episode: 426	 Score: 33.89999924227595	 Avg. Reward: 24.22449945854023
    Episode: 426	 Score: 33.89999924227595	 Avg. Reward: 24.22449945854023
    Episode: 427	 Score: 25.529999429360032	 Avg. Reward: 24.270599457509817
    Episode: 427	 Score: 25.529999429360032	 Avg. Reward: 24.270599457509817
    Episode: 428	 Score: 20.069999551400542	 Avg. Reward: 24.316799456477167
    Episode: 428	 Score: 20.069999551400542	 Avg. Reward: 24.316799456477167
    Episode: 429	 Score: 30.339999321848154	 Avg. Reward: 24.35689945558086
    Episode: 429	 Score: 30.339999321848154	 Avg. Reward: 24.35689945558086
    Episode: 430	 Score: 29.409999342635274	 Avg. Reward: 24.317599456459284
    Episode: 430	 Score: 29.409999342635274	 Avg. Reward: 24.317599456459284
    Episode: 431	 Score: 28.249999368563294	 Avg. Reward: 24.39979945462197
    Episode: 431	 Score: 28.249999368563294	 Avg. Reward: 24.39979945462197
    Episode: 432	 Score: 26.979999396950006	 Avg. Reward: 24.45369945341721
    Episode: 432	 Score: 26.979999396950006	 Avg. Reward: 24.45369945341721
    Episode: 433	 Score: 31.5199992954731	 Avg. Reward: 24.46109945325181
    Episode: 433	 Score: 31.5199992954731	 Avg. Reward: 24.46109945325181
    Episode: 434	 Score: 26.21999941393733	 Avg. Reward: 24.47559945292771
    Episode: 434	 Score: 26.21999941393733	 Avg. Reward: 24.47559945292771
    Episode: 435	 Score: 28.629999360069633	 Avg. Reward: 24.61859944973141
    Episode: 435	 Score: 28.629999360069633	 Avg. Reward: 24.61859944973141
    Episode: 436	 Score: 28.549999361857772	 Avg. Reward: 24.77009944634512
    Episode: 436	 Score: 28.549999361857772	 Avg. Reward: 24.77009944634512
    Episode: 437	 Score: 27.29999938979745	 Avg. Reward: 24.886099443752318
    Episode: 437	 Score: 27.29999938979745	 Avg. Reward: 24.886099443752318
    Episode: 438	 Score: 32.80999926663935	 Avg. Reward: 25.01589944085106
    Episode: 438	 Score: 32.80999926663935	 Avg. Reward: 25.01589944085106
    Episode: 439	 Score: 26.849999399855733	 Avg. Reward: 24.990699441414325
    Episode: 439	 Score: 26.849999399855733	 Avg. Reward: 24.990699441414325
    Episode: 440	 Score: 27.709999380633235	 Avg. Reward: 25.070199439637364
    Episode: 440	 Score: 27.709999380633235	 Avg. Reward: 25.070199439637364
    Episode: 441	 Score: 36.00999919511378	 Avg. Reward: 25.28479943484068
    Episode: 441	 Score: 36.00999919511378	 Avg. Reward: 25.28479943484068
    Episode: 442	 Score: 25.069999439641833	 Avg. Reward: 25.297199434563517
    Episode: 442	 Score: 25.069999439641833	 Avg. Reward: 25.297199434563517
    Episode: 443	 Score: 29.93999933078885	 Avg. Reward: 25.39569943236187
    Episode: 443	 Score: 29.93999933078885	 Avg. Reward: 25.39569943236187
    Episode: 444	 Score: 28.529999362304807	 Avg. Reward: 25.40769943209365
    Episode: 444	 Score: 28.529999362304807	 Avg. Reward: 25.40769943209365
    Episode: 445	 Score: 23.869999466463923	 Avg. Reward: 25.405799432136117
    Episode: 445	 Score: 23.869999466463923	 Avg. Reward: 25.405799432136117
    Episode: 446	 Score: 18.729999581351876	 Avg. Reward: 25.53369942927733
    Episode: 446	 Score: 18.729999581351876	 Avg. Reward: 25.53369942927733
    Episode: 447	 Score: 28.799999356269836	 Avg. Reward: 25.658599426485598
    Episode: 447	 Score: 28.799999356269836	 Avg. Reward: 25.658599426485598
    Episode: 448	 Score: 28.879999354481697	 Avg. Reward: 25.76079942420125
    Episode: 448	 Score: 28.879999354481697	 Avg. Reward: 25.76079942420125
    Episode: 449	 Score: 33.2399992570281	 Avg. Reward: 25.919399420656262
    Episode: 449	 Score: 33.2399992570281	 Avg. Reward: 25.919399420656262
    Episode: 450	 Score: 34.14999923668802	 Avg. Reward: 26.132099415902047
    Episode: 450	 Score: 34.14999923668802	 Avg. Reward: 26.132099415902047
    Episode: 451	 Score: 24.919999442994595	 Avg. Reward: 26.198799414411187
    Episode: 451	 Score: 24.919999442994595	 Avg. Reward: 26.198799414411187
    Episode: 452	 Score: 33.61999924853444	 Avg. Reward: 26.330299411471934
    Episode: 452	 Score: 33.61999924853444	 Avg. Reward: 26.330299411471934
    Episode: 453	 Score: 25.459999430924654	 Avg. Reward: 26.375999410450458
    Episode: 453	 Score: 25.459999430924654	 Avg. Reward: 26.375999410450458
    Episode: 454	 Score: 24.029999462887645	 Avg. Reward: 26.442699408959598
    Episode: 454	 Score: 24.029999462887645	 Avg. Reward: 26.442699408959598
    Episode: 455	 Score: 31.819999288767576	 Avg. Reward: 26.581099405866116
    Episode: 455	 Score: 31.819999288767576	 Avg. Reward: 26.581099405866116
    Episode: 456	 Score: 25.9499994199723	 Avg. Reward: 26.59259940560907
    Episode: 456	 Score: 25.9499994199723	 Avg. Reward: 26.59259940560907
    Episode: 457	 Score: 23.219999480992556	 Avg. Reward: 26.64749940438196
    Episode: 457	 Score: 23.219999480992556	 Avg. Reward: 26.64749940438196
    Episode: 458	 Score: 24.389999454841018	 Avg. Reward: 26.59259940560907
    Episode: 458	 Score: 24.389999454841018	 Avg. Reward: 26.59259940560907
    Episode: 459	 Score: 23.749999469146132	 Avg. Reward: 26.673299403805284
    Episode: 459	 Score: 23.749999469146132	 Avg. Reward: 26.673299403805284
    Episode: 460	 Score: 30.12999932654202	 Avg. Reward: 26.736099402401596
    Episode: 460	 Score: 30.12999932654202	 Avg. Reward: 26.736099402401596
    Episode: 461	 Score: 23.58999947272241	 Avg. Reward: 26.744599402211605
    Episode: 461	 Score: 23.58999947272241	 Avg. Reward: 26.744599402211605
    Episode: 462	 Score: 26.06999941729009	 Avg. Reward: 26.76619940172881
    Episode: 462	 Score: 26.06999941729009	 Avg. Reward: 26.76619940172881
    Episode: 463	 Score: 35.55999920517206	 Avg. Reward: 26.88059939917177
    Episode: 463	 Score: 35.55999920517206	 Avg. Reward: 26.88059939917177
    Episode: 464	 Score: 26.739999402314425	 Avg. Reward: 26.921999398246406
    Episode: 464	 Score: 26.739999402314425	 Avg. Reward: 26.921999398246406
    Episode: 465	 Score: 28.4199993647635	 Avg. Reward: 26.95059939760715
    Episode: 465	 Score: 28.4199993647635	 Avg. Reward: 26.95059939760715
    Episode: 466	 Score: 24.119999460875988	 Avg. Reward: 26.89719939880073
    Episode: 466	 Score: 24.119999460875988	 Avg. Reward: 26.89719939880073
    Episode: 467	 Score: 29.169999347999692	 Avg. Reward: 26.977299397010356
    Episode: 467	 Score: 29.169999347999692	 Avg. Reward: 26.977299397010356
    Episode: 468	 Score: 30.32999932207167	 Avg. Reward: 27.10819939408451
    Episode: 468	 Score: 30.32999932207167	 Avg. Reward: 27.10819939408451
    Episode: 469	 Score: 29.0499993506819	 Avg. Reward: 27.195699392128734
    Episode: 469	 Score: 29.0499993506819	 Avg. Reward: 27.195699392128734
    Episode: 470	 Score: 34.059999238699675	 Avg. Reward: 27.21239939175546
    Episode: 470	 Score: 34.059999238699675	 Avg. Reward: 27.21239939175546
    Episode: 471	 Score: 35.029999217018485	 Avg. Reward: 27.31099938955158
    Episode: 471	 Score: 35.029999217018485	 Avg. Reward: 27.31099938955158
    Episode: 472	 Score: 36.36999918706715	 Avg. Reward: 27.383999387919904
    Episode: 472	 Score: 36.36999918706715	 Avg. Reward: 27.383999387919904
    Episode: 473	 Score: 23.38999947719276	 Avg. Reward: 27.427799386940897
    Episode: 473	 Score: 23.38999947719276	 Avg. Reward: 27.427799386940897
    Episode: 474	 Score: 34.04999923892319	 Avg. Reward: 27.55879938401282
    Episode: 474	 Score: 34.04999923892319	 Avg. Reward: 27.55879938401282
    Episode: 475	 Score: 29.529999339953065	 Avg. Reward: 27.523099384810777
    Episode: 475	 Score: 29.529999339953065	 Avg. Reward: 27.523099384810777
    Episode: 476	 Score: 30.509999318048358	 Avg. Reward: 27.545399384312333
    Episode: 476	 Score: 30.509999318048358	 Avg. Reward: 27.545399384312333
    Episode: 477	 Score: 34.58999922685325	 Avg. Reward: 27.599599383100866
    Episode: 477	 Score: 34.58999922685325	 Avg. Reward: 27.599599383100866
    Episode: 478	 Score: 22.369999499991536	 Avg. Reward: 27.633599382340908
    Episode: 478	 Score: 22.369999499991536	 Avg. Reward: 27.633599382340908
    Episode: 479	 Score: 26.589999405667186	 Avg. Reward: 27.726199380271137
    Episode: 479	 Score: 26.589999405667186	 Avg. Reward: 27.726199380271137
    Episode: 480	 Score: 21.619999516755342	 Avg. Reward: 27.664999381639063
    Episode: 480	 Score: 21.619999516755342	 Avg. Reward: 27.664999381639063
    Episode: 481	 Score: 29.219999346882105	 Avg. Reward: 27.663199381679295
    Episode: 481	 Score: 29.219999346882105	 Avg. Reward: 27.663199381679295
    Episode: 482	 Score: 27.789999378845096	 Avg. Reward: 27.63949938220903
    Episode: 482	 Score: 27.789999378845096	 Avg. Reward: 27.63949938220903
    Episode: 483	 Score: 24.369999455288053	 Avg. Reward: 27.627499382477254
    Episode: 483	 Score: 24.369999455288053	 Avg. Reward: 27.627499382477254
    Episode: 484	 Score: 27.33999938890338	 Avg. Reward: 27.729899380188435
    Episode: 484	 Score: 27.33999938890338	 Avg. Reward: 27.729899380188435
    Episode: 485	 Score: 27.409999387338758	 Avg. Reward: 27.663999381661416
    Episode: 485	 Score: 27.409999387338758	 Avg. Reward: 27.663999381661416
    Episode: 486	 Score: 26.579999405890703	 Avg. Reward: 27.585399383418263
    Episode: 486	 Score: 26.579999405890703	 Avg. Reward: 27.585399383418263
    Episode: 487	 Score: 24.619999449700117	 Avg. Reward: 27.435199386775494
    Episode: 487	 Score: 24.619999449700117	 Avg. Reward: 27.435199386775494
    Episode: 488	 Score: 32.3799992762506	 Avg. Reward: 27.514799384996294
    Episode: 488	 Score: 32.3799992762506	 Avg. Reward: 27.514799384996294
    Episode: 489	 Score: 39.42999911867082	 Avg. Reward: 27.652199381925165
    Episode: 489	 Score: 39.42999911867082	 Avg. Reward: 27.652199381925165
    Episode: 490	 Score: 33.40999925322831	 Avg. Reward: 27.77389937920496
    Episode: 490	 Score: 33.40999925322831	 Avg. Reward: 27.77389937920496
    Episode: 491	 Score: 36.3999991863966	 Avg. Reward: 27.83009937794879
    Episode: 491	 Score: 36.3999991863966	 Avg. Reward: 27.83009937794879
    Episode: 492	 Score: 31.029999306425452	 Avg. Reward: 27.860699377264826
    Episode: 492	 Score: 31.029999306425452	 Avg. Reward: 27.860699377264826
    Episode: 493	 Score: 30.079999327659607	 Avg. Reward: 27.907899376209826
    Episode: 493	 Score: 30.079999327659607	 Avg. Reward: 27.907899376209826
    Episode: 494	 Score: 22.299999501556158	 Avg. Reward: 27.823699378091842
    Episode: 494	 Score: 22.299999501556158	 Avg. Reward: 27.823699378091842
    Episode: 495	 Score: 22.559999495744705	 Avg. Reward: 27.77579937916249
    Episode: 495	 Score: 22.559999495744705	 Avg. Reward: 27.77579937916249
    Episode: 496	 Score: 26.969999397173524	 Avg. Reward: 27.802399378567934
    Episode: 496	 Score: 26.969999397173524	 Avg. Reward: 27.802399378567934
    Episode: 497	 Score: 25.299999434500933	 Avg. Reward: 27.8331993778795
    Episode: 497	 Score: 25.299999434500933	 Avg. Reward: 27.8331993778795
    Episode: 498	 Score: 32.359999276697636	 Avg. Reward: 27.92289937587455
    Episode: 498	 Score: 32.359999276697636	 Avg. Reward: 27.92289937587455
    Episode: 499	 Score: 18.799999579787254	 Avg. Reward: 27.837699377778918
    Episode: 499	 Score: 18.799999579787254	 Avg. Reward: 27.837699377778918
    Episode: 500	 Score: 32.3799992762506	 Avg. Reward: 27.901299376357347
    Episode: 500	 Score: 32.3799992762506	 Avg. Reward: 27.901299376357347
    Episode: 501	 Score: 20.789999535307288	 Avg. Reward: 27.787899378892035
    Episode: 501	 Score: 20.789999535307288	 Avg. Reward: 27.787899378892035
    Episode: 502	 Score: 21.249999525025487	 Avg. Reward: 27.683299381230025
    Episode: 502	 Score: 21.249999525025487	 Avg. Reward: 27.683299381230025
    Episode: 503	 Score: 22.569999495521188	 Avg. Reward: 27.580499383527787
    Episode: 503	 Score: 22.569999495521188	 Avg. Reward: 27.580499383527787
    Episode: 504	 Score: 27.939999375492334	 Avg. Reward: 27.617999382689597
    Episode: 504	 Score: 27.939999375492334	 Avg. Reward: 27.617999382689597
    Episode: 505	 Score: 22.199999503791332	 Avg. Reward: 27.597099383156745
    Episode: 505	 Score: 22.199999503791332	 Avg. Reward: 27.597099383156745
    Episode: 506	 Score: 37.33999916538596	 Avg. Reward: 27.69659938093275
    Episode: 506	 Score: 37.33999916538596	 Avg. Reward: 27.69659938093275
    Episode: 507	 Score: 16.749999625608325	 Avg. Reward: 27.617999382689597
    Episode: 507	 Score: 16.749999625608325	 Avg. Reward: 27.617999382689597
    Episode: 508	 Score: 23.0799994841218	 Avg. Reward: 27.548099384251984
    Episode: 508	 Score: 23.0799994841218	 Avg. Reward: 27.548099384251984
    Episode: 509	 Score: 1.339999970048666	 Avg. Reward: 27.34209938885644
    Episode: 509	 Score: 1.339999970048666	 Avg. Reward: 27.34209938885644
    Episode: 510	 Score: 21.669999515637755	 Avg. Reward: 27.294099389929325
    Episode: 510	 Score: 21.669999515637755	 Avg. Reward: 27.294099389929325
    Episode: 511	 Score: 35.44999920763075	 Avg. Reward: 27.40469938745722
    Episode: 511	 Score: 35.44999920763075	 Avg. Reward: 27.40469938745722
    Episode: 512	 Score: 21.519999518990517	 Avg. Reward: 27.32509938923642
    Episode: 512	 Score: 21.519999518990517	 Avg. Reward: 27.32509938923642
    Episode: 513	 Score: 22.669999493286014	 Avg. Reward: 27.389699387792497
    Episode: 513	 Score: 22.669999493286014	 Avg. Reward: 27.389699387792497
    Episode: 514	 Score: 18.70999958179891	 Avg. Reward: 27.335999388992786
    Episode: 514	 Score: 18.70999958179891	 Avg. Reward: 27.335999388992786
    Episode: 515	 Score: 18.019999597221613	 Avg. Reward: 27.272499390412122
    Episode: 515	 Score: 18.019999597221613	 Avg. Reward: 27.272499390412122
    Episode: 516	 Score: 24.889999443665147	 Avg. Reward: 27.254499390814452
    Episode: 516	 Score: 24.889999443665147	 Avg. Reward: 27.254499390814452
    Episode: 517	 Score: 26.229999413713813	 Avg. Reward: 27.250999390892684
    Episode: 517	 Score: 26.229999413713813	 Avg. Reward: 27.250999390892684
    Episode: 518	 Score: 30.799999311566353	 Avg. Reward: 27.214999391697347
    Episode: 518	 Score: 30.799999311566353	 Avg. Reward: 27.214999391697347
    Episode: 519	 Score: 23.919999465346336	 Avg. Reward: 27.234899391252547
    Episode: 519	 Score: 23.919999465346336	 Avg. Reward: 27.234899391252547
    Episode: 520	 Score: 24.55999945104122	 Avg. Reward: 27.08579939458519
    Episode: 520	 Score: 24.55999945104122	 Avg. Reward: 27.08579939458519
    Episode: 521	 Score: 24.97999944165349	 Avg. Reward: 27.03619939569384
    Episode: 521	 Score: 24.97999944165349	 Avg. Reward: 27.03619939569384
    Episode: 522	 Score: 21.64999951608479	 Avg. Reward: 27.090599394477906
    Episode: 522	 Score: 21.64999951608479	 Avg. Reward: 27.090599394477906
    Episode: 523	 Score: 33.6999992467463	 Avg. Reward: 27.102299394216388
    Episode: 523	 Score: 33.6999992467463	 Avg. Reward: 27.102299394216388
    Episode: 524	 Score: 17.629999605938792	 Avg. Reward: 27.112399393990636
    Episode: 524	 Score: 17.629999605938792	 Avg. Reward: 27.112399393990636
    Episode: 525	 Score: 32.51999927312136	 Avg. Reward: 27.226499391440303
    Episode: 525	 Score: 32.51999927312136	 Avg. Reward: 27.226499391440303
    Episode: 526	 Score: 18.4299995880574	 Avg. Reward: 27.071799394898118
    Episode: 526	 Score: 18.4299995880574	 Avg. Reward: 27.071799394898118
    Episode: 527	 Score: 28.479999363422394	 Avg. Reward: 27.10129939423874
    Episode: 527	 Score: 28.479999363422394	 Avg. Reward: 27.10129939423874
    Episode: 528	 Score: 21.21999952569604	 Avg. Reward: 27.112799393981696
    Episode: 528	 Score: 21.21999952569604	 Avg. Reward: 27.112799393981696
    Episode: 529	 Score: 21.479999519884586	 Avg. Reward: 27.02419939596206
    Episode: 529	 Score: 21.479999519884586	 Avg. Reward: 27.02419939596206
    Episode: 530	 Score: 27.87999937683344	 Avg. Reward: 27.00889939630404
    Episode: 530	 Score: 27.87999937683344	 Avg. Reward: 27.00889939630404
    Episode: 531	 Score: 17.069999618455768	 Avg. Reward: 26.897099398802965
    Episode: 531	 Score: 17.069999618455768	 Avg. Reward: 26.897099398802965
    Episode: 532	 Score: 22.33999950066209	 Avg. Reward: 26.850699399840085
    Episode: 532	 Score: 22.33999950066209	 Avg. Reward: 26.850699399840085
    Episode: 533	 Score: 19.999999552965164	 Avg. Reward: 26.73549940241501
    Episode: 533	 Score: 19.999999552965164	 Avg. Reward: 26.73549940241501
    Episode: 534	 Score: 33.31999925523996	 Avg. Reward: 26.806499400828034
    Episode: 534	 Score: 33.31999925523996	 Avg. Reward: 26.806499400828034
    Episode: 535	 Score: 0.4499999899417162	 Avg. Reward: 26.524699407126754
    Episode: 535	 Score: 0.4499999899417162	 Avg. Reward: 26.524699407126754
    Episode: 536	 Score: 29.54999933950603	 Avg. Reward: 26.534699406903236
    Episode: 536	 Score: 29.54999933950603	 Avg. Reward: 26.534699406903236
    Episode: 537	 Score: 19.619999561458826	 Avg. Reward: 26.457899408619852
    Episode: 537	 Score: 19.619999561458826	 Avg. Reward: 26.457899408619852
    Episode: 538	 Score: 28.709999358281493	 Avg. Reward: 26.41689940953627
    Episode: 538	 Score: 28.709999358281493	 Avg. Reward: 26.41689940953627
    Episode: 539	 Score: 19.109999572858214	 Avg. Reward: 26.339499411266296
    Episode: 539	 Score: 19.109999572858214	 Avg. Reward: 26.339499411266296
    Episode: 540	 Score: 23.31999947875738	 Avg. Reward: 26.29559941224754
    Episode: 540	 Score: 23.31999947875738	 Avg. Reward: 26.29559941224754
    Episode: 541	 Score: 31.299999300390482	 Avg. Reward: 26.248499413300305
    Episode: 541	 Score: 31.299999300390482	 Avg. Reward: 26.248499413300305
    Episode: 542	 Score: 21.18999952636659	 Avg. Reward: 26.209699414167552
    Episode: 542	 Score: 21.18999952636659	 Avg. Reward: 26.209699414167552
    Episode: 543	 Score: 24.409999454393983	 Avg. Reward: 26.154399415403603
    Episode: 543	 Score: 24.409999454393983	 Avg. Reward: 26.154399415403603
    Episode: 544	 Score: 27.009999396279454	 Avg. Reward: 26.139199415743352
    Episode: 544	 Score: 27.009999396279454	 Avg. Reward: 26.139199415743352
    Episode: 545	 Score: 30.339999321848154	 Avg. Reward: 26.203899414297194
    Episode: 545	 Score: 30.339999321848154	 Avg. Reward: 26.203899414297194
    Episode: 546	 Score: 31.319999299943447	 Avg. Reward: 26.32979941148311
    Episode: 546	 Score: 31.319999299943447	 Avg. Reward: 26.32979941148311
    Episode: 547	 Score: 20.8399995341897	 Avg. Reward: 26.250199413262308
    Episode: 547	 Score: 20.8399995341897	 Avg. Reward: 26.250199413262308
    Episode: 548	 Score: 27.369999388232827	 Avg. Reward: 26.23509941359982
    Episode: 548	 Score: 27.369999388232827	 Avg. Reward: 26.23509941359982
    Episode: 549	 Score: 27.169999392703176	 Avg. Reward: 26.174399414956568
    Episode: 549	 Score: 27.169999392703176	 Avg. Reward: 26.174399414956568
    Episode: 550	 Score: 21.089999528601766	 Avg. Reward: 26.043799417875707
    Episode: 550	 Score: 21.089999528601766	 Avg. Reward: 26.043799417875707
    Episode: 551	 Score: 29.759999334812164	 Avg. Reward: 26.092199416793882
    Episode: 551	 Score: 29.759999334812164	 Avg. Reward: 26.092199416793882
    Episode: 552	 Score: 33.37999925389886	 Avg. Reward: 26.089799416847526
    Episode: 552	 Score: 33.37999925389886	 Avg. Reward: 26.089799416847526
    Episode: 553	 Score: 32.63999927043915	 Avg. Reward: 26.161599415242673
    Episode: 553	 Score: 32.63999927043915	 Avg. Reward: 26.161599415242673
    Episode: 554	 Score: 34.49999922886491	 Avg. Reward: 26.266299412902445
    Episode: 554	 Score: 34.49999922886491	 Avg. Reward: 26.266299412902445
    Episode: 555	 Score: 21.679999515414238	 Avg. Reward: 26.16489941516891
    Episode: 555	 Score: 21.679999515414238	 Avg. Reward: 26.16489941516891
    Episode: 556	 Score: 24.659999448806047	 Avg. Reward: 26.151999415457247
    Episode: 556	 Score: 24.659999448806047	 Avg. Reward: 26.151999415457247
    Episode: 557	 Score: 25.279999434947968	 Avg. Reward: 26.172599414996803
    Episode: 557	 Score: 25.279999434947968	 Avg. Reward: 26.172599414996803
    Episode: 558	 Score: 30.819999311119318	 Avg. Reward: 26.236899413559584
    Episode: 558	 Score: 30.819999311119318	 Avg. Reward: 26.236899413559584
    Episode: 559	 Score: 18.259999591857195	 Avg. Reward: 26.181999414786695
    Episode: 559	 Score: 18.259999591857195	 Avg. Reward: 26.181999414786695
    Episode: 560	 Score: 18.269999591633677	 Avg. Reward: 26.063399417437612
    Episode: 560	 Score: 18.269999591633677	 Avg. Reward: 26.063399417437612
    Episode: 561	 Score: 24.769999446347356	 Avg. Reward: 26.075199417173863
    Episode: 561	 Score: 24.769999446347356	 Avg. Reward: 26.075199417173863
    Episode: 562	 Score: 33.65999924764037	 Avg. Reward: 26.151099415477365
    Episode: 562	 Score: 33.65999924764037	 Avg. Reward: 26.151099415477365
    Episode: 563	 Score: 21.199999526143074	 Avg. Reward: 26.007499418687075
    Episode: 563	 Score: 21.199999526143074	 Avg. Reward: 26.007499418687075
    Episode: 564	 Score: 30.189999325200915	 Avg. Reward: 26.04199941791594
    Episode: 564	 Score: 30.189999325200915	 Avg. Reward: 26.04199941791594
    Episode: 565	 Score: 25.739999424666166	 Avg. Reward: 26.015199418514968
    Episode: 565	 Score: 25.739999424666166	 Avg. Reward: 26.015199418514968
    Episode: 566	 Score: 27.41999938711524	 Avg. Reward: 26.048199417777358
    Episode: 566	 Score: 27.41999938711524	 Avg. Reward: 26.048199417777358
    Episode: 567	 Score: 28.169999370351434	 Avg. Reward: 26.038199418000875
    Episode: 567	 Score: 28.169999370351434	 Avg. Reward: 26.038199418000875
    Episode: 568	 Score: 17.949999598786235	 Avg. Reward: 25.91439942076802
    Episode: 568	 Score: 17.949999598786235	 Avg. Reward: 25.91439942076802
    Episode: 569	 Score: 28.019999373704195	 Avg. Reward: 25.904099420998246
    Episode: 569	 Score: 28.019999373704195	 Avg. Reward: 25.904099420998246
    Episode: 570	 Score: 27.30999938957393	 Avg. Reward: 25.83659942250699
    Episode: 570	 Score: 27.30999938957393	 Avg. Reward: 25.83659942250699
    Episode: 571	 Score: 27.439999386668205	 Avg. Reward: 25.760699424203484
    Episode: 571	 Score: 27.439999386668205	 Avg. Reward: 25.760699424203484
    Episode: 572	 Score: 23.829999467357993	 Avg. Reward: 25.635299427006395
    Episode: 572	 Score: 23.829999467357993	 Avg. Reward: 25.635299427006395
    Episode: 573	 Score: 25.17999943718314	 Avg. Reward: 25.653199426606296
    Episode: 573	 Score: 25.17999943718314	 Avg. Reward: 25.653199426606296
    Episode: 574	 Score: 26.309999411925673	 Avg. Reward: 25.57579942833632
    Episode: 574	 Score: 26.309999411925673	 Avg. Reward: 25.57579942833632
    Episode: 575	 Score: 30.269999323412776	 Avg. Reward: 25.583199428170918
    Episode: 575	 Score: 30.269999323412776	 Avg. Reward: 25.583199428170918
    Episode: 576	 Score: 23.35999947786331	 Avg. Reward: 25.51169942976907
    Episode: 576	 Score: 23.35999947786331	 Avg. Reward: 25.51169942976907
    Episode: 577	 Score: 15.17999966070056	 Avg. Reward: 25.317599434107542
    Episode: 577	 Score: 15.17999966070056	 Avg. Reward: 25.317599434107542
    Episode: 578	 Score: 25.97999941930175	 Avg. Reward: 25.353699433300644
    Episode: 578	 Score: 25.97999941930175	 Avg. Reward: 25.353699433300644
    Episode: 579	 Score: 18.39999958872795	 Avg. Reward: 25.271799435131253
    Episode: 579	 Score: 18.39999958872795	 Avg. Reward: 25.271799435131253
    Episode: 580	 Score: 29.679999336600304	 Avg. Reward: 25.3523994333297
    Episode: 580	 Score: 29.679999336600304	 Avg. Reward: 25.3523994333297
    Episode: 581	 Score: 20.21999954804778	 Avg. Reward: 25.262399435341358
    Episode: 581	 Score: 20.21999954804778	 Avg. Reward: 25.262399435341358
    Episode: 582	 Score: 19.199999570846558	 Avg. Reward: 25.176499437261374
    Episode: 582	 Score: 19.199999570846558	 Avg. Reward: 25.176499437261374
    Episode: 583	 Score: 25.229999436065555	 Avg. Reward: 25.185099437069148
    Episode: 583	 Score: 25.229999436065555	 Avg. Reward: 25.185099437069148
    Episode: 584	 Score: 17.689999604597688	 Avg. Reward: 25.08859943922609
    Episode: 584	 Score: 17.689999604597688	 Avg. Reward: 25.08859943922609
    Episode: 585	 Score: 14.94999966584146	 Avg. Reward: 24.96399944201112
    Episode: 585	 Score: 14.94999966584146	 Avg. Reward: 24.96399944201112
    Episode: 586	 Score: 26.699999403208494	 Avg. Reward: 24.965199441984296
    Episode: 586	 Score: 26.699999403208494	 Avg. Reward: 24.965199441984296
    Episode: 587	 Score: 19.35999956727028	 Avg. Reward: 24.912599443159998
    Episode: 587	 Score: 19.35999956727028	 Avg. Reward: 24.912599443159998
    Episode: 588	 Score: 29.74999933503568	 Avg. Reward: 24.88629944374785
    Episode: 588	 Score: 29.74999933503568	 Avg. Reward: 24.88629944374785
    Episode: 589	 Score: 28.049999373033643	 Avg. Reward: 24.772499446291476
    Episode: 589	 Score: 28.049999373033643	 Avg. Reward: 24.772499446291476
    Episode: 590	 Score: 18.46999958716333	 Avg. Reward: 24.62309944963083
    Episode: 590	 Score: 18.46999958716333	 Avg. Reward: 24.62309944963083
    Episode: 591	 Score: 33.41999925300479	 Avg. Reward: 24.593299450296907
    Episode: 591	 Score: 33.41999925300479	 Avg. Reward: 24.593299450296907
    Episode: 592	 Score: 19.43999956548214	 Avg. Reward: 24.477399452887475
    Episode: 592	 Score: 19.43999956548214	 Avg. Reward: 24.477399452887475
    Episode: 593	 Score: 28.409999364987016	 Avg. Reward: 24.460699453260748
    Episode: 593	 Score: 28.409999364987016	 Avg. Reward: 24.460699453260748
    Episode: 594	 Score: 28.609999360516667	 Avg. Reward: 24.523799451850355
    Episode: 594	 Score: 28.609999360516667	 Avg. Reward: 24.523799451850355
    Episode: 595	 Score: 25.379999432712793	 Avg. Reward: 24.551999451220034
    Episode: 595	 Score: 25.379999432712793	 Avg. Reward: 24.551999451220034
    Episode: 596	 Score: 25.55999942868948	 Avg. Reward: 24.537899451535196
    Episode: 596	 Score: 25.55999942868948	 Avg. Reward: 24.537899451535196
    Episode: 597	 Score: 20.619999539107084	 Avg. Reward: 24.491099452581256
    Episode: 597	 Score: 20.619999539107084	 Avg. Reward: 24.491099452581256
    Episode: 598	 Score: 24.839999444782734	 Avg. Reward: 24.415899454262107
    Episode: 598	 Score: 24.839999444782734	 Avg. Reward: 24.415899454262107
    Episode: 599	 Score: 19.099999573081732	 Avg. Reward: 24.418899454195053
    Episode: 599	 Score: 19.099999573081732	 Avg. Reward: 24.418899454195053
    Episode: 600	 Score: 30.93999930843711	 Avg. Reward: 24.404499454516916
    Episode: 600	 Score: 30.93999930843711	 Avg. Reward: 24.404499454516916
    Episode: 601	 Score: 38.55999913811684	 Avg. Reward: 24.582199450545012
    Episode: 601	 Score: 38.55999913811684	 Avg. Reward: 24.582199450545012
    Episode: 602	 Score: 33.85999924317002	 Avg. Reward: 24.708299447726457
    Episode: 602	 Score: 33.85999924317002	 Avg. Reward: 24.708299447726457
    Episode: 603	 Score: 33.059999261051416	 Avg. Reward: 24.81319944538176
    Episode: 603	 Score: 33.059999261051416	 Avg. Reward: 24.81319944538176
    Episode: 604	 Score: 29.619999337941408	 Avg. Reward: 24.82999944500625
    Episode: 604	 Score: 29.619999337941408	 Avg. Reward: 24.82999944500625
    Episode: 605	 Score: 32.28999927826226	 Avg. Reward: 24.93089944275096
    Episode: 605	 Score: 32.28999927826226	 Avg. Reward: 24.93089944275096
    Episode: 606	 Score: 31.979999285191298	 Avg. Reward: 24.877299443949013
    Episode: 606	 Score: 31.979999285191298	 Avg. Reward: 24.877299443949013
    Episode: 607	 Score: 30.099999327212572	 Avg. Reward: 25.010799440965055
    Episode: 607	 Score: 30.099999327212572	 Avg. Reward: 25.010799440965055
    Episode: 608	 Score: 30.369999321177602	 Avg. Reward: 25.083699439335614
    Episode: 608	 Score: 30.369999321177602	 Avg. Reward: 25.083699439335614
    Episode: 609	 Score: 33.439999252557755	 Avg. Reward: 25.404699432160704
    Episode: 609	 Score: 33.439999252557755	 Avg. Reward: 25.404699432160704
    Episode: 610	 Score: 32.71999926865101	 Avg. Reward: 25.51519942969084
    Episode: 610	 Score: 32.71999926865101	 Avg. Reward: 25.51519942969084
    Episode: 611	 Score: 39.409999119117856	 Avg. Reward: 25.55479942880571
    Episode: 611	 Score: 39.409999119117856	 Avg. Reward: 25.55479942880571
    Episode: 612	 Score: 26.91999939829111	 Avg. Reward: 25.608799427598715
    Episode: 612	 Score: 26.91999939829111	 Avg. Reward: 25.608799427598715
    Episode: 613	 Score: 25.889999421313405	 Avg. Reward: 25.640999426878988
    Episode: 613	 Score: 25.889999421313405	 Avg. Reward: 25.640999426878988
    Episode: 614	 Score: 24.97999944165349	 Avg. Reward: 25.703699425477534
    Episode: 614	 Score: 24.97999944165349	 Avg. Reward: 25.703699425477534
    Episode: 615	 Score: 29.229999346658587	 Avg. Reward: 25.815799422971903
    Episode: 615	 Score: 29.229999346658587	 Avg. Reward: 25.815799422971903
    Episode: 616	 Score: 24.719999447464943	 Avg. Reward: 25.814099423009903
    Episode: 616	 Score: 24.719999447464943	 Avg. Reward: 25.814099423009903
    Episode: 617	 Score: 20.459999542683363	 Avg. Reward: 25.756399424299598
    Episode: 617	 Score: 20.459999542683363	 Avg. Reward: 25.756399424299598
    Episode: 618	 Score: 18.54999958537519	 Avg. Reward: 25.633899427037687
    Episode: 618	 Score: 18.54999958537519	 Avg. Reward: 25.633899427037687
    Episode: 619	 Score: 21.789999512955546	 Avg. Reward: 25.61259942751378
    Episode: 619	 Score: 21.789999512955546	 Avg. Reward: 25.61259942751378
    Episode: 620	 Score: 20.269999546930194	 Avg. Reward: 25.569699428472667
    Episode: 620	 Score: 20.269999546930194	 Avg. Reward: 25.569699428472667
    Episode: 621	 Score: 25.039999440312386	 Avg. Reward: 25.570299428459258
    Episode: 621	 Score: 25.039999440312386	 Avg. Reward: 25.570299428459258
    Episode: 622	 Score: 28.979999352246523	 Avg. Reward: 25.643599426820874
    Episode: 622	 Score: 28.979999352246523	 Avg. Reward: 25.643599426820874
    Episode: 623	 Score: 33.56999924965203	 Avg. Reward: 25.64229942684993
    Episode: 623	 Score: 33.56999924965203	 Avg. Reward: 25.64229942684993
    Episode: 624	 Score: 31.979999285191298	 Avg. Reward: 25.785799423642455
    Episode: 624	 Score: 31.979999285191298	 Avg. Reward: 25.785799423642455
    Episode: 625	 Score: 28.73999935761094	 Avg. Reward: 25.747999424487354
    Episode: 625	 Score: 28.73999935761094	 Avg. Reward: 25.747999424487354
    Episode: 626	 Score: 26.439999409019947	 Avg. Reward: 25.82809942269698
    Episode: 626	 Score: 26.439999409019947	 Avg. Reward: 25.82809942269698
    Episode: 627	 Score: 29.519999340176582	 Avg. Reward: 25.83849942246452
    Episode: 627	 Score: 29.519999340176582	 Avg. Reward: 25.83849942246452
    Episode: 628	 Score: 33.81999924406409	 Avg. Reward: 25.9644994196482
    Episode: 628	 Score: 33.81999924406409	 Avg. Reward: 25.9644994196482
    Episode: 629	 Score: 21.249999525025487	 Avg. Reward: 25.96219941969961
    Episode: 629	 Score: 21.249999525025487	 Avg. Reward: 25.96219941969961
    Episode: 630	 Score: 31.56999929435551	 Avg. Reward: 25.99909941887483
    Episode: 630	 Score: 31.56999929435551	 Avg. Reward: 25.99909941887483
    Episode: 631	 Score: 20.94999953173101	 Avg. Reward: 26.037899418007584
    Episode: 631	 Score: 20.94999953173101	 Avg. Reward: 26.037899418007584
    Episode: 632	 Score: 30.489999318495393	 Avg. Reward: 26.119399416185914
    Episode: 632	 Score: 30.489999318495393	 Avg. Reward: 26.119399416185914
    Episode: 633	 Score: 30.20999932475388	 Avg. Reward: 26.221499413903803
    Episode: 633	 Score: 30.20999932475388	 Avg. Reward: 26.221499413903803
    Episode: 634	 Score: 34.07999923825264	 Avg. Reward: 26.22909941373393
    Episode: 634	 Score: 34.07999923825264	 Avg. Reward: 26.22909941373393
    Episode: 635	 Score: 12.329999724403024	 Avg. Reward: 26.347899411078544
    Episode: 635	 Score: 12.329999724403024	 Avg. Reward: 26.347899411078544
    Episode: 636	 Score: 20.579999540001154	 Avg. Reward: 26.258199413083492
    Episode: 636	 Score: 20.579999540001154	 Avg. Reward: 26.258199413083492
    Episode: 637	 Score: 28.299999367445707	 Avg. Reward: 26.344999411143363
    Episode: 637	 Score: 28.299999367445707	 Avg. Reward: 26.344999411143363
    Episode: 638	 Score: 24.74999944679439	 Avg. Reward: 26.305399412028493
    Episode: 638	 Score: 24.74999944679439	 Avg. Reward: 26.305399412028493
    Episode: 639	 Score: 26.659999404102564	 Avg. Reward: 26.380899410340934
    Episode: 639	 Score: 26.659999404102564	 Avg. Reward: 26.380899410340934
    Episode: 640	 Score: 29.289999345317483	 Avg. Reward: 26.440599409006538
    Episode: 640	 Score: 29.289999345317483	 Avg. Reward: 26.440599409006538
    Episode: 641	 Score: 25.63999942690134	 Avg. Reward: 26.383999410271645
    Episode: 641	 Score: 25.63999942690134	 Avg. Reward: 26.383999410271645
    Episode: 642	 Score: 22.559999495744705	 Avg. Reward: 26.397699409965426
    Episode: 642	 Score: 22.559999495744705	 Avg. Reward: 26.397699409965426
    Episode: 643	 Score: 23.059999484568834	 Avg. Reward: 26.384199410267176
    Episode: 643	 Score: 23.059999484568834	 Avg. Reward: 26.384199410267176
    Episode: 644	 Score: 23.85999946668744	 Avg. Reward: 26.352699410971255
    Episode: 644	 Score: 23.85999946668744	 Avg. Reward: 26.352699410971255
    Episode: 645	 Score: 29.789999334141612	 Avg. Reward: 26.347199411094188
    Episode: 645	 Score: 29.789999334141612	 Avg. Reward: 26.347199411094188
    Episode: 646	 Score: 27.399999387562275	 Avg. Reward: 26.30799941197038
    Episode: 646	 Score: 27.399999387562275	 Avg. Reward: 26.30799941197038
    Episode: 647	 Score: 33.369999254122376	 Avg. Reward: 26.433299409169702
    Episode: 647	 Score: 33.369999254122376	 Avg. Reward: 26.433299409169702
    Episode: 648	 Score: 24.959999442100525	 Avg. Reward: 26.409199409708382
    Episode: 648	 Score: 24.959999442100525	 Avg. Reward: 26.409199409708382
    Episode: 649	 Score: 26.6399994045496	 Avg. Reward: 26.403899409826845
    Episode: 649	 Score: 26.6399994045496	 Avg. Reward: 26.403899409826845
    Episode: 650	 Score: 26.1799994148314	 Avg. Reward: 26.45479940868914
    Episode: 650	 Score: 26.1799994148314	 Avg. Reward: 26.45479940868914
    Episode: 651	 Score: 31.579999294131994	 Avg. Reward: 26.47299940828234
    Episode: 651	 Score: 31.579999294131994	 Avg. Reward: 26.47299940828234
    Episode: 652	 Score: 5.379999879747629	 Avg. Reward: 26.19299941454083
    Episode: 652	 Score: 5.379999879747629	 Avg. Reward: 26.19299941454083
    Episode: 653	 Score: 21.349999522790313	 Avg. Reward: 26.08009941706434
    Episode: 653	 Score: 21.349999522790313	 Avg. Reward: 26.08009941706434
    Episode: 654	 Score: 35.4799992069602	 Avg. Reward: 26.08989941684529
    Episode: 654	 Score: 35.4799992069602	 Avg. Reward: 26.08989941684529
    Episode: 655	 Score: 22.019999507814646	 Avg. Reward: 26.093299416769295
    Episode: 655	 Score: 22.019999507814646	 Avg. Reward: 26.093299416769295
    Episode: 656	 Score: 32.48999927379191	 Avg. Reward: 26.171599415019156
    Episode: 656	 Score: 32.48999927379191	 Avg. Reward: 26.171599415019156
    Episode: 657	 Score: 22.019999507814646	 Avg. Reward: 26.138999415747822
    Episode: 657	 Score: 22.019999507814646	 Avg. Reward: 26.138999415747822
    Episode: 658	 Score: 21.45999952033162	 Avg. Reward: 26.045399417839945
    Episode: 658	 Score: 21.45999952033162	 Avg. Reward: 26.045399417839945
    Episode: 659	 Score: 23.959999464452267	 Avg. Reward: 26.102399416565895
    Episode: 659	 Score: 23.959999464452267	 Avg. Reward: 26.102399416565895
    Episode: 660	 Score: 19.759999558329582	 Avg. Reward: 26.117299416232854
    Episode: 660	 Score: 19.759999558329582	 Avg. Reward: 26.117299416232854
    Episode: 661	 Score: 30.579999316483736	 Avg. Reward: 26.17539941493422
    Episode: 661	 Score: 30.579999316483736	 Avg. Reward: 26.17539941493422
    Episode: 662	 Score: 24.389999454841018	 Avg. Reward: 26.082699417006225
    Episode: 662	 Score: 24.389999454841018	 Avg. Reward: 26.082699417006225
    Episode: 663	 Score: 23.88999946601689	 Avg. Reward: 26.10959941640496
    Episode: 663	 Score: 23.88999946601689	 Avg. Reward: 26.10959941640496
    Episode: 664	 Score: 24.419999454170465	 Avg. Reward: 26.051899417694656
    Episode: 664	 Score: 24.419999454170465	 Avg. Reward: 26.051899417694656
    Episode: 665	 Score: 27.7299993801862	 Avg. Reward: 26.07179941724986
    Episode: 665	 Score: 27.7299993801862	 Avg. Reward: 26.07179941724986
    Episode: 666	 Score: 25.05999943986535	 Avg. Reward: 26.048199417777358
    Episode: 666	 Score: 25.05999943986535	 Avg. Reward: 26.048199417777358
    Episode: 667	 Score: 31.389999298378825	 Avg. Reward: 26.080399417057635
    Episode: 667	 Score: 31.389999298378825	 Avg. Reward: 26.080399417057635
    Episode: 668	 Score: 39.38999911956489	 Avg. Reward: 26.29479941226542
    Episode: 668	 Score: 39.38999911956489	 Avg. Reward: 26.29479941226542
    Episode: 669	 Score: 32.55999927222729	 Avg. Reward: 26.34019941125065
    Episode: 669	 Score: 32.55999927222729	 Avg. Reward: 26.34019941125065
    Episode: 670	 Score: 34.53999922797084	 Avg. Reward: 26.41249940963462
    Episode: 670	 Score: 34.53999922797084	 Avg. Reward: 26.41249940963462
    Episode: 671	 Score: 27.37999938800931	 Avg. Reward: 26.41189940964803
    Episode: 671	 Score: 27.37999938800931	 Avg. Reward: 26.41189940964803
    Episode: 672	 Score: 27.249999390915036	 Avg. Reward: 26.4460994088836
    Episode: 672	 Score: 27.249999390915036	 Avg. Reward: 26.4460994088836
    Episode: 673	 Score: 39.60999911464751	 Avg. Reward: 26.590399405658246
    Episode: 673	 Score: 39.60999911464751	 Avg. Reward: 26.590399405658246
    Episode: 674	 Score: 34.81999922171235	 Avg. Reward: 26.675499403756113
    Episode: 674	 Score: 34.81999922171235	 Avg. Reward: 26.675499403756113
    Episode: 675	 Score: 34.80999922193587	 Avg. Reward: 26.720899402741342
    Episode: 675	 Score: 34.80999922193587	 Avg. Reward: 26.720899402741342
    Episode: 676	 Score: 32.71999926865101	 Avg. Reward: 26.81449940064922
    Episode: 676	 Score: 32.71999926865101	 Avg. Reward: 26.81449940064922
    Episode: 677	 Score: 33.60999924875796	 Avg. Reward: 26.998799396529794
    Episode: 677	 Score: 33.60999924875796	 Avg. Reward: 26.998799396529794
    Episode: 678	 Score: 36.31999918818474	 Avg. Reward: 27.102199394218623
    Episode: 678	 Score: 36.31999918818474	 Avg. Reward: 27.102199394218623
    Episode: 679	 Score: 31.999999284744263	 Avg. Reward: 27.238199391178785
    Episode: 679	 Score: 31.999999284744263	 Avg. Reward: 27.238199391178785
    Episode: 680	 Score: 34.07999923825264	 Avg. Reward: 27.28219939019531
    Episode: 680	 Score: 34.07999923825264	 Avg. Reward: 27.28219939019531
    Episode: 681	 Score: 29.849999332800508	 Avg. Reward: 27.378499388042837
    Episode: 681	 Score: 29.849999332800508	 Avg. Reward: 27.378499388042837
    Episode: 682	 Score: 31.749999290332198	 Avg. Reward: 27.503999385237694
    Episode: 682	 Score: 31.749999290332198	 Avg. Reward: 27.503999385237694
    Episode: 683	 Score: 33.91999924182892	 Avg. Reward: 27.590899383295326
    Episode: 683	 Score: 33.91999924182892	 Avg. Reward: 27.590899383295326
    Episode: 684	 Score: 20.149999549612403	 Avg. Reward: 27.615499382745476
    Episode: 684	 Score: 20.149999549612403	 Avg. Reward: 27.615499382745476
    Episode: 685	 Score: 19.819999556988478	 Avg. Reward: 27.664199381656946
    Episode: 685	 Score: 19.819999556988478	 Avg. Reward: 27.664199381656946
    Episode: 686	 Score: 20.199999548494816	 Avg. Reward: 27.59919938310981
    Episode: 686	 Score: 20.199999548494816	 Avg. Reward: 27.59919938310981
    Episode: 687	 Score: 30.509999318048358	 Avg. Reward: 27.710699380617587
    Episode: 687	 Score: 30.509999318048358	 Avg. Reward: 27.710699380617587
    Episode: 688	 Score: 24.04999946244061	 Avg. Reward: 27.653699381891638
    Episode: 688	 Score: 24.04999946244061	 Avg. Reward: 27.653699381891638
    Episode: 689	 Score: 35.57999920472503	 Avg. Reward: 27.728999380208553
    Episode: 689	 Score: 35.57999920472503	 Avg. Reward: 27.728999380208553
    Episode: 690	 Score: 35.38999920897186	 Avg. Reward: 27.898199376426636
    Episode: 690	 Score: 35.38999920897186	 Avg. Reward: 27.898199376426636
    Episode: 691	 Score: 31.63999929279089	 Avg. Reward: 27.8803993768245
    Episode: 691	 Score: 31.63999929279089	 Avg. Reward: 27.8803993768245
    Episode: 692	 Score: 33.629999248310924	 Avg. Reward: 28.022299373652785
    Episode: 692	 Score: 33.629999248310924	 Avg. Reward: 28.022299373652785
    Episode: 693	 Score: 30.32999932207167	 Avg. Reward: 28.041499373223633
    Episode: 693	 Score: 30.32999932207167	 Avg. Reward: 28.041499373223633
    Episode: 694	 Score: 32.009999284520745	 Avg. Reward: 28.075499372463675
    Episode: 694	 Score: 32.009999284520745	 Avg. Reward: 28.075499372463675
    Episode: 695	 Score: 35.26999921165407	 Avg. Reward: 28.174399370253084
    Episode: 695	 Score: 35.26999921165407	 Avg. Reward: 28.174399370253084
    Episode: 696	 Score: 27.88999937660992	 Avg. Reward: 28.19769936973229
    Episode: 696	 Score: 27.88999937660992	 Avg. Reward: 28.19769936973229
    Episode: 697	 Score: 29.929999331012368	 Avg. Reward: 28.290799367651342
    Episode: 697	 Score: 29.929999331012368	 Avg. Reward: 28.290799367651342
    Episode: 698	 Score: 28.559999361634254	 Avg. Reward: 28.32799936681986
    Episode: 698	 Score: 28.559999361634254	 Avg. Reward: 28.32799936681986
    Episode: 699	 Score: 33.249999256804585	 Avg. Reward: 28.469499363657086
    Episode: 699	 Score: 33.249999256804585	 Avg. Reward: 28.469499363657086
    Episode: 700	 Score: 28.1899993699044	 Avg. Reward: 28.44199936427176
    Episode: 700	 Score: 28.1899993699044	 Avg. Reward: 28.44199936427176
    Episode: 701	 Score: 32.82999926619232	 Avg. Reward: 28.384699365552514
    Episode: 701	 Score: 32.82999926619232	 Avg. Reward: 28.384699365552514
    Episode: 702	 Score: 23.0799994841218	 Avg. Reward: 28.27689936796203
    Episode: 702	 Score: 23.0799994841218	 Avg. Reward: 28.27689936796203
    Episode: 703	 Score: 27.519999384880066	 Avg. Reward: 28.22149936920032
    Episode: 703	 Score: 27.519999384880066	 Avg. Reward: 28.22149936920032
    Episode: 704	 Score: 25.17999943718314	 Avg. Reward: 28.177099370192735
    Episode: 704	 Score: 25.17999943718314	 Avg. Reward: 28.177099370192735
    Episode: 705	 Score: 33.169999258592725	 Avg. Reward: 28.18589936999604
    Episode: 705	 Score: 33.169999258592725	 Avg. Reward: 28.18589936999604
    Episode: 706	 Score: 31.029999306425452	 Avg. Reward: 28.176399370208383
    Episode: 706	 Score: 31.029999306425452	 Avg. Reward: 28.176399370208383
    Episode: 707	 Score: 28.409999364987016	 Avg. Reward: 28.159499370586126
    Episode: 707	 Score: 28.409999364987016	 Avg. Reward: 28.159499370586126
    Episode: 708	 Score: 35.179999213665724	 Avg. Reward: 28.20759936951101
    Episode: 708	 Score: 35.179999213665724	 Avg. Reward: 28.20759936951101
    Episode: 709	 Score: 32.86999926529825	 Avg. Reward: 28.20189936963841
    Episode: 709	 Score: 32.86999926529825	 Avg. Reward: 28.20189936963841
    Episode: 710	 Score: 27.87999937683344	 Avg. Reward: 28.153499370720237
    Episode: 710	 Score: 27.87999937683344	 Avg. Reward: 28.153499370720237
    Episode: 711	 Score: 27.779999379068613	 Avg. Reward: 28.037199373319744
    Episode: 711	 Score: 27.779999379068613	 Avg. Reward: 28.037199373319744
    Episode: 712	 Score: 38.24999914504588	 Avg. Reward: 28.150499370787294
    Episode: 712	 Score: 38.24999914504588	 Avg. Reward: 28.150499370787294
    Episode: 713	 Score: 31.629999293014407	 Avg. Reward: 28.207899369504304
    Episode: 713	 Score: 31.629999293014407	 Avg. Reward: 28.207899369504304
    Episode: 714	 Score: 33.829999243840575	 Avg. Reward: 28.296399367526174
    Episode: 714	 Score: 33.829999243840575	 Avg. Reward: 28.296399367526174
    Episode: 715	 Score: 30.799999311566353	 Avg. Reward: 28.31209936717525
    Episode: 715	 Score: 30.799999311566353	 Avg. Reward: 28.31209936717525
    Episode: 716	 Score: 27.25999939069152	 Avg. Reward: 28.337499366607517
    Episode: 716	 Score: 27.25999939069152	 Avg. Reward: 28.337499366607517
    Episode: 717	 Score: 31.239999301731586	 Avg. Reward: 28.445299364198
    Episode: 717	 Score: 31.239999301731586	 Avg. Reward: 28.445299364198
    Episode: 718	 Score: 33.4699992518872	 Avg. Reward: 28.594499360863118
    Episode: 718	 Score: 33.4699992518872	 Avg. Reward: 28.594499360863118
    Episode: 719	 Score: 27.819999378174543	 Avg. Reward: 28.65479935951531
    Episode: 719	 Score: 27.819999378174543	 Avg. Reward: 28.65479935951531
    Episode: 720	 Score: 25.82999942265451	 Avg. Reward: 28.710399358272554
    Episode: 720	 Score: 25.82999942265451	 Avg. Reward: 28.710399358272554
    Episode: 721	 Score: 33.89999924227595	 Avg. Reward: 28.79899935629219
    Episode: 721	 Score: 33.89999924227595	 Avg. Reward: 28.79899935629219
    Episode: 722	 Score: 33.57999924942851	 Avg. Reward: 28.844999355264008
    Episode: 722	 Score: 33.57999924942851	 Avg. Reward: 28.844999355264008
    Episode: 723	 Score: 29.529999339953065	 Avg. Reward: 28.804599356167017
    Episode: 723	 Score: 29.529999339953065	 Avg. Reward: 28.804599356167017
    Episode: 724	 Score: 36.689999179914594	 Avg. Reward: 28.851699355114253
    Episode: 724	 Score: 36.689999179914594	 Avg. Reward: 28.851699355114253
    Episode: 725	 Score: 25.569999428465962	 Avg. Reward: 28.8199993558228
    Episode: 725	 Score: 25.569999428465962	 Avg. Reward: 28.8199993558228
    Episode: 726	 Score: 35.08999921567738	 Avg. Reward: 28.906499353889377
    Episode: 726	 Score: 35.08999921567738	 Avg. Reward: 28.906499353889377
    Episode: 727	 Score: 29.459999341517687	 Avg. Reward: 28.905899353902786
    Episode: 727	 Score: 29.459999341517687	 Avg. Reward: 28.905899353902786
    Episode: 728	 Score: 39.46999911777675	 Avg. Reward: 28.962399352639913
    Episode: 728	 Score: 39.46999911777675	 Avg. Reward: 28.962399352639913
    Episode: 729	 Score: 36.71999917924404	 Avg. Reward: 29.117099349182098
    Episode: 729	 Score: 36.71999917924404	 Avg. Reward: 29.117099349182098
    Episode: 730	 Score: 19.779999557882547	 Avg. Reward: 28.99919935181737
    Episode: 730	 Score: 19.779999557882547	 Avg. Reward: 28.99919935181737
    Episode: 731	 Score: 29.339999344199896	 Avg. Reward: 29.08309934994206
    Episode: 731	 Score: 29.339999344199896	 Avg. Reward: 29.08309934994206
    Episode: 732	 Score: 18.349999589845538	 Avg. Reward: 28.96169935265556
    Episode: 732	 Score: 18.349999589845538	 Avg. Reward: 28.96169935265556
    Episode: 733	 Score: 29.619999337941408	 Avg. Reward: 28.955799352787434
    Episode: 733	 Score: 29.619999337941408	 Avg. Reward: 28.955799352787434
    Episode: 734	 Score: 23.569999473169446	 Avg. Reward: 28.8506993551366
    Episode: 734	 Score: 23.569999473169446	 Avg. Reward: 28.8506993551366
    Episode: 735	 Score: 29.35999934375286	 Avg. Reward: 29.0209993513301
    Episode: 735	 Score: 29.35999934375286	 Avg. Reward: 29.0209993513301
    Episode: 736	 Score: 28.719999358057976	 Avg. Reward: 29.10239934951067
    Episode: 736	 Score: 28.719999358057976	 Avg. Reward: 29.10239934951067
    Episode: 737	 Score: 33.26999925635755	 Avg. Reward: 29.152099348399787
    Episode: 737	 Score: 33.26999925635755	 Avg. Reward: 29.152099348399787
    Episode: 738	 Score: 37.67999915778637	 Avg. Reward: 29.28139934550971
    Episode: 738	 Score: 37.67999915778637	 Avg. Reward: 29.28139934550971
    Episode: 739	 Score: 36.12999919243157	 Avg. Reward: 29.376099343392998
    Episode: 739	 Score: 36.12999919243157	 Avg. Reward: 29.376099343392998
    Episode: 740	 Score: 36.00999919511378	 Avg. Reward: 29.44329934189096
    Episode: 740	 Score: 36.00999919511378	 Avg. Reward: 29.44329934189096
    Episode: 741	 Score: 25.499999430030584	 Avg. Reward: 29.44189934192225
    Episode: 741	 Score: 25.499999430030584	 Avg. Reward: 29.44189934192225
    Episode: 742	 Score: 36.60999918170273	 Avg. Reward: 29.582399338781833
    Episode: 742	 Score: 36.60999918170273	 Avg. Reward: 29.582399338781833
    Episode: 743	 Score: 27.279999390244484	 Avg. Reward: 29.62459933783859
    Episode: 743	 Score: 27.279999390244484	 Avg. Reward: 29.62459933783859
    Episode: 744	 Score: 29.369999343529344	 Avg. Reward: 29.67969933660701
    Episode: 744	 Score: 29.369999343529344	 Avg. Reward: 29.67969933660701
    Episode: 745	 Score: 31.129999304190278	 Avg. Reward: 29.693099336307498
    Episode: 745	 Score: 31.129999304190278	 Avg. Reward: 29.693099336307498
    Episode: 746	 Score: 32.69999926909804	 Avg. Reward: 29.746099335122853
    Episode: 746	 Score: 32.69999926909804	 Avg. Reward: 29.746099335122853
    Episode: 747	 Score: 38.549999138340354	 Avg. Reward: 29.797899333965034
    Episode: 747	 Score: 38.549999138340354	 Avg. Reward: 29.797899333965034
    Episode: 748	 Score: 29.9699993301183	 Avg. Reward: 29.847999332845212
    Episode: 748	 Score: 29.9699993301183	 Avg. Reward: 29.847999332845212
    Episode: 749	 Score: 37.549999160692096	 Avg. Reward: 29.957099330406635
    Episode: 749	 Score: 37.549999160692096	 Avg. Reward: 29.957099330406635
    Episode: 750	 Score: 38.72999913431704	 Avg. Reward: 30.082599327601493
    	--> SOLVED! <--	
    Episode: 750	 Score: 38.72999913431704	 Avg. Reward: 30.082599327601493
    	--> SOLVED! <--	



    
![png](Continuous_Control_files/Continuous_Control_18_1.png)
    



```python



env.close()
```
