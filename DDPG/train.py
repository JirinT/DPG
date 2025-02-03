import gymnasium as gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from policy_nn import DeterministicPolicy
from replay_memory import ReplayMemory
from Q_net import Q_nn
from ornstein_uhlbeck_noise import OU_noise
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

def compute_q_values(batch):
    states = torch.stack([experience[0] for experience in batch])
    actions = torch.tensor([experience[1] for experience in batch]).unsqueeze(dim=1)
    rewards = torch.tensor([experience[2] for experience in batch]).unsqueeze(dim=1)
    next_states = torch.stack([torch.from_numpy(experience[3]) for experience in batch])
    terminated = torch.tensor([int(experience[4]) for experience in batch]).unsqueeze(dim=1)

    with torch.no_grad():
        next_q_values = target_Q(torch.cat((next_states, target_policy(next_states)), dim=1)) # the input needs to be one tensor
    
    q_target = rewards + (1-terminated)*GAMMA*next_q_values
    q_predicted = online_Q(torch.cat((states, actions), dim=1))

    return q_target, q_predicted

def update_online_q_net(target_q, predicted_q):
    loss = online_Q.loss_fcn(target_q, predicted_q)

    online_Q.optimizer.zero_grad()
    loss.backward()
    online_Q.optimizer.step()

def update_policy(batch):
    # the actions needs to be computed again because they need to be predicted by the current policy if I want to use it to update the current policy!!!
    # so the q_value needs to be recomputed as well!!!
    states = torch.stack([experience[0] for experience in batch])
    actions = online_policy(states) # deterministic actions used to update policy
    current_q_values = online_Q(torch.cat((states, actions), dim=1))

    for p in online_policy.parameters():
        p.grad = None
    current_q_values.sum().backward()
    with torch.no_grad():
        for p in online_policy.parameters():
             p.copy_(p - POLICY_LEARNING_RATE * p.grad)

def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau*source_param.data + (1-tau)*target_param.data)

NUM_EPISODES = 11
Q_LEARNING_RATE = .0005
POLICY_LEARNING_RATE = .00005
REPLAY_MEMORY_SIZE = 100000
BATCH_SIZE = 128
TAU = .01 # Update filter constant
OU_LAMBDA = 0.9
OU_SIGMA = 0.3
GAMMA = .99
VIDEO_PERIOD = 5
torch.autograd.set_detect_anomaly(True)

env = gym.make("MountainCarContinuous-v0")
evaluation_env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
evaluation_env = RecordVideo(evaluation_env, video_folder="cartpole-agent", name_prefix="ep",
                  episode_trigger=lambda x: x % VIDEO_PERIOD == 0)

online_policy = DeterministicPolicy(env.observation_space.shape[0])
target_policy = DeterministicPolicy(env.observation_space.shape[0])
target_policy.load_state_dict(online_policy.state_dict()) # initialise both networks with the same parameters

online_Q = Q_nn(env.observation_space.shape[0], env.action_space.shape[0], learning_rate=Q_LEARNING_RATE)
target_Q = Q_nn(env.observation_space.shape[0], env.action_space.shape[0], learning_rate=Q_LEARNING_RATE)
target_Q.load_state_dict(online_Q.state_dict()) # initialise both networks with the same parameters

replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
ou_noise = OU_noise(lam=OU_LAMBDA, sigma=OU_SIGMA) # Ornstein Uhlbeck noise

reward_list = []

for _ in tqdm(range(NUM_EPISODES)):
    state = env.reset()[0]

    done = False
    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = online_policy(state)
            action += ou_noise.sample()
            action = torch.clamp(action, -1, 1) # ensure action is in the correct range after adding noise

        next_state, reward, terminated, truncated, info = env.step(action.detach().numpy())
        replay_memory.push((state, action, reward, next_state, terminated))
        state = next_state

        done = terminated or truncated

        if len(replay_memory) >= BATCH_SIZE:
            batch = replay_memory.sample(BATCH_SIZE)
            target_q, predicted_q = compute_q_values(batch)
            update_online_q_net(target_q, predicted_q)
            update_policy(batch)
            soft_update(target_Q, online_Q, TAU)
            soft_update(target_policy, online_policy, TAU)

    done = False
    state = evaluation_env.reset()[0]
    rewards = 0
    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = target_policy(state)
        next_state, reward, terminated, truncated, _ = evaluation_env.step(action.detach().numpy())
        rewards += reward
        state = next_state
        done = terminated or truncated

    reward_list.append(rewards)

env.close()

plt.figure()
plt.plot(reward_list)
plt.show()