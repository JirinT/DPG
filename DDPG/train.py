import gymnasium as gym
import torch
import numpy as np
import statistics
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
    next_states = torch.stack([torch.as_tensor(experience[3], dtype=torch.float32) for experience in batch])
    terminated = torch.tensor([int(experience[4]) for experience in batch]).unsqueeze(dim=1)

    with torch.no_grad():
        next_q_values = target_Q(torch.cat((next_states, target_policy(next_states)), dim=1)) # the input needs to be one tensor
    
    q_target = rewards + (1-terminated)*GAMMA*next_q_values
    q_predicted = online_Q(torch.cat((states, actions), dim=1))

    return q_target, q_predicted

def update_online_q_net(target_q, predicted_q):
    loss = online_Q.loss_fcn(predicted_q, target_q)
    loss_dict["q_value_loss"].append(loss)

    online_Q.optimizer.zero_grad()
    loss.backward()
    online_Q.optimizer.step()

def update_policy(batch):
    # the actions needs to be computed again because they need to be predicted by the current policy if I want to use it to update the current policy!!!
    # so the q_value needs to be recomputed as well!!!
    states = torch.stack([experience[0] for experience in batch])
    actions = online_policy(states) # deterministic actions used to update policy
    current_q_values = online_Q(torch.cat((states, actions), dim=1))
    loss = -current_q_values.mean()
    loss_dict["policy_loss"].append(loss)

    online_policy.optimizer.zero_grad()
    loss.backward()
    online_policy.optimizer.step()

def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau*source_param.data + (1-tau)*target_param.data)

NUM_EPISODES = 101
Q_LEARNING_RATE = .001
POLICY_LEARNING_RATE = .0001
REPLAY_MEMORY_SIZE = 200000
BATCH_SIZE = 128
MIN_BUFFER_TO_PLAY = 10000
TAU = .01 # Update filter constant
OU_LAMBDA = 1
OU_SIGMA = 0.8
GAMMA = .99
VIDEO_PERIOD = 10

torch.autograd.set_detect_anomaly(True)

env = gym.make("MountainCarContinuous-v0")
# env= gym.make("Pendulum-v1")
evaluation_env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
# evaluation_env = gym.make("Pendulum-v1", render_mode="rgb_array")
evaluation_env = RecordVideo(evaluation_env, video_folder="cartpole-agent", name_prefix="ep",
                  episode_trigger=lambda x: x % VIDEO_PERIOD == 0)

online_policy = DeterministicPolicy(env.observation_space.shape[0], learning_rate=POLICY_LEARNING_RATE)
target_policy = DeterministicPolicy(env.observation_space.shape[0], learning_rate=POLICY_LEARNING_RATE)
target_policy.load_state_dict(online_policy.state_dict()) # initialise both networks with the same parameters

online_Q = Q_nn(env.observation_space.shape[0], env.action_space.shape[0], learning_rate=Q_LEARNING_RATE)
target_Q = Q_nn(env.observation_space.shape[0], env.action_space.shape[0], learning_rate=Q_LEARNING_RATE)
target_Q.load_state_dict(online_Q.state_dict()) # initialise both networks with the same parameters

replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
ou_noise = OU_noise(lam=OU_LAMBDA, sigma=OU_SIGMA) # Ornstein Uhlbeck noise

reward_list = []

online_policy_stats = {
    "episode_mean_actions": [],
    "episode_std_actions": []
}

target_policy_stats = {
    "episode_actions":  [],
    "episode_mean_actions": [],
    "episode_std_actions": [],
    "episode_rewards": [],
    "episode_mean_rewards": [],
    "episode_std_rewards": []
}

loss_dict = {
    "policy_loss": [],
    "q_value_loss": []
}

for episode in tqdm(range(NUM_EPISODES)):
    state = env.reset()[0]
    ou_noise.reset()
    done = False
    episode_actions = []
    online_policy.train()
    step = 0
    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = online_policy(state)
            action += ou_noise.sample()
            action = torch.clamp(action, -1, 1) # ensure action is in the correct range after adding noise
        episode_actions.append(action.item())

        next_state, reward, terminated, truncated, info = env.step(action.detach().numpy())

        replay_memory.push((state, action, reward, next_state, terminated))
        state = next_state

        done = terminated or truncated

        if len(replay_memory) >= MIN_BUFFER_TO_PLAY and step%20==0:
            batch = replay_memory.sample(BATCH_SIZE)
            target_q, predicted_q = compute_q_values(batch)
            update_online_q_net(target_q, predicted_q)
            update_policy(batch)
        step += 1

    soft_update(target_Q, online_Q, TAU)
    soft_update(target_policy, online_policy, TAU)

    # update statistics dictionary:
    online_policy_stats["episode_mean_actions"].append(statistics.mean(episode_actions))
    online_policy_stats["episode_std_actions"].append(statistics.stdev(episode_actions))

    # reset episode setting for evaluation:
    done = False
    state = evaluation_env.reset()[0]
    episode_actions = []
    episode_rewards = []
    online_policy.eval()
    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = online_policy(state)

        next_state, reward, terminated, truncated, _ = evaluation_env.step(action.detach().numpy())
        state = next_state
        done = terminated or truncated
        episode_actions.append(action.item())
        episode_rewards.append(reward)

    target_policy_stats["episode_mean_actions"].append(statistics.mean(episode_actions))
    target_policy_stats["episode_std_actions"].append(statistics.stdev(episode_actions))
    target_policy_stats["episode_mean_rewards"].append(statistics.mean(episode_rewards))
    target_policy_stats["episode_std_rewards"].append(statistics.stdev(episode_rewards))

env.close()


# Visualize data
plt.figure("Statistics")
plt.subplot(3,1,1)
plt.plot(online_policy_stats["episode_mean_actions"], c="r", label="Mean")
plt.plot(online_policy_stats["episode_std_actions"], c="b", label="Std")
plt.legend()
plt.title("Action statistics in training")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(target_policy_stats["episode_mean_actions"], c="r", label="Mean")
plt.plot(target_policy_stats["episode_std_actions"], c="b", label="Std")
plt.legend()
plt.title("Action statistics in evalutation")
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(target_policy_stats["episode_mean_rewards"], c="r", label="Mean")
plt.plot(target_policy_stats["episode_std_rewards"], c="b", label="Std")
plt.legend()
plt.title("Reward statistics in evaluation")
plt.xlabel("Episodes")
plt.grid(True)

plt.show()