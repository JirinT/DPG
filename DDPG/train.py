import gymnasium as gym
import torch
import statistics
import numpy as np

from matplotlib import pyplot as plt
from tqdm import tqdm
from policy_nn import DeterministicPolicy
from replay_memory import ReplayMemory
# from Q_net import Q_nn
from halfed_Q_net import Q_nn
from ornstein_uhlbeck_noise import OU_noise
from normalizer import Normalizer
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

def compute_q_values(target_Q, online_Q, batch):
    states = torch.stack([torch.as_tensor(experience[0], dtype=torch.float32) for experience in batch])
    actions = torch.tensor([experience[1] for experience in batch], dtype=torch.float32).unsqueeze(dim=1)
    rewards = torch.tensor([experience[2] for experience in batch], dtype=torch.float32).unsqueeze(dim=1)
    next_states = torch.stack([torch.as_tensor(experience[3], dtype=torch.float32) for experience in batch])
    terminated = torch.tensor([int(experience[4]) for experience in batch], dtype=torch.float32).unsqueeze(dim=1)

    # compute target Q:
    with torch.no_grad():
        # normalized_next_states_actions = normalizer.normalize(torch.cat((next_states, target_policy(next_states)), dim=1))
        # next_q_values = target_Q(normalized_next_states_actions) # the input needs to be one tensor
        next_q_values = target_Q(next_states, target_policy(next_states)) # the input needs to be one tensor
    
    q_target = rewards + (1-terminated)*GAMMA*next_q_values.detach()

    # compute predicted Q:
    # normalized_states_actions = normalizer.normalize(torch.cat((states, actions), dim=1))
    # q_predicted = online_Q(normalized_states_actions)
    q_predicted = online_Q(states, actions)

    return q_target, q_predicted

def update_online_q_net(online_Q, target_q, predicted_q): # critic
    loss = Q_loss_fcn(predicted_q, target_q)
    loss_dict["q_value_loss"].append(loss.item())

    online_Q.optimizer.zero_grad()
    loss.backward()
    online_Q.optimizer.step()

def update_policy(online_policy, batch): # actor
    # the actions needs to be computed again because they need to be predicted by the current policy if I want to use it to update the current policy!!!
    # so the q_value needs to be recomputed as well!!!
    states = torch.stack([experience[0] for experience in batch])
    actions = online_policy(states) # deterministic actions used to update policy
    # normalized_states_actions = normalizer.normalize(torch.cat((states, actions), dim=1))
    # current_q_values = online_Q(normalized_states_actions)
    current_q_values = online_Q(states, actions)

    loss = -current_q_values.mean()
    loss_dict["policy_loss"].append(loss.item())

    online_policy.optimizer.zero_grad()
    loss.backward()
    online_policy.optimizer.step()

def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

NUM_EPISODES = 1003  # Increase from MountainCarContinuous since Pendulum is more complex
Q_LEARNING_RATE = 1e-3
POLICY_LEARNING_RATE = 1e-4
REPLAY_MEMORY_SIZE = 100000  # Larger buffer helps stabilize learning
BATCH_SIZE = 256
MIN_BUFFER_TO_PLAY = 256  # Start training after collecting enough experience
TAU = 5e-3 # Smaller value ensures smooth target network updates
OU_THETA = .2  # Lower noise decay rate for more stability
OU_SIGMA = .7  # Less exploration noise than MountainCarContinuous
GAMMA = 0.99  # Standard discount factor
VIDEO_PERIOD = 100  # For evaluation/visualization
UPDATE_PERIOD = 20
STD_NOISE = 0.5

# NUM_EPISODES = 1000  # Increase from MountainCarContinuous since Pendulum is more complex
# Q_LEARNING_RATE = 1e-4
# POLICY_LEARNING_RATE = 1e-5
# REPLAY_MEMORY_SIZE = 100000  # Larger buffer helps stabilize learning
# BATCH_SIZE = 256
# MIN_BUFFER_TO_PLAY = 256  # Start training after collecting enough experience
# TAU = 5e-3 # Smaller value ensures smooth target network updates
# OU_THETA = .15  # Lower noise decay rate for more stability
# OU_SIGMA = 0.2  # Less exploration noise than MountainCarContinuous
# GAMMA = 0.95  # Standard discount factor
# VIDEO_PERIOD = 100  # For evaluation/visualization
# UPDATE_PERIOD = 20
# STD_NOISE = 0.2

torch.autograd.set_detect_anomaly(True)

env = gym.make("MountainCarContinuous-v0")
# env = gym.make("Pendulum-v1")
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
Q_loss_fcn = torch.nn.MSELoss()

normalizer = Normalizer(num_features=4)
# set target policies to eval mode since they do not require gradients
target_policy.eval()
target_Q.eval()

replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
ou_noise = OU_noise(theta=OU_THETA, sigma=OU_SIGMA) # Ornstein Uhlbeck noise

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

q_values = np.array([])
step = 0
for episode in tqdm(range(NUM_EPISODES)):
    state = env.reset()[0]
    ou_noise.reset()
    done = False
    episode_actions = []
    online_policy.train()

    while not done:
        step += 1
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = online_policy(state)
            action += ou_noise.sample()
            # action += torch.normal(mean=torch.tensor(0, dtype=torch.float32), std=torch.tensor(STD_NOISE, dtype=torch.float32))
            action = torch.clamp(action, -1, 1) # ensure action is in the correct range after adding noise
        episode_actions.append(action.item())

        next_state, reward, terminated, truncated, info = env.step(action.detach().numpy())

        replay_memory.push((state.detach(), action.detach(), reward, next_state, terminated))
        state = next_state

        done = terminated or truncated

        if len(replay_memory) >= MIN_BUFFER_TO_PLAY and step%UPDATE_PERIOD == 0:
            batch = replay_memory.sample(BATCH_SIZE)
            target_q, predicted_q = compute_q_values(target_Q, online_Q, batch)
            q_values = np.append(q_values, predicted_q.detach().numpy())
            update_online_q_net(online_Q, target_q, predicted_q)
            update_policy(online_policy, batch)
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

plt.figure("Learning curves")
plt.subplot(2,1,1)
plt.plot(loss_dict["policy_loss"], c="r")
plt.ylabel("Loss")
plt.title("Policy learning curve")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(loss_dict["q_value_loss"], c="r")
plt.ylabel("Loss")
plt.title("Q-value learning curve")
plt.grid(True)

plt.figure("Q-values")
plt.plot(q_values, c="r")
plt.title("Predicted Q-values")
plt.xlabel("steps")
plt.grid(True)

plt.show()