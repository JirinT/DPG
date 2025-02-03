import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from tqdm import tqdm
from policy import Policy
from Q_value_function import QValue
from replay_memory import ReplayMemory
from value_function import ValueFunction
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

# functions:
def update_parameters(batch):
    for experience in batch:
        state = experience[0]
        action = experience[1]
        next_state = experience[2]
        reward = experience[3]

        delta = reward + GAMMA*Q_value_fcn(next_state, policy(next_state)) - Q_value_fcn(state, action)

        # update Q-parameters w:
        Q_value_fcn.w = Q_value_fcn.w.detach() + Q_A_LR*delta.detach()*Q_value_fcn.fi.detach()
        value_fcn.v = value_fcn.v.detach() + VALUE_LR*delta*state
        policy.theta = policy.theta.detach() + POLICY_LR*Q_value_fcn.w
        policy.theta.requires_grad = True

# HYPERPARAMETERS:
NUM_EPISODES = 5
POLICY_LR = .001
Q_A_LR = .03
VALUE_LR = .03
BATCH_SIZE = 50
BATCH_UPDATE = 10
GAMMA = .99
VIDEO_PERIOD = 1
# ------- MAIN ------:
torch.autograd.set_detect_anomaly(True)

# Environments init:
train_env = gym.make("MountainCarContinuous-v0")
test_env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
test_env = RecordEpisodeStatistics(test_env)
train_env = RecordEpisodeStatistics(train_env)
test_env = RecordVideo(test_env, video_folder="cartpole-agent", name_prefix="ep",
                  episode_trigger=lambda x: x % VIDEO_PERIOD == 0)

train_epi_stats = {
    "time" : [],
    "total_reward" : [],
    "length" : []
}

test_epi_stats = {
    "time" : [],
    "total_reward" : [],
    "length" : []
}

state_space = train_env.observation_space.shape[0]
action_space = train_env.action_space.shape[0]

# Init policy with theta parameters:
theta_init = torch.tensor(np.random.uniform(-1, 1, state_space), dtype=torch.float32, requires_grad=True)
policy = Policy(theta_init)
# Init value function with v parameters:
v_init = torch.tensor(np.random.uniform(-1, 1, state_space), dtype=torch.float32)
value_fcn = ValueFunction(v_init)
# Init Q-value function with w parameters:
w_init = torch.tensor(np.random.uniform(-1, 1, state_space), dtype=torch.float32)
Q_value_fcn = QValue(w_init, policy, value_fcn)

# Init replay memory:
replay_memory = ReplayMemory(100)

for _ in tqdm(range(NUM_EPISODES)):
    state = train_env.reset()[0]
    state = torch.tensor(state) # behavioral state
    done = False
    step = 0
    # first train on the behavioral policy - sample actions from random distribution
    while not done:
        action = train_env.action_space.sample() # behavioral action - just random sample
        next_state, reward, terminated, truncated, info = train_env.step(action)
        next_state = torch.Tensor(next_state) # behavioral state
        done = terminated or truncated
        
        experience = (state, action, next_state, reward)
        replay_memory.push(experience)

        state = next_state

        if step % BATCH_UPDATE == 0 and len(replay_memory) >= BATCH_SIZE:
            batch = replay_memory.sample_batch(BATCH_SIZE)
            update_parameters(batch)

    train_epi_stats["length"].append(info["episode"]["l"])
    train_epi_stats["time"].append(info["episode"]["t"])
    train_epi_stats["total_reward"].append(info["episode"]["r"])
    
    # now test the trained policy:
    done = False
    state = test_env.reset()[0]
    state = torch.tensor(state) # behavioral state

    while not done:
        action = policy(state)
        next_state, reward, terminated, truncated, info = test_env.step([action.item()])
        next_state = torch.Tensor(next_state) # behavioral state
        done = terminated or truncated
        
        state = next_state

    test_epi_stats["length"].append(info["episode"]["l"])
    test_epi_stats["time"].append(info["episode"]["t"])
    test_epi_stats["total_reward"].append(info["episode"]["r"])

test_env.close()
train_env.close()

# plot the results:

plt.figure()
plt.plot(range(NUM_EPISODES), train_epi_stats["total_reward"], c="b", label="Behavioral Policy")
plt.plot(range(NUM_EPISODES), test_epi_stats["total_reward"], c="r", label="Target Policy")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Rewards over episodes")
plt.grid(True)
plt.legend()
plt.show()