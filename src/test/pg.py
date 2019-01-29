
import torch
from torch import nn, optim
import numpy as np
import gym


env = gym.make('CartPole-v1')
env.seed(1)
torch.manual_seed(1)

# Hyperparameters
learning_rate = 0.01
gamma = 0.99


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

        self.gamma = gamma

        # Episode policy and reward history
        self.policy_history = torch.Tensor(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []


    def forward(self, x):
        model = torch.nn.Sequential(
        self.l1,
        nn.Dropout(p=0.6),
        nn.ReLU(),
        self.l2,
        nn.Softmax(dim=-1)
            )
        return model(x)




policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

def select_action(state):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(torch.Tensor(state))
    print('state', state)
    c = torch.distributions.Categorical(state)
    action = c.sample()

    # Add log probability of our chosen action to our history
    if policy.policy_history.dim() != 0:
        print(policy.policy_history, c.log_prob(action))
        print(policy.policy_history.size(), c.log_prob(action).size())
        policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action)])
    else:
        policy.policy_history = (c.log_prob(action))
    return action


def update_policy():
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, torch.Tensor(rewards)).mul(-1), -1))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.data[0])
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = torch.Tensor(torch.Tensor())
    policy.reward_episode = []


def main(episodes):
    running_reward = 10
    for episode in range(episodes):
        state = env.reset()  # Reset environment and record the starting state
        done = False

        for time in range(1000):
            action = select_action(state)
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.data[0])
            # Save reward
            policy.reward_episode.append(reward)
            if done:
                break

        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)
        update_policy()
        if episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward,
                                                                                                        time))
            break

main(50)