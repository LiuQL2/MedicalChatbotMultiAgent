import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()



# env = gym.make('CartPole-v0')
env = gym.make('MountainCarContinuous-v0')
index = 0
while index < 10:
    action = env.action_space.sample()
    print('action', type(action), action)
    index += 1

print('action space', env.action_space)
print('obs space', env.observation_space)
print('threshold', env.spec.reward_threshold)
print('reward range', env.reward_range)
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.mean_affine1 = nn.Linear(2, 128)
        self.mean_affine2 = nn.Linear(128, 1)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = x.view(-1, 2)
        mean = F.relu(self.mean_affine1(x))
        mean = self.mean_affine2(mean)
        return F.tanh(mean)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    # print('state', state, state.size())
    mean = policy(state)
    m = Normal(mean, 1)
    # action = m.sample()
    action = m.rsample()
    policy.saved_log_probs.append(m.log_prob(action))
    action = action.view(1, )
    # print(action)
    # print(type(action.numpy()), action.numpy())

    return action.detach().cpu()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R )
        # policy_loss.append(-R.view(1))
    optimizer.zero_grad()
    # print('policy_loss', policy_loss)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            # env.render()
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()