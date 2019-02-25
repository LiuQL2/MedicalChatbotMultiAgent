# -*- coding: utf8 -*-
"""
使用 Policy Gradient 的方法训练一个termination function, 用于HRL中的 sub-goal是否完成。
"""

import torch
import torch.nn.functional
from collections import deque
import os
import sys
import numpy as np
from torch.distributions import Categorical, Bernoulli
from collections import namedtuple
sys.path.append(os.getcwd().replace("src/dialogue_system/policy_learning",""))
from src.dialogue_system.agent.utils import state_to_representation_last


class Policy(torch.nn.Module):
    """
    DQN model with one fully connected layer, written in pytorch.
    """
    def __init__(self, input_size, hidden_size, output_size, goal_num, parameter):
        super(Policy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = parameter
        self.goal_num = goal_num
        # different layers. Two layers.
        self.action_generator = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, 1, bias=True),
            # torch.nn.Sigmoid()
        )
        self.action_generator[3].bias = torch.nn.Parameter(torch.FloatTensor([-10.0]))

    def forward(self, x, goal):
        batch_size = x.size()[0]
        # one hot for goals
        goal_one_hot = torch.zeros(batch_size, self.goal_num).to(self.device)
        goal_one_hot.scatter_(1, goal.long().view(-1,1),1)
        input_x = torch.cat((x, goal_one_hot),1)
        action = self.action_generator(input_x)
        action = torch.sigmoid(action)
        return action


class InternalCritic(object):
    def __init__(self, input_size, hidden_size, output_size, goal_num, goal_embedding_value, slot_set, parameter):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = parameter
        self.critic = Policy(input_size, hidden_size, 2, goal_num, parameter)
        if torch.cuda.is_available():
            if parameter["multi_GPUs"] == True: # multi GPUs
                self.critic = torch.nn.DataParallel(self.critic)
            else:# Single GPU
                self.critic.cuda(device=self.device)
        self.experience_replay_pool = deque(maxlen=3000)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.params.get("dqn_learning_rate"))

    def training_with_one_episode(self, action_log_probs, reward_list):
        if len(reward_list) <= 1:
            return None
        # print('action log prob', len(action_log_probs), action_log_probs)
        # print('reward list', len(reward_list), reward_list)
        eps = np.finfo(np.float32).eps.item()
        policy_loss = []
        returns = []
        R = 0
        gamma = self.params.get("gamma")
        for r in reward_list[::-1]:
             R = r + gamma * R
             returns.insert(0, R)
        returns = torch.Tensor(returns).to(self.device)
        # print('returns1', returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        # print('returns', returns)
        for log_prob, R in zip(action_log_probs, returns):
            policy_loss.append(-log_prob * R)
        # print('policy loss:', policy_loss)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

    def next(self, state_rep, goal):
        state_rep = torch.Tensor([state_rep]).to(self.device)
        goal = torch.Tensor([goal]).long().to(self.device)
        action_prob = self.critic(state_rep, goal)
        # print('action_prob', action_prob)
        # m = Categorical(action_prob)
        m = Bernoulli(action_prob)
        action = m.sample()
        # policy.saved_log_probs.append(m.log_prob(action))
        return int(action.item()), m.log_prob(action)

    def save_model(self, model_performance,episodes_index, checkpoint_path):
        """
        Saving the trained model.

        Args:
            model_performance (dict): the test result of the model, which contains different metrics.
            episodes_index (int): the current step of training. And this will be appended to the model name at the end.
            checkpoint_path (str): the directory that the model is going to save to. Default None.
        """
        if os.path.isdir(checkpoint_path) == False:
            # os.mkdir(checkpoint_path)
            os.makedirs(checkpoint_path)
        agent_id = self.params.get("agent_id")
        disease_number = self.params.get("disease_number")
        success_rate = model_performance["success_rate"]
        average_reward = model_performance["average_reward"]
        average_turn = model_performance["average_turn"]
        average_wrong_disease = model_performance["average_wrong_disease"]
        model_file_name = os.path.join(checkpoint_path, "model_d" + str(disease_number) + "_agent" + str(agent_id) + "_s" + str(success_rate) + "_r" + str(average_reward) + "_t" + str(average_turn)\
                          + "_wd" + str(average_wrong_disease) + "_e-" + str(episodes_index) + ".pkl")

        torch.save(self.critic.state_dict(), model_file_name)

    def restore_model(self, saved_model):
        """
        Restoring the trained parameters for the model. Both current and target net are restored from the same parameter.

        Args:
            saved_model (str): the file name which is the trained model.
        """
        print("loading trained model", saved_model)
        if torch.cuda.is_available() is False:
            map_location = 'cpu'
        else:
            map_location = None
        self.critic.load_state_dict(torch.load(saved_model,map_location=map_location))