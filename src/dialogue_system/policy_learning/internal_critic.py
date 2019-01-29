# -*- coding: utf8 -*-
"""
Internal critic for HRL agent.
"""

import torch
import numpy as np
import sys, os
import pickle
import copy
from collections import deque
from src.dialogue_system import dialogue_configuration


def state_vec(slot_set, state):
    current_slots = copy.deepcopy(state["current_slots"]["inform_slots"])
    current_slots.update(state["current_slots"]["explicit_inform_slots"])
    current_slots.update(state["current_slots"]["implicit_inform_slots"])
    current_slots.update(state["current_slots"]["proposed_slots"])
    current_slots.update(state["current_slots"]["agent_request_slots"])
    # one-hot vector for each symptom.
    current_slots_rep = np.zeros((len(slot_set.keys()),3))
    for slot in current_slots.keys():
        # different values for different slot values.
        if current_slots[slot] == True:
            current_slots_rep[slot_set[slot]][0] = 1.0
        elif current_slots[slot] == False:
            current_slots_rep[slot_set[slot]][1] = 1.0
        elif current_slots[slot] == 'UNK':
            current_slots_rep[slot_set[slot]][2] = 1.0
        # elif current_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
        #     current_slots_rep[slot_set[slot]][3] = 1.0
    current_slots_rep = np.reshape(current_slots_rep, (len(slot_set.keys())*3))
    return current_slots_rep


class CriticModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, goal_num, goal_embedding_value):
        super(CriticModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.goal_num = goal_num
        self.goal_embed_layer = torch.nn.Embedding.from_pretrained(torch.Tensor(goal_embedding_value), freeze=True)
        self.goal_embed_layer.weight.requires_grad_(False)

        self.goal_generator_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x, goal):
        batch_size = x.size()[0]
        # one hot for goals
        goal_one_hot = torch.zeros(batch_size, self.goal_num).to(self.device)
        goal_one_hot.scatter_(1, goal.long().view(-1,1),1)
        input_x = torch.cat((x, goal_one_hot),1)
        goal_gen = self.goal_generator_layer(input_x)

        # cosine similarity.
        goal_embedding = self.goal_embed_layer(goal.long())
        similarity = torch.nn.functional.cosine_similarity(goal_embedding, goal_gen)
        return goal_gen, similarity


class InternalCritic(object):
    def __init__(self, input_size, hidden_size, output_size, goal_num,goal_embedding_value, slot_set, parameter):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = CriticModel(input_size, hidden_size, output_size, goal_num, goal_embedding_value)
        if torch.cuda.is_available():
            if parameter["multi_GPUs"] == True: # multi GPUs
                self.critic = torch.nn.DataParallel(self.critic)
            else:# Single GPU
                self.critic.cuda(device=self.device)
        self.slot_set = slot_set
        self.positive_sample_buffer = deque(maxlen=2000)
        self.negative_sample_buffer = deque(maxlen=2000)
        self.optimizer = torch.optim.Adam(params=self.critic.parameters(),lr=parameter.get("dqn_learning_rate"))

    def train(self, positive_data_batch, positive_goal, negative_data_batch, negative_goal,
              positive_weight=1, negative_weight=1):
        positive_data_batch = torch.Tensor(positive_data_batch).to(self.device)
        positive_goal = torch.Tensor(positive_goal).to(self.device)
        negative_data_batch = torch.Tensor(negative_data_batch).to(self.device)
        negative_goal = torch.Tensor(negative_goal).to(self.device)
        _, positive_similarity = self.critic(positive_data_batch, positive_goal)
        _, negative_similarity = self.critic(negative_data_batch, negative_goal)
        positive_loss = torch.mean(positive_similarity)
        negative_loss = torch.mean(negative_similarity)
        loss = - positive_weight * positive_loss + negative_weight * negative_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'total_loss': loss.item(), 'positive_similarity':positive_loss.item(), 'negative_similarity':negative_loss.item()}

    def save_model(self, model_path):
        torch.save(self.critic.state_dict(), model_path)

    def get_similarity(self, batch, goal):
        batch = torch.Tensor(batch).to(self.device)
        goal = torch.Tensor(goal).to(self.device)
        goal_gen, similarity = self.critic(batch, goal)
        return similarity.detach().cpu().numpy()

    def get_similarity_state_dict(self, batch, goal):
        new_batch = [state_vec(self.slot_set, state) for state in batch]
        return self.get_similarity(new_batch, goal)

    def restore_model(self, saved_model):
        print('loading model from {}'.format(saved_model))
        self.critic.load_state_dict(torch.load(saved_model))
