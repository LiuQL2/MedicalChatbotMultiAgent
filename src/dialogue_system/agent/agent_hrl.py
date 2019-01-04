# -*- coding: utf-8 -*-
"""
Agent for hierarchical reinforcement learning.
"""

import numpy as np
import copy
import sys, os
import random
from collections import deque
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))
from src.dialogue_system import dialogue_configuration
from src.dialogue_system.agent.agent import Agent
from src.dialogue_system.agent.agent_dqn import AgentDQN as LowerAgent
from src.dialogue_system.policy_learning.dqn_torch import DQN
from src.dialogue_system.agent.utils import state_to_representation_last


class AgentHRL(object):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        self.action_set = action_set
        self.slot_set = slot_set
        self.disease_symptom = self.disease_symptom
        ##################################
        # Building lower agents. The state representation that the master agent and lower agents are the same, so the
        # slot set are same among these agents.
        ###########################
        self.id2disease = {}
        self.id2lowerAgent = {}
        for disease, v in disease_symptom.keys():
            self.id2disease[v["index"]] = disease
            temp_disease_symptom = {}
            temp_disease_symptom[disease] = {}
            temp_disease_symptom[disease]["index"] = 0
            temp_disease_symptom[disease]["symptom"] = v["symptom"]
            temp_slot_set = {}
            for symptom in v['symptom'].keys():
                temp_slot_set.setdefault(symptom, len(temp_slot_set))
            self.id2lowerAgent[v["index"]] = LowerAgent(action_set=action_set, slot_set=slot_set, disease_symptom=temp_disease_symptom, parameter=parameter)

        # Master policy.
        input_size = parameter.get("input_size_dqn")
        hidden_size = parameter.get("hidden_size_dqn", 100)
        output_size = len(self.id2lowerAgent)
        self.dqn = DQN(input_size=input_size, hidden_size=hidden_size,output_size=output_size, parameter=parameter)
        self.parameter = self.parameter
        self.experience_replay_pool = deque(maxlen=parameter.get("experience_replay_pool_size"))
        self.current_lower_agent_id = -1

    def next(self, state, turn, greedy_strategy):
        """
        Taking action based on different methods, e.g., DQN-based AgentDQN, rule-based AgentRule.
        Detail codes will be implemented in different sub-class of this class.
        :param state: a vector, the representation of current dialogue state.
        :param turn: int, the time step of current dialogue session.
        :return: the agent action, a tuple consists of the selected agent action and action index.
        """
        # disease_symptom are not used in state_rep.
        state_rep = state_to_representation_last(state=state,
                                                 action_set=self.action_set,
                                                 slot_set=self.slot_set,
                                                 disease_symptom=self.disease_symptom,
                                                 max_turn=self.parameter["max_turn"]) # sequence representation.

        # Master agent takes an action.
        if greedy_strategy == True:
            greedy = random.random()
            if greedy < self.parameter.get("epsilon"):
                action_index = random.randint(0, len(self.id2lowerAgent) - 1)
            else:
                action_index = self.dqn.predict(Xs=[state_rep])[1]
        # Evaluating mode.
        else:
            action_index = self.dqn.predict(Xs=[state_rep])[1]

        self.current_lower_agent_id = self.id2lowerAgent[action_index]

        # Lower agent takes an agent.
        agent_action, action_index = self.id2lowerAgent[self.current_lower_agent_id].next(state, turn, greedy_strategy)
        return agent_action, action_index

    def train(self, batch):
        """
        Training the agent.
        Args:
            batch: the sample used to training.

        Return:
             dict with a key `loss` whose value it a float.
        """
        loss = self.dqn.singleBatch(batch=batch,params=self.parameter)
        return loss

    def update_target_network(self):
        self.dqn.update_target_network()

    def save_model(self, model_performance, episodes_index, checkpoint_path=None):
        # Saving master agent
        self.dqn.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=checkpoint_path)
        # Saving lower agent
        for key, lower_agent in self.id2lowerAgent.items():
            checkpoint_path = os.path.join(checkpoint_path, 'lower/' + str(key))
            lower_agent.dqn.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=checkpoint_path)

    def train_dqn(self):
        """
        Train dqn.
        :return:
        """
        # Training of master agent
        cur_bellman_err = 0.0
        batch_size = self.parameter.get("batch_size",16)
        for iter in range(int(len(self.experience_replay_pool) / (batch_size))):
            batch = random.sample(self.experience_replay_pool, batch_size)
            loss = self.train(batch=batch)
            cur_bellman_err += loss["loss"]
        print("[Master agent] cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) / (len(self.experience_replay_pool) + 1e-10), len(self.experience_replay_pool)))

        # Training of lower agents.
        for disease_id, lower_agent in self.id2lowerAgent.items():
            print(disease_id)
            lower_agent.train_dqn()

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over):
        # samples of lower agent
        self.id2lowerAgent[self.current_lower_agent_id].record_training_sample(state, agent_action, reward, next_state, episode_over)

        # samples of master agent.
        state_rep = state_to_representation_last(state=state,
                                                 action_set=self.action_set,
                                                 slot_set=self.slot_set,
                                                 disease_symptom=self.disease_symptom,
                                                 max_turn=self.parameter["max_turn"])
        next_state_rep = state_to_representation_last(state=next_state,
                                                      action_set=self.action_set,
                                                      slot_set=self.slot_set,
                                                      disease_symptom=self.disease_symptom,
                                                      max_turn=self.parameter["max_turn"])
        q_values = self.id2lowerAgent[self.current_lower_agent_id].get_q_values(state)
        master_reward = max(q_values)
        self.experience_replay_pool.append((state_rep, self.current_lower_agent_id, master_reward, next_state_rep, episode_over))