# -*-coding:utf-8 -*
"""
The agent will maintain two ranked list of candidate disease and symptoms, the two list will be updated every turn based
on the information agent collected. The two ranked list will affect each other according <disease-symptom> pairs.
Agent will choose the first symptom with request as the agent action aiming to ask if the user has the symptom. The rank
model will change if the user's answer is no in continual several times.
"""

import random
import sys, os
import copy
import json
import numpy as np
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))
from src.dialogue_system.agent.agent_dqn import AgentDQN
from src.dialogue_system.policy_learning.dqn_with_goal_joint import DQNWithGoalJoint
from src.dialogue_system.agent.utils import state_to_representation_last
from src.dialogue_system import dialogue_configuration

random.seed(12345)


class AgentWithGoalJoint(AgentDQN):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        super(AgentWithGoalJoint, self).__init__(action_set=action_set, slot_set=slot_set,disease_symptom=disease_symptom, parameter=parameter)
        input_size = parameter.get("input_size_dqn")
        hidden_size = parameter.get("hidden_size_dqn", 100)
        output_size = len(self.action_space)
        del self.dqn
        self.dqn = DQNWithGoalJoint(input_size=input_size, hidden_size=hidden_size, output_size=output_size, parameter=parameter)

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over, **kwargs):
        shaping = self.reward_shaping(state, next_state)
        alpha = self.parameter.get("weight_for_reward_shaping")
        # if True:
        #     print('shaping', shaping)
        reward = reward + alpha * shaping
        state_rep = state_to_representation_last(state=state, action_set=self.action_set, slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter["max_turn"])
        next_state_rep = state_to_representation_last(state=next_state, action_set=self.action_set, slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter["max_turn"])
        self.experience_replay_pool.append((state_rep, agent_action, reward, next_state_rep, episode_over))

    def reward_shaping(self, state, next_state):
        def delete_item_from_dict(item, value):
            new_item = {}
            for k, v in item.items():
                if v != value: new_item[k] = v
            return new_item

        # slot number in state.
        slot_dict = copy.deepcopy(state["current_slots"]["inform_slots"])
        slot_dict.update(state["current_slots"]["explicit_inform_slots"])
        slot_dict.update(state["current_slots"]["implicit_inform_slots"])
        slot_dict.update(state["current_slots"]["proposed_slots"])
        slot_dict.update(state["current_slots"]["agent_request_slots"])
        slot_dict = delete_item_from_dict(slot_dict, dialogue_configuration.I_DO_NOT_KNOW)

        next_slot_dict = copy.deepcopy(next_state["current_slots"]["inform_slots"])
        next_slot_dict.update(next_state["current_slots"]["explicit_inform_slots"])
        next_slot_dict.update(next_state["current_slots"]["implicit_inform_slots"])
        next_slot_dict.update(next_state["current_slots"]["proposed_slots"])
        next_slot_dict.update(next_state["current_slots"]["agent_request_slots"])
        next_slot_dict = delete_item_from_dict(next_slot_dict, dialogue_configuration.I_DO_NOT_KNOW)
        gamma = self.parameter.get("gamma")
        # print(len(slot_dict), len(next_slot_dict))
        # print(slot_dict)
        # print(next_slot_dict)
        # print(state)
        # print(next_state)
        return gamma * len(next_slot_dict) - len(slot_dict)
