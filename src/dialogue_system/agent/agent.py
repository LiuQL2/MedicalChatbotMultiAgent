# -*- coding: utf-8 -*-
"""
Basic agent class that other complicated agent, e.g., rule-based agent, DQN-based agent.
"""

import numpy as np
import copy
import sys, os
from collections import deque
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))
from src.dialogue_system import dialogue_configuration
from src.dialogue_system.agent.utils import state_to_representation_last


class Agent(object):
    """
    Basic class of agent.
    """
    def __init__(self, action_set, slot_set, disease_symptom, parameter, disease_as_action=True):

        symptom_set = set()
        for key, v in disease_symptom.items():
            # print(key, len(v['symptom'].keys()))
            symptom_set = symptom_set | set(list(v['symptom'].keys()))
        # exit(0)

        self.action_set = action_set
        self.slot_set = slot_set
        # self.disease_symptom = disease_symptom
        self.experience_replay_pool = deque(maxlen=parameter.get("experience_replay_pool_size"))
        self.parameter = parameter
        self.candidate_disease_list = []
        self.candidate_symptom_list = []
        self.action_space = self._build_action_space(disease_symptom,disease_as_action)
        self.disease_symptom = self.disease_symptom_clip(disease_symptom, 2.5, parameter)

        self.agent_action = {
            "turn":1,
            "action":None,
            "request_slots":{},
            "inform_slots":{},
            "explicit_inform_slots":{},
            "implicit_inform_slots":{},
            "speaker":"agent"
        }

    def initialize(self):
        """
        Initializing an dialogue session.
        :return: nothing to return.
        """
        self.candidate_disease_list = []
        self.candidate_symptom_list = []
        self.agent_action = {
            "turn":None,
            "action":None,
            "request_slots":{},
            "inform_slots":{},
            "explicit_inform_slots":{},
            "implicit_inform_slots":{},
            "speaker":"agent"
        }

    def next(self, *args, **kwargs):
        """
        Taking action based on different methods, e.g., DQN-based AgentDQN, rule-based AgentRule.
        Detail codes will be implemented in different sub-class of this class.
        :param state: a vector, the representation of current dialogue state.
        :param turn: int, the time step of current dialogue session.
        :return: the agent action, a tuple consists of the selected agent action and action index.
        """
        raise NotImplementedError('The `next` function of agent has not been implemented.')

    def train(self, batch):
        """
        Training the agent.
        Detail codes will be implemented in different sub-class of this class.
        :param batch: the sample used to training.
        :return:
        """
        raise NotImplementedError('The `train` function of agent has not been implemented.')

    def _build_action_space(self, disease_symptom, disease_as_action):
        """
        Building the Action Space for the RL-based Agent.
        All diseases are treated as actions.
        :return: Action Space, a list of feasible actions.
        """

        feasible_actions = [
            {'action': "confirm_question", 'inform_slots': {}, 'request_slots': {},"explicit_inform_slots":{}, "implicit_inform_slots":{}},
            {'action': "confirm_answer", 'inform_slots': {}, 'request_slots': {},"explicit_inform_slots":{}, "implicit_inform_slots":{}},
            {'action': "deny", 'inform_slots': {}, 'request_slots': {},"explicit_inform_slots":{}, "implicit_inform_slots":{}},
            {'action': dialogue_configuration.CLOSE_DIALOGUE, 'inform_slots': {}, 'request_slots': {},"explicit_inform_slots":{}, "implicit_inform_slots":{}},
            {'action': dialogue_configuration.THANKS, 'inform_slots': {}, 'request_slots': {}, "explicit_inform_slots": {}, "implicit_inform_slots": {}}
        ]
        # Adding the inform actions and request actions. And the slots are extracted from the links
        # between disease and symptom, i.e., disease_symptom
        slot_set = []
        for disease, v in disease_symptom.items():
            slot_set = slot_set + list(v["symptom"])
        slot_set = list(set(slot_set))
        for slot in slot_set:
            feasible_actions.append({'action': 'request', 'inform_slots': {}, 'request_slots': {slot: dialogue_configuration.VALUE_UNKNOWN},"explicit_inform_slots":{}, "implicit_inform_slots":{}})
            if slot != "disease":
                feasible_actions.append({'action': 'inform', 'inform_slots': {slot: True}, 'request_slots': {}, "explicit_inform_slots":{}, "implicit_inform_slots":{}})
        # Diseases as actions.
        if disease_as_action is True:
            for disease in disease_symptom.keys():
                feasible_actions.append({'action': 'inform', 'inform_slots': {"disease":disease}, 'request_slots': {},"explicit_inform_slots":{}, "implicit_inform_slots":{}})

        return feasible_actions

    @staticmethod
    def disease_symptom_clip(disease_symptom, denominator, parameter):
        """
        Keep the top min(symptom_num, max_turn//denominator) for each disease, and the related symptoms are sorted
        descendent according to their frequencies.
        Args:
            disease_symptom: a dict, key is the names of diseases, and the corresponding value is a dict too which
                contains the index of this disease and the related symptoms.
            denominator: int, the number of symptoms for each diseases is  max_turn // denominator.
            parameter: the super-parameter.
        Returns:
             and dict, whose keys are the names of diseases, and the values are dicts too with two keys: {'index', symptom}
        """
        max_turn = parameter.get('max_turn')
        temp_disease_symptom = copy.deepcopy(disease_symptom)
        for key, value in disease_symptom.items():
            symptom_list = sorted(value['symptom'].items(), key=lambda x: x[1], reverse=True)
            symptom_list = [v[0] for v in symptom_list]
            symptom_list = symptom_list[0:min(len(symptom_list), int(max_turn / float(denominator)))]
            temp_disease_symptom[key]['symptom'] = symptom_list
        return temp_disease_symptom

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over, **kwargs):
        symptom_dist_as_input = self.parameter.get("symptom_dist_as_input")
        agent_id = self.parameter.get("agent_id")

        state = state_to_representation_last(state=state, action_set=self.action_set, slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter["max_turn"])
        next_state = state_to_representation_last(state=next_state, action_set=self.action_set, slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter["max_turn"])
        if symptom_dist_as_input is True and agent_id.lower() == 'agenthrl':
            symptom_dist = kwargs.get('symptom_dist')
            state = np.concatenate((state, symptom_dist), axis=0)
            next_state = np.concatenate((next_state, symptom_dist), axis=0)
        self.experience_replay_pool.append((state, agent_action, reward, next_state, episode_over))

    def flush_pool(self):
        self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))
