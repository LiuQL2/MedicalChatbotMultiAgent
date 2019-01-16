# -*-coding:utf-8 -*
"""
The agent will maintain two ranked list of candidate disease and symptoms, the two list will be updated every turn based
on the information agent collected. The two ranked list will affect each other according <disease-symptom> pairs.
Agent will choose the first symptom with request as the agent action aiming to ask if the user has the symptom. The rank
model will change if the user's answer is no in continual several times.
"""

import random
import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))
from src.dialogue_system.agent.agent_dqn import AgentDQN
from src.dialogue_system.policy_learning.dqn_with_goal import DQNWithGoal


class AgentWithGoal(AgentDQN):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        super(AgentWithGoal, self).__init__(action_set=action_set, slot_set=slot_set,disease_symptom=disease_symptom, parameter=parameter)
        input_size = parameter.get("input_size_dqn")
        hidden_size = parameter.get("hidden_size_dqn", 100)
        output_size = len(self.action_space)
        del self.dqn
        self.dqn = DQNWithGoal(input_size=input_size, hidden_size=hidden_size, output_size=output_size, parameter=parameter)