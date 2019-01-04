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
from src.dialogue_system.agent.agent import Agent
from src.dialogue_system.policy_learning.dqn_torch import DQN
from src.dialogue_system.agent.utils import state_to_representation_last, state_to_representation_history


class AgentDQN(Agent):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        super(AgentDQN, self).__init__(action_set=action_set, slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
        input_size = parameter.get("input_size_dqn")
        hidden_size = parameter.get("hidden_size_dqn", 100)
        output_size = len(self.action_sapce)

        self.dqn = DQN(input_size=input_size, hidden_size=hidden_size,output_size=output_size, parameter=parameter)

    def next(self, state, turn, greedy_strategy):
        """
        Taking action based on different methods, e.g., DQN-based AgentDQN, rule-based AgentRule.
        Detail codes will be implemented in different sub-class of this class.
        :param state: a vector, the representation of current dialogue state.
        :param turn: int, the time step of current dialogue session.
        :return: the agent action, a tuple consists of the selected agent action and action index.
        """
        self.agent_action["turn"] = turn
        state_rep = state_to_representation_last(state=state, action_set=self.action_set, slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter["max_turn"]) # sequence representation.

        if greedy_strategy == True:
            greedy = random.random()
            if greedy < self.parameter.get("epsilon"):
                action_index = random.randint(0, len(self.action_sapce) - 1)
            else:
                action_index = self.dqn.predict(Xs=[state_rep])[1]
        # Evaluating mode.
        else:
            action_index = self.dqn.predict(Xs=[state_rep])[1]

        agent_action = self.action_sapce[action_index]
        agent_action["turn"] = turn
        agent_action["speaker"] = "agent"

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

    def save_model(self, model_performance,episodes_index, checkpoint_path = None):
        self.dqn.save_model(model_performance=model_performance, episodes_index = episodes_index, checkpoint_path=checkpoint_path)

    def train_dqn(self):
        """
        Train dqn.
        :return:
        """
        cur_bellman_err = 0.0
        batch_size = self.parameter.get("batch_size", 16)
        for iter in range(int(len(self.experience_replay_pool) / (batch_size))):
            batch = random.sample(self.experience_replay_pool, batch_size)
            loss = self.train(batch=batch)
            cur_bellman_err += loss["loss"]
        print("cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) / (len(self.experience_replay_pool) + 1e-10), len(self.experience_replay_pool)))

    def get_q_values(self, state):
        state_rep = state_to_representation_last(state=state, action_set=self.action_set, slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter["max_turn"])
        Q_values = self.dqn.predict(Xs=[state_rep])
        return Q_values
