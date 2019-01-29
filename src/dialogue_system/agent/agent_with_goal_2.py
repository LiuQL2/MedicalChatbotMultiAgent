# -*- coding: utf8 -*-
"""
这个agent的整体模型中有一个critic，而internal critic有两个作用，一个是用来判断当前这个goal是否结束，第二个作用是用来判断如果这个goal
完成了，那么这个goal所对应的疾病时应该inform给用户，还是返回给master重新选择。
"""

import numpy as np
import copy
import sys, os
import random
from collections import namedtuple
from collections import deque
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))
from src.dialogue_system.agent.agent_dqn import AgentDQN as LowerAgent
from src.dialogue_system.policy_learning.dqn_torch import DQN
from src.dialogue_system.agent.utils import state_to_representation_last
from src.dialogue_system import dialogue_configuration


class AgentWithGoal(object):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        self.action_set = action_set
        self.slot_set = slot_set
        self.disease_symptom = disease_symptom

        # symptom distribution by diseases.
        temp_slot_set = copy.deepcopy(slot_set)
        temp_slot_set.pop('disease')
        self.disease_to_symptom_dist = {}
        total_count = np.zeros(len(temp_slot_set))
        for disease, v in self.disease_symptom.items():
            dist = np.zeros(len(temp_slot_set))
            for symptom, count in v['symptom'].items():
                dist[temp_slot_set[symptom]] = count
                total_count[temp_slot_set[symptom]] += count
            self.disease_to_symptom_dist[disease] = dist

        for disease in self.disease_to_symptom_dist.keys():
            self.disease_to_symptom_dist[disease] = self.disease_to_symptom_dist[disease] / total_count

        # Master policy.
        input_size = parameter.get("input_size_dqn")
        hidden_size = parameter.get("hidden_size_dqn", 100)
        self.output_size = parameter.get('goal_dim', 5)
        self.dqn = DQN(input_size=input_size,
                       hidden_size=hidden_size,
                       output_size=self.output_size,
                       parameter=parameter,
                       named_tuple=('state', 'agent_action', 'reward', 'next_state', 'episode_over'))
        self.parameter = parameter
        self.experience_replay_pool = deque(maxlen=parameter.get("experience_replay_pool_size"))

        # Lower agent.
        temp_parameter = copy.deepcopy(parameter)
        temp_parameter['input_size_dqn'] = input_size + self.output_size
        self.lower_agent = LowerAgent(action_set=action_set, slot_set=slot_set, disease_symptom=disease_symptom, parameter=temp_parameter)
        named_tuple = ('state', 'agent_action', 'reward', 'next_state', 'episode_over','goal')
        self.lower_agent.dqn.Transition = namedtuple('Transition', named_tuple)
        self.visitation_count = np.zeros(shape=(self.output_size, len(self.lower_agent.action_space))) # [goal_num, lower_action_num]

    def initialize(self):
        """
        Initializing an dialogue session.
        :return: nothing to return.
        """
        self.master_reward = 0.
        self.sub_task_terminal = True
        self.intrinsic_reward = 0.0
        self.lower_agent.initialize()

    def next(self, state, turn, greedy_strategy,**kwargs):
        """
        The master first select a goal, then the lower agent takes an action based on this goal and state.
        :param state: a vector, the representation of current dialogue state.
        :param turn: int, the time step of current dialogue session.
        :return: the agent action, a tuple consists of the selected agent action and action index.
        """
        if self.sub_task_terminal is True:
            self.master_reward = 0.0
            self.master_state = state
            self.master_agent_index = self.__master_next__(state, greedy_strategy)
        else:
            pass

        # Lower agent takes an agent.
        goal = np.zeros(self.output_size)
        goal[self.master_agent_index] = 1
        agent_action, action_index = self.lower_agent.next(state, turn, greedy_strategy, goal=goal)
        # intrinsic critic.
        self.sub_task_terminal, self.intrinsic_reward = self.intrinsic_critic(state, agent_action, self.master_agent_index)
        # print(self.master_agent_index, self.sub_task_terminal)
        return agent_action, action_index

    def __master_next__(self, state, greedy_strategy):
        # disease_symptom are not used in state_rep.
        epsilon = self.parameter.get("epsilon")
        state_rep = state_to_representation_last(state=state,
                                                 action_set=self.action_set,
                                                 slot_set=self.slot_set,
                                                 disease_symptom=self.disease_symptom,
                                                 max_turn=self.parameter["max_turn"])  # sequence representation.
        # Master agent takes an action, i.e., selects a goal.
        if greedy_strategy == True:
            greedy = random.random()
            if greedy < epsilon:
                master_action_index = random.randint(0, self.output_size - 1)
            else:
                master_action_index = self.dqn.predict(Xs=[state_rep])[1]
        # Evaluating mode.
        else:
            master_action_index = self.dqn.predict(Xs=[state_rep])[1]
        return master_action_index

    def train(self, batch):
        """
        Training the agent.
        Args:
            batch: the sam ple used to training.
        Return:
             dict with a key `loss` whose value it a float.
        """
        loss = self.dqn.singleBatch(batch=batch,params=self.parameter,weight_correction=self.parameter.get("weight_correction"))
        return loss

    def update_target_network(self):
        self.dqn.update_target_network()
        self.lower_agent.update_target_network()

    def save_model(self, model_performance, episodes_index, checkpoint_path=None):
        # Saving master agent
        self.dqn.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=checkpoint_path)
        # Saving lower agent
        temp_checkpoint_path = os.path.join(checkpoint_path, 'lower/')
        self.lower_agent.dqn.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=temp_checkpoint_path)

    def train_dqn(self):
        """
        Train dqn.
        :return:
        """
        # ('state', 'agent_action', 'reward', 'next_state', 'episode_over')
        # Training of master agent
        cur_bellman_err = 0.0
        batch_size = self.parameter.get("batch_size",16)
        for iter in range(int(len(self.experience_replay_pool) / (batch_size))):
            batch = random.sample(self.experience_replay_pool, batch_size)
            loss = self.train(batch=batch)
            cur_bellman_err += loss["loss"]
        print("[Master agent] cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) / (len(self.experience_replay_pool) + 1e-10), len(self.experience_replay_pool)))
        # Training of lower agents.
        self.lower_agent.train_dqn()

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over):
        """
        这里lower agent和master agent的sample都是在这里直接保存的，没有再另外调用函数。
        """
        # reward shaping
        alpha = self.parameter.get("weight_for_reward_shaping")
        shaping = self.reward_shaping(agent_action, self.master_agent_index)
        reward = reward + alpha * shaping
        # state to vec.
        state_rep = state_to_representation_last(state=state, action_set=self.action_set, slot_set=self.slot_set,disease_symptom=self.disease_symptom, max_turn=self.parameter['max_turn'])
        next_state_rep = state_to_representation_last(state=next_state, action_set=self.action_set,slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter['max_turn'])
        master_state_rep = state_to_representation_last(state=self.master_state, action_set=self.action_set,slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter['max_turn'])
        # samples of master agent.
        self.master_reward += reward
        if self.sub_task_terminal is False:
            pass
        else:
            self.experience_replay_pool.append((master_state_rep, self.master_agent_index, self.master_reward, next_state_rep, episode_over))

        # samples of lower agent
        goal = np.zeros(self.output_size)
        goal[self.master_agent_index] = 1
        state_rep = np.concatenate((state_rep, goal), axis=0)
        next_state_rep = np.concatenate((next_state_rep, goal), axis=0)

        #如果达到固定长度，同时去掉即将删除transition的计数。
        self.visitation_count[self.master_agent_index, agent_action] += 1
        if len(self.lower_agent.experience_replay_pool) == self.lower_agent.experience_replay_pool.maxlen:
            _, pre_agent_action, _, _, _, pre_master_action = self.lower_agent.experience_replay_pool.popleft()
            self.visitation_count[pre_master_action, pre_agent_action] -= 1
        # self.lower_agent.experience_replay_pool.append((state_rep, agent_action, self.intrinsic_reward, next_state_rep, self.sub_task_terminal, self.master_agent_index))
        self.lower_agent.experience_replay_pool.append((state_rep, agent_action, reward, next_state_rep, self.sub_task_terminal, self.master_agent_index))# extrinsic reward is returned to lower agent directly.

    def flush_pool(self):
        self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))
        self.lower_agent.flush_pool()
        self.visitation_count = np.zeros(shape=(self.output_size, len(self.lower_agent.action_space))) # [goal_num, lower_action_num]

    def intrinsic_critic(self, state, agent_action, goal):
        """
        Heuristic intrinsic critic. output intrinsic reward and the termination of this sub-task.
        """
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

        action_slot_dict = copy.deepcopy(agent_action['request_slots'])
        action_slot_dict.update(agent_action['inform_slots'])
        action_slot_dict.update(agent_action['explicit_inform_slots'])
        action_slot_dict.update(agent_action['implicit_inform_slots'])

        # if len(list(set(action_slot_dict.keys()) - set(slot_dict.keys()))) > 0:
        #     return True, 1
        # else:
        #     return False, 0

        if goal != 0: # symptom inquiring sub-task
            for new_slot in action_slot_dict.keys():
                if new_slot not in slot_dict.keys():
                    return True, 1
        elif goal == 0 and 'disease' in action_slot_dict.keys(): # disease sub-task
            return True, 1
        return False, 0

    def reward_shaping(self, lower_agent_action, goal):
        prob_action_goal = self.visitation_count[goal, lower_agent_action] / (self.visitation_count.sum() + 1e-8)
        prob_goal = self.visitation_count.sum(1)[goal] / (self.visitation_count.sum() + 1e-8)
        prob_action = self.visitation_count.sum(0)[lower_agent_action] / (self.visitation_count.sum() + 1e-8)
        return np.log(prob_action_goal / (prob_action * prob_goal + 1e-8))