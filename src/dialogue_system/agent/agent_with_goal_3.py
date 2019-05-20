# -*- coding: utf8 -*-
"""
Agent for hierarchical reinforcement learning. The master agent first generates a goal, and the goal will be inputted
into the lower agent.
这里terminate function是想用policy gradient的方法进行训练，使用extrinsic reward来作为terminate function的reward来进行参数的更新，
不过目前测试不太好。不是EMNLP论文使用的模型
"""

import numpy as np
import copy
import sys, os
import random
import math
import torch
from collections import namedtuple
from collections import deque
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))
from src.dialogue_system.agent.agent_dqn import AgentDQN as LowerAgent
from src.dialogue_system.policy_learning.dqn_torch import DQN
from src.dialogue_system.agent.utils import state_to_representation_last
from src.dialogue_system import dialogue_configuration
from src.dialogue_system.policy_learning.internal_critic import InternalCritic
from src.dialogue_system.policy_learning.internal_critic_pg import InternalCritic as InternalCriticPG


class AgentWithGoal(object):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        self.action_set = action_set
        self.slot_set = slot_set
        self.disease_symptom = disease_symptom

        ##################
        # Master policy.
        #######################
        input_size = parameter.get("input_size_dqn")
        hidden_size = parameter.get("hidden_size_dqn", 100)
        self.output_size = parameter.get('goal_dim', 2*len(self.disease_symptom))
        self.dqn = DQN(input_size=input_size + self.output_size,
                       hidden_size=hidden_size,
                       output_size=self.output_size,
                       parameter=parameter,
                       named_tuple=('state', 'agent_action', 'reward', 'next_state', 'episode_over'))
        self.parameter = parameter
        self.experience_replay_pool = deque(maxlen=parameter.get("experience_replay_pool_size"))

        # # Initialize weights.
        # if torch.cuda.is_available():
        #     self.dqn.current_net.policy_layer[3].bias = torch.nn.Parameter(torch.Tensor([5]*len(disease_symptom) + [-5] * len(disease_symptom)).cuda())
        #     self.dqn.target_net.policy_layer[3].bias = torch.nn.Parameter(torch.Tensor([5]*len(disease_symptom) + [-5] * len(disease_symptom)).cuda())
        # else:
        #     self.dqn.current_net.policy_layer[3].bias = torch.nn.Parameter(torch.Tensor([5]*len(disease_symptom) + [-5] * len(disease_symptom)))
        #     self.dqn.target_net.policy_layer[3].bias = torch.nn.Parameter(torch.Tensor([5]*len(disease_symptom) + [-5] * len(disease_symptom)))
        #
        # if parameter.get("train_mode") is False :
        #     self.dqn.restore_model(parameter.get("saved_model"))
        #     self.dqn.current_net.eval()
        #     self.dqn.target_net.eval()

        ###############################
        # Internal critic
        ##############################
        # symptom distribution by diseases.
        temp_slot_set = copy.deepcopy(slot_set)
        temp_slot_set.pop('disease')
        self.disease_to_symptom_dist = {}
        self.id2disease = {}
        total_count = np.zeros(len(temp_slot_set))
        for disease, v in self.disease_symptom.items():
            dist = np.zeros(len(temp_slot_set))
            self.id2disease[v['index']] = disease
            for symptom, count in v['symptom'].items():
                dist[temp_slot_set[symptom]] = count
                total_count[temp_slot_set[symptom]] += count
            self.disease_to_symptom_dist[disease] = dist

        for disease in self.disease_to_symptom_dist.keys():
            self.disease_to_symptom_dist[disease] = self.disease_to_symptom_dist[disease] / total_count
        goal_embed_value = [0] * len(disease_symptom)
        for disease in self.disease_to_symptom_dist.keys():
            self.disease_to_symptom_dist[disease] = self.disease_to_symptom_dist[disease] / total_count
            goal_embed_value[disease_symptom[disease]['index']] = list(self.disease_to_symptom_dist[disease])

        temp_parameter = copy.deepcopy(parameter)
        path_list = parameter['saved_model'].split('/')
        path_list.insert(-1, 'critic')
        temp_parameter['saved_model'] = '/'.join(path_list)
        self.internal_critic = InternalCriticPG(input_size=input_size + self.output_size,
                                                hidden_size=50,
                                                output_size=2, goal_num=self.output_size,
                                                goal_embedding_value=goal_embed_value, slot_set=temp_slot_set,
                                                parameter=temp_parameter)
        if temp_parameter.get("train_mode") is False:
            self.internal_critic.restore_model(temp_parameter.get("saved_model"))
            self.internal_critic.critic.eval()

        #################
        # Lower agent.
        ##############
        temp_parameter = copy.deepcopy(parameter)
        temp_parameter['input_size_dqn'] = input_size + len(self.disease_symptom)
        path_list = parameter['saved_model'].split('/')
        path_list.insert(-1, 'lower')
        temp_parameter['saved_model'] = '/'.join(path_list)
        temp_parameter['gamma'] = temp_parameter['gamma_worker'] # discount factor for the lower agent.
        self.lower_agent = LowerAgent(action_set=action_set, slot_set=slot_set, disease_symptom=disease_symptom, parameter=temp_parameter,disease_as_action=False)
        # 为每一个下层的action计数，原类里面定义了为每种disease计数，这里还要为每个goal计数
        self.lower_agent.action_visitation_count["goal"] = {}
        for i in range(len(disease_symptom)):
            self.lower_agent.action_visitation_count["goal"].setdefault(i, dict())# 根据user那边的情况，为每个goal计数
        named_tuple = ('state', 'agent_action', 'reward', 'next_state', 'episode_over','goal')
        self.lower_agent.dqn.Transition = namedtuple('Transition', named_tuple)
        self.visitation_count = np.zeros(shape=(self.output_size, len(self.lower_agent.action_space))) # [goal_num, lower_action_num]
        if temp_parameter.get("train_mode") is False:
            self.lower_agent.dqn.restore_model(temp_parameter.get("saved_model"))
            self.lower_agent.dqn.current_net.eval()
            self.lower_agent.dqn.target_net.eval()

    def initialize(self):
        """
        Initializing an dialogue session.
        :return: nothing to return.
        """
        # print('{} new session {}'.format('*'*20, '*'*20))
        # print('***' * 20)
        self.dialogue_turn = 0
        self.master_reward = 0.
        self.sub_task_terminal = True
        self.inform_disease = False
        self.master_action_index = None
        self.last_master_action_index = None
        self.worker_action_index = None
        self.last_worker_action_index = None
        self.intrinsic_reward = 0.0
        self.sub_task_turn = 0
        self.master_takes_action_this_turn = False
        self.states_of_one_session = []

        self.master_previous_actions = set()
        self.worker_previous_actions = set()
        self.lower_agent.initialize()

        # For internal critic.
        self.critic_action_prob_of_one_session = []
        self.extrinsic_reward_of_one_session = []
        self.action = {'action': 'inform',
                       'inform_slots': {"disease": 'UNK'},
                        'request_slots': {},
                        "explicit_inform_slots": {},
                        "implicit_inform_slots": {}}

    def next(self, state, turn, greedy_strategy, **kwargs):
        """
        The master first select a goal, then the lower agent takes an action based on this goal and state.
        :param state: a vector, the representation of current dialogue state.
        :param turn: int, the time step of current dialogue session.
        :return: the agent action, a tuple consists of the selected agent action and action index.
        """
        self.dialogue_turn = turn
        self.disease_tag = kwargs.get("disease_tag")

        # The current sub-task is terminated or the first turn of the session.
        if self.sub_task_terminal is True or self.master_action_index is None:
            self.master_takes_action_this_turn = True
            self.master_reward = 0.0
            self.master_state = state
            self.sub_task_turn = 0
            self.last_master_action_index = copy.deepcopy(self.master_action_index)
            self.master_previous_actions.add(self.last_master_action_index)
            self.master_action_index = self.__master_next__(state, self.master_action_index, greedy_strategy)
        else:
            pass

        # Inform disease.
        if self.master_action_index >= len(self.disease_symptom):
            self.action["turn"] = turn
            self.action["inform_slots"] = {"disease": self.id2disease[self.master_action_index - len(self.disease_symptom)]}
            self.action["speaker"] = 'agent'
            self.action["action_index"] = None
            return self.action, None

        # print('turn: {}, goal: {}, label: {}, sub-task finish: {}, inform disease: {}, intrinsic reward: {}, similar score: {}'.format(turn, self.master_action_index, self.disease_symptom[self.disease_tag]['index'], self.sub_task_terminal, self.inform_disease, self.intrinsic_reward, similar_score))

        # Lower agent takes an agent. Not inform disease.
        self.master_takes_action_this_turn = False
        goal = np.zeros(len(self.disease_symptom))
        self.sub_task_turn += 1
        goal[self.master_action_index] = 1
        self.last_worker_action_index = self.worker_action_index
        self.worker_previous_actions.add(self.last_worker_action_index)
        agent_action, action_index = self.lower_agent.next(state, turn, greedy_strategy, goal=goal)
        self.worker_action_index = action_index
        # print('action', agent_action)
        # self.sub_task_terminal, self.intrinsic_reward, similar_score = self.intrinsic_critic_pg(state, self.master_action_index, keep_log_prob=True)

        return agent_action, action_index

    def __master_next__(self, state, last_master_action, greedy_strategy):
        # disease_symptom are not used in state_rep.
        epsilon = self.parameter.get("epsilon")
        state_rep = state_to_representation_last(state=state,
                                                 action_set=self.action_set,
                                                 slot_set=self.slot_set,
                                                 disease_symptom=self.disease_symptom,
                                                 max_turn=self.parameter["max_turn"])  # sequence representation.
        last_action_rep = np.zeros(self.output_size)
        if last_master_action is not None:
            last_action_rep[last_master_action] = 1
        state_rep = np.concatenate((state_rep, last_action_rep), axis=0)
        # Master agent takes an action, i.e., selects a goal.
        if greedy_strategy is True:
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

        # Saving internal_critic
        temp_checkpoint_path = os.path.join(checkpoint_path, 'critic/')
        self.internal_critic.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=temp_checkpoint_path)

    def train_dqn(self):
        """
        Train dqn.
        :return:
        """
        # ('state', 'agent_action', 'reward', 'next_state', 'episode_over')
        # Training of master agent
        cur_bellman_err = 0.0
        batch_size = self.parameter.get("batch_size",16)
        for iter in range(math.ceil(len(self.experience_replay_pool) / batch_size)):
            batch = random.sample(self.experience_replay_pool, min(batch_size,len(self.experience_replay_pool)))
            loss = self.train(batch=batch)
            cur_bellman_err += loss["loss"]
        print("[Master agent] cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) / (len(self.experience_replay_pool) + 1e-10), len(self.experience_replay_pool)))
        # Training of lower agents.
        self.lower_agent.train_dqn()
        # Training of internal critic.
        # self.internal_critic.buffer_replay()

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over):
        """
        这里lower agent和master agent的sample都是在这里直接保存的，没有再另外调用函数。
        """


        # for internal critic trained with Policy Gradient.
        self.extrinsic_reward_of_one_session.append(reward)
        self.sub_task_terminal, self.intrinsic_reward, similar_score = self.intrinsic_critic_pg(next_state, self.master_action_index, keep_log_prob=True)

        if episode_over is False:# 只要不是最后一轮都会经过internal critic
            pass
            # self.extrinsic_reward_of_one_session.append(reward)
        if self.parameter["train_mode"] is True and episode_over is True:
            # self.extrinsic_reward_of_one_session.pop(0)
            self.internal_critic.training_with_one_episode(self.critic_action_prob_of_one_session, self.extrinsic_reward_of_one_session)

        # state to vec.
        state_rep = state_to_representation_last(state=state, action_set=self.action_set, slot_set=self.slot_set,disease_symptom=self.disease_symptom, max_turn=self.parameter['max_turn'])
        next_state_rep = state_to_representation_last(state=next_state, action_set=self.action_set,slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter['max_turn'])
        master_state_rep = state_to_representation_last(state=self.master_state, action_set=self.action_set,slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter['max_turn'])

        self.master_reward += reward
        # 这一轮中master采取了动作，且下一轮中也要采取动作
        if self.sub_task_terminal is True and self.master_takes_action_this_turn is True:
            last_master_action_rep = np.zeros(self.output_size)
            current_master_action_rep = np.zeros(self.output_size)
            # 将master所有已经选择的动作加入到状态表示中。
            for last_master_action_index in self.master_previous_actions:
                if last_master_action_index is not None:
                    last_master_action_rep[last_master_action_index] = 1
                    current_master_action_rep[last_master_action_index] = 1
            if self.master_action_index is not None: current_master_action_rep[self.master_action_index] = 1
            master_state_rep = np.concatenate((master_state_rep, last_master_action_rep), axis=0)
            next_master_state_rep = np.concatenate((next_state_rep, current_master_action_rep), axis=0)
            self.experience_replay_pool.append((master_state_rep, self.master_action_index, reward, next_master_state_rep, episode_over))


        # samples of lower agent.
        if agent_action is not None: # session is not over. Otherwise the agent_action is not one of the lower agent's actions.
            goal = np.zeros(len(self.disease_symptom))
            goal[self.master_action_index] = 1
            state_rep = np.concatenate((state_rep, goal), axis=0)
            next_state_rep = np.concatenate((next_state_rep, goal), axis=0)
            # reward shaping for lower agent on intrinsic reward.
            alpha = self.parameter.get("weight_for_reward_shaping")
            shaping = self.reward_shaping(state, next_state)
            self.intrinsic_reward += alpha * shaping
            self.lower_agent.experience_replay_pool.append((state_rep, agent_action, self.intrinsic_reward, next_state_rep, self.sub_task_terminal, self.master_action_index))
            if self.dialogue_turn >=0:
                # visitation count. 因为这里没有调用lower_agent的record training sample 函数，所以要在这里进行计数。
                self.lower_agent.action_visitation_count["disease"][
                    self.disease_symptom[self.disease_tag]['index']].setdefault(agent_action, 0)
                self.lower_agent.action_visitation_count["disease"][self.disease_symptom[self.disease_tag]['index']][
                    agent_action] += 1
                self.lower_agent.action_visitation_count["goal"][self.master_action_index].setdefault(agent_action, 0)
                self.lower_agent.action_visitation_count["goal"][self.master_action_index][agent_action] += 1
                # 所有的计数
                self.lower_agent.action_visitation_count["total"].setdefault(agent_action, 0)
                self.lower_agent.action_visitation_count["total"][agent_action] += 1

    def flush_pool(self):
        self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))
        self.lower_agent.flush_pool()
        self.visitation_count = np.zeros(shape=(self.output_size, len(self.lower_agent.action_space))) # [goal_num, lower_action_num]

    def intrinsic_critic_pg(self, state, master_action_index, keep_log_prob):
        self.internal_critic.critic.eval()
        state_rep = state_to_representation_last(state, action_set=self.lower_agent.action_set, slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter["max_turn"])
        critic_action, action_log_prob = self.internal_critic.next(state_rep, master_action_index)
        if keep_log_prob is True:
            self.critic_action_prob_of_one_session.append(action_log_prob)
        else:
            pass
        # # sub task turn limitation.
        # if self.sub_task_turn >= 4:
        #     sub_task_terminal = True
        #     intrinsic_reward = self.parameter.get("reward_for_fail") / 2
        # else:
        #     sub_task_terminal = bool(critic_action)
        #     if sub_task_terminal:
        #         intrinsic_reward = self.parameter.get("reward_for_success") / 2
        #     else:
        #         intrinsic_reward = -1


        # No sub-task turn limitaion.
        sub_task_terminal = bool(critic_action)
        if sub_task_terminal:
            intrinsic_reward = self.parameter.get("reward_for_success") / 2
        else:
            intrinsic_reward = -1

        return sub_task_terminal, intrinsic_reward, 0

    def reward_shaping1(self, lower_agent_action, goal):
        prob_action_goal = self.visitation_count[goal, lower_agent_action] / (self.visitation_count.sum() + 1e-8)
        prob_goal = self.visitation_count.sum(1)[goal] / (self.visitation_count.sum() + 1e-8)
        prob_action = self.visitation_count.sum(0)[lower_agent_action] / (self.visitation_count.sum() + 1e-8)
        return np.log(prob_action_goal / (prob_action * prob_goal + 1e-8))

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
        return gamma * len(next_slot_dict) - len(slot_dict)

    def train_mode(self):
        self.dqn.current_net.train()
        self.lower_agent.dqn.current_net.train()
        self.internal_critic.critic.train()

    def eval_mode(self):
        self.dqn.current_net.eval()
        self.lower_agent.dqn.current_net.eval()
        self.internal_critic.critic.eval()

    def save_visitation(self, epoch_index):
        self.lower_agent.save_visitation(epoch_index)