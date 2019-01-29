# -*- coding:utf-8 -*-

import copy
import random
from collections import deque
import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system/dialogue_manager",""))

from src.dialogue_system.state_tracker import StateTracker as StateTracker
from src.dialogue_system import dialogue_configuration


class DialogueManager(object):
    """
    Dialogue manager of this dialogue system.
    """
    def __init__(self, user, agent, parameter):
        self.state_tracker = StateTracker(user=user, agent=agent, parameter=parameter)
        self.parameter = parameter
        self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))
        self.inform_wrong_disease_count = 0
        self.dialogue_output_file = parameter.get("dialogue_file")
        self.save_dialogue = parameter.get("save_dialogue")

    def next(self,train_mode, greedy_strategy):
        """
        The next two turn of this dialogue session. The agent will take action first and then followed by user simulator.
        :param save_record: bool, save record?
        :param train_mode: bool, True: the purpose of simulation is to train the model, False: just for simulation and the
                           parameters of the model will not be updated.
        :return: immediate reward for taking this agent action.
        """
        # Agent takes action.
        state = self.state_tracker.get_state()
        agent_action, action_index = self.state_tracker.agent.next(state=state,turn=self.state_tracker.turn,greedy_strategy=greedy_strategy, train_mode=train_mode)
        self.state_tracker.state_updater(agent_action=agent_action)
        # print("turn:%2d, state for agent:\n" % (state["turn"]) , json.dumps(state))

        # User takes action.
        user_action, reward, episode_over, dialogue_status = self.state_tracker.user.next(agent_action=agent_action,turn=self.state_tracker.turn)
        self.state_tracker.state_updater(user_action=user_action)
        # print("turn:%2d, update after user :\n" % (state["turn"]), json.dumps(state))

        # if self.state_tracker.turn == self.state_tracker.max_turn:
        #     episode_over = True

        if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_DISEASE:
            self.inform_wrong_disease_count += 1

        # if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
        #     print("success:", self.state_tracker.user.state)
        # elif dialogue_status == dialogue_configuration.DIALOGUE_STATUS_NOT_COME_YET:
        #     print("not come:", self.state_tracker.user.state)
        # else:
        #     print("failed:", self.state_tracker.user.state)
        # if len(self.state_tracker.user.state["rest_slots"].keys()) ==0:
        #     print(self.state_tracker.user.goal)
        #     print(dialogue_status,self.state_tracker.user.state)

        self.record_training_sample(
                state=state,
                agent_action=action_index,
                next_state=self.state_tracker.get_state(),
                reward=reward,
                episode_over=episode_over
            )

        # Output the dialogue.
        if episode_over == True and self.save_dialogue == True and train_mode == False:
            state = self.state_tracker.get_state()
            goal = self.state_tracker.user.get_goal()
            self.__output_dialogue(state=state, goal=goal)

        return reward, episode_over,dialogue_status

    def initialize(self, train_mode=True, goal_index=None):
        self.state_tracker.initialize()
        self.inform_wrong_disease_count = 0
        user_action = self.state_tracker.user.initialize(train_mode = train_mode, goal_index=goal_index)
        self.state_tracker.state_updater(user_action=user_action)
        self.state_tracker.agent.initialize()
        # print("#"*30 + "\n" + "user goal:\n", json.dumps(self.state_tracker.user.goal))
        # state = self.state_tracker.get_state()
        # print("turn:%2d, initialized state:\n" % (state["turn"]), json.dumps(state))

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over):
        self.state_tracker.agent.record_training_sample(state, agent_action, reward, next_state, episode_over)

    def set_agent(self,agent):
        self.state_tracker.set_agent(agent=agent)

    def train(self):
        self.state_tracker.agent.train_dqn()
        self.state_tracker.agent.update_target_network()

    def __output_dialogue(self,state, goal):
        history = state["history"]
        file = open(file=self.dialogue_output_file,mode="a+",encoding="utf-8")
        file.write("User goal: " + str(goal)+"\n")
        for turn in history:
            speaker = turn["speaker"]
            action = turn["action"]
            inform_slots = turn["inform_slots"]
            request_slots = turn["request_slots"]
            file.write(speaker + ": " + action + "; inform_slots:" + str(inform_slots) + "; request_slots:" + str(request_slots) + "\n")
        file.write("\n\n")
        file.close()