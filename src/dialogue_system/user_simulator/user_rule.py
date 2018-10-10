# -*- coding: utf-8 -*-
"""
User simulator which is based on rules.
"""

import sys, os
sys.path.append(os.getcwd().replace("src/dialogue_system/user_simulator",""))
from src.dialogue_system.user_simulator.user import User


class UserRule(User):
    def __init__(self, goal_set, action_set, parameter):
        super(UserRule,self).__init__(goal_set=goal_set,action_set=action_set,parameter=parameter)