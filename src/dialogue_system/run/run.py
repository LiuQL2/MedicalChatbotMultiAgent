# -*- coding:utf-8 -*-

import time
import argparse
import pickle
import sys, os
import json
sys.path.append(os.getcwd().replace("src/dialogue_system/run",""))

from src.dialogue_system.dialogue_manager import DialogueManager
from src.dialogue_system.agent import AgentRandom
from src.dialogue_system.agent import AgentDQN
from src.dialogue_system.agent import AgentRule
from src.dialogue_system.agent import AgentActorCritic
from src.dialogue_system.user_simulator import UserRule as User
from src.dialogue_system import dialogue_configuration

from src.dialogue_system.run import RunningSteward


def construct_run_info(parameter):
    """
    Constructing a string which contains the primary super-parameters.

    Args:
        parameter: the super-parameter

    Returns:
        A dict, the updated parameter.
    """
    agent_id = parameter.get("agent_id")
    dqn_id = parameter.get("dqn_id")
    disease_number = parameter.get("disease_number")
    lr = parameter.get("dqn_learning_rate")
    reward_for_success = parameter.get("reward_for_success")
    reward_for_fail = parameter.get("reward_for_fail")
    reward_for_not_come_yet = parameter.get("reward_for_not_come_yet")
    reward_for_inform_right_symptom = parameter.get("reward_for_inform_right_symptom")

    max_turn = parameter.get("max_turn")
    minus_left_slots = parameter.get("minus_left_slots")
    gamma = parameter.get("gamma")
    epsilon = parameter.get("epsilon")
    data_set_name = parameter.get("goal_set").split("/")[-2]
    run_id = parameter.get('run_id')
    info = "learning_rate_d" + str(disease_number) + "_" + agent_id + \
           "_dqn" + str(dqn_id) + "_T" + str(max_turn) + "_lr" + str(lr) + "_RFS" + str(reward_for_success) + \
           "_RFF" + str(reward_for_fail) + "_RFNCY" + str(reward_for_not_come_yet) + "_RFIRS" + \
           str(reward_for_inform_right_symptom) + "_mls" + str(int(minus_left_slots)) + "_gamma" + str(gamma) + "_epsilon" + \
           str(epsilon) + "_RID" + str(run_id) + "_data" + str(data_set_name)
    parameter['run_info'] = info
    return parameter


def run(parameter):
    """
    The entry function of this code.

    Args:
        parameter: the super-parameter

    """
    agent_id = parameter.get("agent_id")
    dqn_id = parameter.get("dqn_id")
    disease_number = parameter.get("disease_number")
    max_turn = parameter.get("max_turn")

    if agent_id == 1:
        checkpoint_path = "./../model/dqn/checkpoint/checkpoint_d" + str(disease_number) + "_" + str(
            agent_id) + "_dqn" + str(dqn_id) + "_T" + str(max_turn) + "/"
    else:
        checkpoint_path = "./../model/dqn/checkpoint/checkpoint_d" + str(disease_number) + "_" + str(
            agent_id) + "_T" + str(max_turn) + "/"
    print(json.dumps(parameter, indent=2))
    time.sleep(1)
    slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
    action_set = pickle.load(file=open(parameter["action_set"], "rb"))
    disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))
    steward = RunningSteward(parameter=parameter,checkpoint_path=checkpoint_path)

    warm_start = parameter.get("warm_start")
    warm_start_epoch_number = parameter.get("warm_start_epoch_number")
    train_mode = parameter.get("train_mode")
    agent_id = parameter.get("agent_id")
    simulate_epoch_number = parameter.get("simulate_epoch_number")

    # Warm start.
    if warm_start == True and train_mode == True:
        print("warm starting...")
        agent = AgentRule(action_set=action_set,slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
        steward.warm_start(agent=agent,epoch_number=warm_start_epoch_number)
    # exit()
    if agent_id.lower() == 'agentdqn':
        agent = AgentDQN(action_set=action_set,slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
    elif agent_id.lower() == 'agentactorcritic':
        agent = AgentActorCritic(action_set=action_set,slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
    elif agent_id.lower() == 'agentrandom':
        agent = AgentRandom(action_set=action_set,slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
    elif agent_id.lower() == 'agentrule':
        agent = AgentRule(action_set=action_set,slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
    else:
        raise ValueError('Agent id should be one of [AgentRule, AgentDQN, AgentActorCritic, AgentRandom].')

    steward.simulate(agent=agent,epoch_number=simulate_epoch_number, train_mode=train_mode)

disease_number = 4

parser = argparse.ArgumentParser()
parser.add_argument("--disease_number", dest="disease_number", type=int,default=disease_number,help="the number of disease.")
parser.add_argument("--device_for_tf", dest="device_for_tf", type=str, default="/device:GPU:3", help="the device for tensorflow running on.")

# TODO: simulation configuration
parser.add_argument("--simulate_epoch_number", dest="simulate_epoch_number", type=int, default=1500, help="The number of simulate epoch.")
parser.add_argument("--epoch_size", dest="epoch_size", type=int, default=50, help="The number of simulated sessions in each simulated epoch.")
parser.add_argument("--evaluate_epoch_number", dest="evaluate_epoch_number", type=int, default=2000, help="the size of each simulate epoch when evaluation.")
parser.add_argument("--experience_replay_pool_size", dest="experience_replay_pool_size", type=int, default=20000, help="the size of experience replay.")
parser.add_argument("--hidden_size_dqn", dest="hidden_size_dqn", type=int, default=300, help="the hidden_size of DQN.")
parser.add_argument("--warm_start", dest="warm_start",type=bool, default=True, help="Filling the replay buffer with the experiences of rule-based agents. {True, False}")
parser.add_argument("--warm_start_epoch_number", dest="warm_start_epoch_number", type=int, default=20, help="the number of epoch of warm starting.")
parser.add_argument("--batch_size", dest="batch_size", type=int, default=30, help="the batch size when training.")
parser.add_argument("--log_dir", dest="log_dir", type=str, default="./../../../log/", help="directory where event file of training will be written, ending with /")
parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.1, help="The greedy probability of DQN")
parser.add_argument("--gamma", dest="gamma", type=float, default=1.0, help="The discount factor of immediate reward in RL.")
parser.add_argument("--train_mode", dest="train_mode", type=bool, default=True, help="Runing this code in training mode? [True, False]")

# TODO: Save model, performance and dialogue content ? And what is the path if yes?
parser.add_argument("--save_performance",dest="save_performance", type=bool, default=False, help="save the performance? [True, False]")
parser.add_argument("--performance_save_path",dest="performance_save_path", type=str, default="./../model/dqn/learning_rate04/", help="The folder where learning rate save to.")
parser.add_argument("--save_model", dest="save_model", type=bool, default=False,help="Save model during training? [True, False]")
parser.add_argument("--save_dialogue", dest="save_dialogue", type=bool, default=False, help="Save the dialogue? [True, False]")
parser.add_argument("--checkpoint_path",dest="checkpoint_path", type=str, default="./../model/dqn/checkpoint/", help="The folder where models save to.")
parser.add_argument("--saved_model", dest="saved_model", type=str, default="./../model/dqn/checkpoint/checkpoint_d4_agt1_dqn1/model_d4_agent1_dqn1_s0.619_r18.221_t4.266_wd0.0_e432.ckpt")
parser.add_argument("--dialogue_file", dest="dialogue_file", type=str, default="./../data/dialogue_output/dialogue_file.txt", help="the file that used to save dialogue content.")


parser.add_argument("--run_id", dest='run_id', type=int, default=1, help='the id of this running.')

# TODO: user configuration.
parser.add_argument("--allow_wrong_disease", dest="allow_wrong_disease", type=int, default=0, help="Allow the agent to inform wrong disease? 1:Yes, 0:No")

# TODO: Learning rate for actor-critic and dqn.
parser.add_argument("--dqn_learning_rate", dest="dqn_learning_rate", type=float, default=0.001, help="the learning rate of dqn.")
parser.add_argument("--actor_learning_rate", dest="actor_learning_rate", type=float, default=0.001, help="the learning rate of actor")
parser.add_argument("--critic_learning_rate", dest="critic_learning_rate", type=float, default=0.001, help="the learning rate of critic")
parser.add_argument("--trajectory_pool_size", dest="trajectory_pool_size", type=int, default=48, help="the size of trajectory pool")

# TODO: the number condition of explicit symptoms and implicit symptoms in each user goal.
parser.add_argument("--explicit_number", dest="explicit_number", type=int, default=0, help="the number of explicit symptoms of used sample")
# parser.add_argument("--implicit_number", dest="implicit_number", type=int, default=1, help="the number of implicit symptoms of used sample")
parser.add_argument("--implicit_number", dest="implicit_number", type=int, default=0, help="the number of implicit symptoms of used sample")

# TODO: agent to use.
parser.add_argument("--agent_id", dest="agent_id", type=str, default='AgentDQN', help="The agent to be used:[AgentRule, AgentDQN, AgentActorCritic, AgentRandom]")
parser.add_argument("--dqn_id", dest="dqn_id", type=int, default=1, help="the dqn to be used in agent:{0:initial dqn of qianlong, 1:dqn with one layer of qianlong, 2:dqn with two layers of qianlong, 3:dqn of Baolin.}")

# TODO: goal set, slot set, action set.
max_turn = 22
parser.add_argument("--action_set", dest="action_set", type=str, default='./../data/label/action_set.p',help='path and filename of the action set')
parser.add_argument("--slot_set", dest="slot_set", type=str, default='./../data/label/slot_set.p',help='path and filename of the slots set')
parser.add_argument("--goal_set", dest="goal_set", type=str, default='./../data/label/goal_set.p',help='path and filename of user goal')
parser.add_argument("--disease_symptom", dest="disease_symptom", type=str,default="./../data/label/disease_symptom.p",help="path and filename of the disease_symptom file")
parser.add_argument("--max_turn", dest="max_turn", type=int, default=max_turn, help="the max turn in one episode.")
# parser.add_argument("--input_size_dqn", dest="input_size_dqn", type=int, default=max_turn+137, help="the input_size of DQN.")
parser.add_argument("--input_size_dqn", dest="input_size_dqn", type=int, default=max_turn + 357, help="the input_size of DQN.")
parser.add_argument("--reward_for_not_come_yet", dest="reward_for_not_come_yet", type=float,default=-1)
parser.add_argument("--reward_for_success", dest="reward_for_success", type=float,default=2*max_turn)
parser.add_argument("--reward_for_fail", dest="reward_for_fail", type=float,default=-1.0*max_turn)
parser.add_argument("--reward_for_inform_right_symptom", dest="reward_for_inform_right_symptom", type=float,default=-1)
parser.add_argument("--minus_left_slots", dest="minus_left_slots", type=bool, default=False,help="Success reward minus the number of left slots as the final reward for a successful session.{True, False}")

parser.add_argument("--gpu", dest="gpu", type=int, default=0,help="The id of GPU on the running machine.")


args = parser.parse_args()
parameter = vars(args)


if __name__ == "__main__":
    gpu = parameter.get("gpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    parameter = construct_run_info(parameter)
    print(parameter['run_info'])

    run(parameter=parameter)