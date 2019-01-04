# -*- coding: utf8 -*-

import os

def verify_params(params):
    dqn_type = params.get("dqn_type")
    if dqn_type not in ['DQN', 'DoubleDQN']:
        raise ValueError("dqn_type should be one of ['DQN', 'DoubleDQN']")

    return construct_info(params)

def construct_info(params):
    """
    Constructing a string which contains the primary super-parameters.

    Args:
        params: the super-parameter

    Returns:
        A dict, the updated parameter.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu"]
    gpu_str = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_str.replace(' ', '')
    if len(gpu_str.split(',')) > 1:
        params.setdefault("multi_GPUs",True)
    else:
        params.setdefault("multi_GPUs", False)

    agent_id = params.get("agent_id")
    disease_number = params.get("disease_number")
    lr = params.get("dqn_learning_rate")
    reward_for_success = params.get("reward_for_success")
    reward_for_fail = params.get("reward_for_fail")
    reward_for_not_come_yet = params.get("reward_for_not_come_yet")
    reward_for_inform_right_symptom = params.get("reward_for_inform_right_symptom")
    allow_wrong_disease = params.get("allow_wrong_disease")
    check_related_symptoms = params.get("check_related_symptoms")

    max_turn = params.get("max_turn")
    minus_left_slots = params.get("minus_left_slots")
    gamma = params.get("gamma")
    epsilon = params.get("epsilon")
    data_set_name = params.get("goal_set").split("/")[-2]
    run_id = params.get('run_id')
    multi_gpu = params.get("multi_GPUs")
    dqn_type = params["dqn_type"]

    info = "learning_rate_d" + str(disease_number) + \
           "_" + agent_id + \
           "_T" + str(max_turn) + \
           "_lr" + str(lr) + \
           "_RFS" + str(reward_for_success) + \
           "_RFF" + str(reward_for_fail) + \
           "_RFNCY" + str(reward_for_not_come_yet) + \
           "_RFIRS" + str(reward_for_inform_right_symptom) +\
           "_mls" + str(int(minus_left_slots)) + \
           "_gamma" + str(gamma) + \
           "_epsilon" + str(epsilon) + \
           "_awd" + str(int(allow_wrong_disease)) + \
           "_crs" + str(int(check_related_symptoms)) + \
           "_RID" + str(run_id) + \
           "_data" + str(data_set_name) + \
           "_mGPU" + str(int(multi_gpu)) + \
           "_" + dqn_type
    params['run_info'] = info

    checkpoint_path = "./../model/" + dqn_type + "/checkpoint/" + info
    params["checkpoint_path"] = checkpoint_path

    performance_save_path = "./../model/" + dqn_type + "/performance/"
    params["performance_save_path"] = performance_save_path

    return params