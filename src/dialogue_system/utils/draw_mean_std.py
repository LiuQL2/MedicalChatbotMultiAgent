# -*-coding:utf8 -*-
'''
Created on Nov 3, 2016

draw a learning curve

@author: xiul
'''

import argparse, json, pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="darkgrid")
# sns.set(font_scale=1.4)
sns.set(font_scale=1.2)

# width = 8
# height = 5.8
width = 8
height = 6.5
# plt.figure(figsize=(width, height))

linewidth = 1.1


def read_performance_records(path):
    """ load the performance score (.json) file """
    success_rate = []
    data = pickle.load(open(path, 'rb'))
    print(data.keys())
    for i, value in data.items():
        success_rate.append(value['success_rate'])
    # for key in sorted(data['success_rate'].keys(), key=lambda k:int(k)):
    #     # print key
    #     if int(key) > -1:
    #         success_rate.append(data['success_rate'][key])
    #         # print("%s\t%s\t%s\t%s" % (key, data['success_rate'][key], data['ave_turns'][key], data['ave_reward'][key]))

    smooth_num = 1
    d = [success_rate[i * smooth_num:i * smooth_num + smooth_num] for i in range(int(len(success_rate) / smooth_num))]

    success_rate_new = []
    cache = 0
    alpha = 0.8
    for i in d:
        cur = sum(i) / float(smooth_num)
        cache = cache * alpha + (1 - alpha) * cur
        success_rate_new.append(cache)

    return success_rate_new


def load_performance_file(path):
    """ load the performance score (.json) file """

    data = json.load(open(path, 'rb'))
    numbers = {'x': [], 'success_rate': [], 'ave_turns': [], 'ave_rewards': []}
    keylist = [int(key) for key in data['success_rate'].keys()]
    keylist.sort()

    for key in keylist:
        if int(key) > -1:
            numbers['x'].append(int(key))
            numbers['success_rate'].append(data['success_rate'][str(key)])
            numbers['ave_turns'].append(data['ave_turns'][str(key)])
            numbers['ave_rewards'].append(data['ave_reward'][str(key)])
    return numbers


def load_performance_file_ma(path):
    """ load the performance score (.json) file """

    data = json.load(open(path, 'rb'))
    numbers = {'x': [], 'success_rate': [], 'ave_turns': [], 'ave_rewards': []}
    keylist = [int(key) for key in data['success_rate'].keys()]
    keylist.sort()

    for key in keylist:
        if int(key) > -1:
            numbers['x'].append(int(key))
            numbers['success_rate'].append(data['success_rate'][str(key)])
            numbers['ave_turns'].append(data['ave_turns'][str(key)])
            numbers['ave_rewards'].append(data['ave_reward'][str(key)])

    smooth_num = 1
    d = [numbers['success_rate'][i * smooth_num:i * smooth_num + smooth_num] for i in
         range(int(len(numbers['success_rate']) / smooth_num))]

    success_rate_new = []
    cache = 0
    alpha = 0.8
    for i in d:
        cur = sum(i) / float(smooth_num)
        cache = cache * alpha + (1 - alpha) * cur
        success_rate_new.append(cache)
    numbers['success_rate'] = success_rate_new
    return success_rate_new


def load_performance_file_BBQ(path):
    """ load the performance score (.json) file """

    fin = open(path)
    success, epoch = [], []
    for lines in fin:
        suc, update = lines.strip().split('\t')
        success.append(int(suc) * 2)
        update = int(update) / 32
        epoch.append(update)

    c = 0
    eepoch = []
    for e in epoch:
        c += e / 100 if e / 100 != 0 else 1
        eepoch.append(c)

    success_rate = []
    for i in zip(zip(eepoch[:-1], eepoch[1:]), zip(success[:-1], success[1:])):
        success_rate.extend(np.linspace(i[1][0] / 100., i[1][1] / 100., abs(i[0][0] - i[0][1])).tolist())

    smooth_num = 1
    d = [success_rate[i * smooth_num:i * smooth_num + smooth_num] for i in range(int(len(success_rate) / smooth_num))]

    success_rate_new = []
    cache = 0
    alpha = 0.8
    for i in d:
        cur = sum(i) / float(smooth_num)
        cache = cache * alpha + (1 - alpha) * cur
        success_rate_new.append(cache)
    return success_rate_new


def draw_learning_curve(numbers):
    """ draw the learning curve """

    plt.xlabel('Simulation Epoch')
    plt.ylabel('Success Rate')
    plt.title('Learning Curve')
    plt.grid(True)

    plt.plot(numbers['x'], numbers['success_rate'], 'r', lw=1)
    plt.show()


def main(params):
    cmd = params['cmd']
    colors = ['#2f79c0', '#278b18', '#ff5186', '#8660a4', '#D49E0F', '#a8d40f']
    global_idx = 1500

    if cmd == 0:
        numbers = load_performance_file(params['result_file'])
        draw_learning_curve(numbers)
    elif cmd == 1:
        # draw_list = range(1,2)
        # draw_list = [1,2,3]
        # dqn_datapoint = []
        # for i in draw_list:
        #     dqn_datapoint.append(read_performance_records('dqn_plan1/noclipped_reward_A5k_U5k_step2_run%d/agt_9_performance_records.json' % (i)))

        # # hdqn = read_performance_records('agt_10_performance_records.json')

        # min_len = min(len(i) for i in dqn_datapoint)
        # print [len(i) for i in dqn_datapoint]
        # data = np.asarray([i[0:min_len] for i in dqn_datapoint])
        # # sns.tsplot(data, err_style="boot_traces", n_boot=500)
        # # sns.plt.show()

        # mean = np.mean(data,axis=0)
        # var = np.std(data,axis=0)

        # # plt.plot(range(mean.shape[0]),[0.33]*mean.shape[0],colors[3],label='Rule Agent')

        # idx = min(mean.shape[0], global_idx)
        # # l1, = plt.plot(range(idx),mean[0:idx],colors[0])
        # l1, = plt.plot(range(mean.shape[0]),mean, colors[0],label='Plan 1 step (True dynamic)', linewidth=linewidth)

        # plt.fill_between(range(mean.shape[0]), mean+var/2, mean-var/2, facecolor=colors[0], alpha=0.2)

        # hdqn_datapoint = []
        # draw_list = range(1,9)
        # for i in draw_list:
        #     hdqn_datapoint.append(read_performance_records('dqn_10/noclipped_reward_A5k_U2k_sb5k_10_noplanning_run%d/agt_9_performance_records.json' % (i)))
        # min_len = min(len(i) for i in hdqn_datapoint)
        # data = np.asarray([i[0:min_len] for i in hdqn_datapoint])
        # # sns.tsplot(data2, err_style="boot_traces", n_boot=500)
        # # sns.plt.show()

        # mean = np.mean(data,axis=0)
        # var = np.std(data,axis=0)
        # idx = min(mean.shape[0], global_idx)
        # l1, = plt.plot(range(idx),mean[0:idx],colors[2],linewidth=linewidth, label='RL Agent 10')

        # plt.fill_between(range(mean.shape[0]), mean+var/2, mean-var/2, facecolor=colors[2], alpha=0.2, label='Plan 4 step with true dynamic')

        # numbers=load_performance_file_ma('rl_agent_1_all/agt_9_performance_records.json')
        # plt.plot(range(len(numbers)), numbers, 'k', lw=1, label='Baseline')

        BBQ_datapoint = []
        draw_list = list(range(1,5))
        for i in draw_list:
            BBQ_datapoint.append(read_performance_records(
                '/Users/qianlong/Desktop/dqn/learning_rate_d4_AgentDQN_dqn1_T22_lr0.001_RFS44_RFF-22.0_RFNCY-1_RFIRS-1_mls0_gamma0.9_epsilon0.1_awd0_crs0_RID%d_datalabel_mGPU0_1499.p' % i))
        min_len = min(len(i) for i in BBQ_datapoint)
        # min_len = 1000
        print([len(i) for i in BBQ_datapoint])
        data = np.asarray([i[0:min_len] for i in BBQ_datapoint])
        mean = np.mean(data, axis=0)
        print(mean)
        var = np.std(data, axis=0)
        idx = min(mean.shape[0], global_idx)
        l1, = plt.plot(range(idx), mean[0:idx], colors[0], label='DQN Agent 44', linewidth=linewidth)
        plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=colors[0], alpha=0.2)

        BBQ_datapoint = []
        draw_list=range(1,5)
        for i in draw_list:
            BBQ_datapoint.append(read_performance_records('/Users/qianlong/Desktop/dqn/learning_rate_d4_AgentDQN_dqn1_T22_lr0.001_RFS44_RFF-22.0_RFNCY-1_RFIRS-1_mls0_gamma0.9_epsilon0.1_awd0_crs0_RID%d_datalabel_mGPU0_DoubleDQN_1499.p'%i))
        min_len = min(len(i) for i in BBQ_datapoint)
        print([len(i) for i in BBQ_datapoint])
        data = np.asarray([i[0:min_len] for i in BBQ_datapoint])
        mean = np.mean(data,axis=0)
        var = np.std(data,axis=0)
        l2, = plt.plot(range(mean.shape[0]),mean,colors[1], label='DoubleDQN Agent 44', linewidth=linewidth)
        plt.fill_between(range(mean.shape[0]), mean+var/2, mean-var/2, facecolor=colors[1], alpha=0.2)

        BBQ_datapoint = []
        draw_list=range(1,5)
        for i in draw_list:
            BBQ_datapoint.append(read_performance_records('/Users/qianlong/Desktop/dqn/learning_rate_d4_AgentDQN_dqn1_T22_lr0.001_RFS55.0_RFF-22.0_RFNCY-1_RFIRS-1_mls0_gamma0.9_epsilon0.1_awd0_crs0_RID%d_datalabel_mGPU0_DoubleDQN_1499.p'%i))
        min_len = min(len(i) for i in BBQ_datapoint)
        print([len(i) for i in BBQ_datapoint])
        data = np.asarray([i[0:min_len] for i in BBQ_datapoint])
        mean = np.mean(data,axis=0)
        var = np.std(data,axis=0)
        l2, = plt.plot(range(mean.shape[0]),mean,colors[2], label='DoubleDQN Agent 55', linewidth=linewidth)
        plt.fill_between(range(mean.shape[0]), mean+var/2, mean-var/2, facecolor=colors[2], alpha=0.2)

        # BBQ_datapoint = []
        # # draw_list = range(2,10)
        # draw_list = range(1,7)
        #
        # for i in draw_list:
        #     BBQ_datapoint.append(read_performance_records('dqn_step4/runMay7_%i/agt_9_performance_records.json' % (i)))
        # min_len = min(len(i) for i in BBQ_datapoint)
        # data = np.asarray([i[0:min_len] for i in BBQ_datapoint])
        # # sns.tsplot(data2, err_style="boot_traces", n_boot=500)
        # # sns.plt.show()
        #
        # mean = np.mean(data,axis=0)
        # var = np.std(data,axis=0)
        #
        # l3, = plt.plot(range(mean.shape[0]),mean,'slategrey', label='DDQ(5)', color=colors[1], linewidth=linewidth)
        #
        # plt.fill_between(range(mean.shape[0]), mean+var/2, mean-var/2, facecolor=colors[1], alpha=0.2)
        #
        #
        # BBQ_datapoint = []
        # draw_list = range(1,7)
        #
        # for i in draw_list:
        #     BBQ_datapoint.append(read_performance_records('dqn_step9/run%i/agt_9_performance_records.json' % (i)))
        # min_len = min(len(i) for i in BBQ_datapoint)
        # data = np.asarray([i[0:min_len] for i in BBQ_datapoint])
        # # sns.tsplot(data2, err_style="boot_traces", n_boot=500)
        # # sns.plt.show()
        # print([len(i) for i in BBQ_datapoint])
        #
        # mean = np.mean(data,axis=0)
        # var = np.std(data,axis=0)
        #
        # l5, = plt.plot(range(mean.shape[0]),mean,'navy', label='DDQ(10)', linewidth=linewidth)
        #
        # plt.fill_between(range(mean.shape[0]), mean+var/2, mean-var/2, facecolor='navy', alpha=0.2)
        #
        #
        #
        # BBQ_datapoint = []
        # draw_list = range(1,6)
        #
        # for i in draw_list:
        #     BBQ_datapoint.append(read_performance_records('dqn_step19/run%i/agt_9_performance_records.json' % (i)))
        # min_len = min(len(i) for i in BBQ_datapoint)
        # print([len(i) for i in BBQ_datapoint])
        # data = np.asarray([i[0:min_len] for i in BBQ_datapoint])
        # # sns.tsplot(data2, err_style="boot_traces", n_boot=500)
        # # sns.plt.show()
        #
        # mean = np.mean(data,axis=0)
        # var = np.std(data,axis=0)
        #
        # l4, = plt.plot(range(mean.shape[0]),mean,colors[3], label='DDQ(20)', linewidth=linewidth)
        #
        # plt.fill_between(range(mean.shape[0]), mean+var/2, mean-var/2, facecolor=colors[3], alpha=0.2)

        plt.grid(True)
        plt.ylabel('Success Rate')
        plt.xlabel('Simulation Epoch')
        plt.hlines(0.23, 0, min_len, label="Rule Agent", linewidth=linewidth, colors=colors[3])
        plt.hlines(0.06, 0, min_len, label="Random Agent", linewidth=linewidth, colors=colors[4])

        # plt.legend(['RL Agent', 'RL Planner-ground (k=9)', 'RL Planner (k=9)'], loc=4)
        plt.xlim([0, min_len])
        # plt.legend(loc=4)
        plt.legend(loc='center right')

        plt.ylim([0, 0.7])
        plt.savefig('/Users/qianlong/Desktop/dqn/learning_curve.pdf')
        plt.show()
        # plt.savefig('sim_exp_2_tanh.pdf')

        # plt.plot(range(len(hdqn)),hdqn)
        # # plt.plot(range(idx),dqn[0:idx])
        # plt.plot(range(len(dqn)),dqn)
        # plt.xlabel('Simulation Epoch')
        # plt.ylabel('Success Rate')
        # plt.legend(['HDQN','DQN'])
        # plt.xlim([0,150])
        # plt.ylim([0,0.8])
        # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cmd', dest='cmd', type=int, default=1, help='cmd')

    # parser.add_argument('--result_file', dest='result_file', type=str, default='agt_10_performance_records.json', help='path to the result file')
    parser.add_argument('--result_file', dest='result_file', type=str,
                        default='/Users/qianlong/Desktop/learning_rate_d4_e_agent1_dqn1_T22_lr0.001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma1.0_epsilon0.1_1499.p',
                        help='path to the result file')

    args = parser.parse_args()
    params = vars(args)
    print(json.dumps(params, indent=2))

    main(params)