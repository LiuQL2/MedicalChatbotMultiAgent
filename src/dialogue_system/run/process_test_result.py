# -*- coding: utf8 -*-

import json
import copy
import numpy as np

result_file = '/Users/qianlong/Desktop/test_result/test_result_dqn.txt'
checkpoint_index =1000
checkpoint_list = [checkpoint_index, ] # + range(1000, 500000, 1000)

best_result = {
    'success_rate':0,
    'average_reward':0,
    'average_turn':0,
    'average_match_rate':0
}


all_run_best_results = {
    'success_rate': [],
    'average_reward': [],
    'average_turn': [],
    'average_match_rate': []
}


final_result = {}
for i in checkpoint_list:
    final_result[i] = copy.deepcopy(all_run_best_results)



def append_result(all_run_results, result):
    if result["success_rate"] > 0.1:
        for key in all_run_results.keys():
            all_run_results[key].append(result[key])

    return all_run_results

def print_mean_std(all_run_best_results):
    for key, v in all_run_best_results.items():
        v = np.array(v)
        print(key, v.mean(), v.std())


previous_epoch_index = 0

file = open(result_file, 'r')
for line in file:
    line = line.strip()

    #  一个新的run
    if '*****************' in line:
        previous_epoch_index = 0
        best_result["success_rate"] = 0.0

    line_list = line.split('    ')
    if len(line_list) == 2:
        epoch_index, result = int(line_list[0]), json.loads(line_list[1])

        # 这个时候已经超过了checkpoint，保存之前最好的结果
        if previous_epoch_index <= checkpoint_index and epoch_index > checkpoint_index:
            print(previous_epoch_index, epoch_index)
            all_run_best_results = append_result(all_run_best_results, best_result)

        if result['success_rate'] > best_result["success_rate"]: # 更新当前最好的结果
            best_result.update(result)
        previous_epoch_index = epoch_index

file.close()

print(json.dumps(all_run_best_results, indent=2))
print_mean_std(all_run_best_results)