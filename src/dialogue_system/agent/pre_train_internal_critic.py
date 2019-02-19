# -*- coding: utf8 -*-
"""
预训练一个internal critic给agent_with_goal使用，训练的模型是输入对话状态以及当前时刻的goal，判断当前goal是否结束，以及如果结束是将这个
goal提供给user当做疾病预测的结果，还是返回给master重新选择。
"""

import torch
import numpy as np
import sys, os
import pickle
import copy
from tqdm import tqdm
from sklearn.metrics import  accuracy_score, confusion_matrix
from src.dialogue_system.policy_learning.internal_critic import InternalCritic
# from src.dialogue_system.policy_learning.internal_critic import MultiClassifier as InternalCritic
import random


slot_set = pickle.load(file=open('./../../data/real_world/slot_set.p', "rb"))
print(len(slot_set))
del slot_set['disease']
disease_symptom = pickle.load(file=open('./../../data/real_world/disease_symptom.p', "rb"))
goal_set = pickle.load(file=open('./../../data/real_world/goal_set.p', "rb"))

# symptom distribution by diseases.
temp_slot_set = copy.deepcopy(slot_set)
disease_to_symptom_dist = {}
total_count = np.zeros(len(temp_slot_set))
for disease, v in disease_symptom.items():
    dist = np.zeros(len(temp_slot_set))
    for symptom, count in v['symptom'].items():
        dist[temp_slot_set[symptom]] = count
        total_count[temp_slot_set[symptom]] += count
    disease_to_symptom_dist[disease] = dist

goal_embed_value = [0]* len(disease_symptom)
for disease in disease_to_symptom_dist.keys():
    disease_to_symptom_dist[disease] = disease_to_symptom_dist[disease] / total_count
    goal_embed_value[disease_symptom[disease]['index']] = list(disease_to_symptom_dist[disease])


def get_batches(goal_list):
    data_ex, data_im, data_both, label_list, fake_label_list = [], [], [], [], []
    for goal in goal_list:
        ############
        # one hot
        ##########
        symptom_rep_ex = np.zeros((len(slot_set.keys()), 3))
        symptom_rep_im = np.zeros((len(slot_set.keys()), 3))
        symptom_rep_ex_im = np.zeros((len(slot_set.keys()), 3))
        # explicit
        for symptom, value in goal["goal"]["explicit_inform_slots"].items():
            if value == True:
                symptom_rep_ex[slot_set[symptom]][0] = 1
                symptom_rep_ex_im[slot_set[symptom]][0] = 1
            elif value == False:
                symptom_rep_ex[slot_set[symptom]][1] = 1
                symptom_rep_ex_im[slot_set[symptom]][1] = 1

            elif value == 'UNK':
                symptom_rep_ex[slot_set[symptom]][2] = 1
                symptom_rep_ex_im[slot_set[symptom]][2] = 1

        # implicit
        for symptom, value in goal["goal"]["implicit_inform_slots"].items():
            if value == True:
                symptom_rep_im[slot_set[symptom]][0] = 1
                symptom_rep_ex_im[slot_set[symptom]][0] = 1
            elif value == False:
                symptom_rep_ex_im[slot_set[symptom]][1] = 1
                symptom_rep_im[slot_set[symptom]][1] = 1
            elif value == 'UNK':
                symptom_rep_ex_im[slot_set[symptom]][2] = 1
                symptom_rep_im[slot_set[symptom]][2] = 1
        symptom_rep_ex = np.reshape(symptom_rep_ex, (3 * len(slot_set.keys())))
        symptom_rep_im = np.reshape(symptom_rep_im, (3 * len(slot_set.keys())))
        symptom_rep_ex_im = np.reshape(symptom_rep_ex_im, (3 * len(slot_set.keys())))

        ###############
        # no one hot
        ############
        # symptom_rep_ex = np.zeros(len(slot_set.keys()))
        # symptom_rep_im = np.zeros(len(slot_set.keys()))
        # symptom_rep_ex_im = np.zeros(len(slot_set.keys()))
        # # explicit
        # for symptom, value in goal["goal"]["explicit_inform_slots"].items():
        #     if value == True:
        #         symptom_rep_ex[slot_set[symptom]] = 1
        #         symptom_rep_ex_im[slot_set[symptom]] = 1
        #     elif value == False:
        #         symptom_rep_ex[slot_set[symptom]] = -1
        #         symptom_rep_ex_im[slot_set[symptom]] = -1
        #
        #     elif value == 'UNK':
        #         symptom_rep_ex[slot_set[symptom]] = 2
        #         symptom_rep_ex_im[slot_set[symptom]] = 2
        #
        # # implicit
        # for symptom, value in goal["goal"]["implicit_inform_slots"].items():
        #     if value == True:
        #         symptom_rep_im[slot_set[symptom]] = 1
        #         symptom_rep_ex_im[slot_set[symptom]] = 1
        #     elif value == False:
        #         symptom_rep_ex_im[slot_set[symptom]] = -1
        #         symptom_rep_im[slot_set[symptom]] = -1
        #     elif value == 'UNK':
        #         symptom_rep_ex_im[slot_set[symptom]] = 2
        #         symptom_rep_im[slot_set[symptom]] = 2

        data_ex.append(symptom_rep_ex)
        data_im.append(symptom_rep_im)
        data_both.append(symptom_rep_ex_im)
        disease_index = disease_symptom[goal['disease_tag']]['index']
        label_list.append(disease_index)
        index_list = [i for i in range(4)]
        index_list.pop(disease_index)
        fake_label_list.append(index_list)
    return data_im, data_ex, data_both, label_list, fake_label_list




batch_size = 32
lr = 0.001
epoch_num = 1000

params = {}
params['dqn_learning_rate'] = 0.001
params['multi_GPUs'] = False
model = InternalCritic(input_size=len(slot_set) * 3 + 4, hidden_size=50, output_size=len(slot_set), goal_num=4,
                       goal_embedding_value=goal_embed_value,
                       slot_set=slot_set,
                       parameter=params)

# model = InternalCritic(input_size=len(slot_set) * 3, hidden_size=100, output_size=4, goal_num=4,
#                        goal_embedding_value=goal_embed_value,
#                        slot_set=slot_set,
#                        parameter=params)

data_im, data_ex, data_both, label_list, fake_label_list = get_batches(goal_set['train'])


batch_num = int(len(data_im) / batch_size)
for epoch_index in range(epoch_num):
    total_loss_positive = 0
    total_loss_negative = 0
    for i in range(batch_num - 1):
        #正样本
        batch = data_ex[i * batch_size:(i + 1) * batch_size]
        label = label_list[i * batch_size:(i + 1) * batch_size]

        #负样本
        batch_negative = [[one_data] * 3 for one_data in batch]
        batch_negative = torch.Tensor(batch_negative)
        batch_negative = batch_negative.view(batch_size*3, -1)
        fake_label = fake_label_list[i * batch_size:(i + 1) * batch_size]
        fake_label = torch.Tensor(fake_label).view(-1)
        train_res = model.train(batch, label, batch_negative,fake_label)
        total_loss_positive += train_res['positive_similarity']
        total_loss_negative += train_res['negative_similarity']
    print(epoch_index, 'positive', total_loss_positive/batch_num, 'negative', total_loss_negative/batch_num)


# model.restore_model('pre_trained_internal_critic_dropout_one_hot.pkl')

# model.critic.eval()
print('validate')
data_im, data_ex, data_both, label_list, fake_label_list = get_batches(goal_set['validate'])
predict = []
index = 0
for one_data in data_ex:
    batch = [one_data] * 4
    label = [0, 1,2,3]
    # similarity = model.get_similarity(batch, label)[0]
    similarity = model.get_similarity(batch, label)
    print(label_list[index], np.argmax(similarity), min(similarity), max(similarity),similarity)
    predict.append(int(np.argmax(similarity)))
    index += 1

res = confusion_matrix(label_list, predict)
accu = accuracy_score(label_list, predict)
print('validate')
print(res)
print(accu)


model.critic.eval()
predict = []
for one_data in data_ex:
    batch = [one_data] * 4
    label = [0, 1,2,3]
    # similarity = model.get_similarity(batch, label)[0]
    similarity = model.get_similarity(batch, label)
    predict.append(int(np.argmax(similarity)))

res = confusion_matrix(label_list, predict)
accu = accuracy_score(label_list, predict)
print('validate')
print(res)
print(accu)
model.save_model('pre_trained_internal_critic_dropout_one_hot.pkl')



model.critic.train()
data_im, data_ex, data_both, label_list, fake_label_list = get_batches(goal_set['test'])
predict = []
index = 0
for one_data in data_ex:
    batch = [one_data] * 4
    label = [0, 1,2,3]
    # similarity = model.get_similarity(batch, label)[0]
    similarity = model.get_similarity(batch, label)
    print(label_list[index], np.argmax(similarity), min(similarity), max(similarity), similarity)
    predict.append(int(np.argmax(similarity)))
    index += 1

res = confusion_matrix(label_list, predict)
accu = accuracy_score(label_list, predict)
print('test')
print(res)
print(accu)


model.critic.eval()
predict = []
for one_data in data_ex:
    batch = [one_data] * 4
    label = [0, 1,2,3]
    # similarity = model.get_similarity(batch, label)[0]
    similarity = model.get_similarity(batch, label)
    predict.append(int(np.argmax(similarity)))

res = confusion_matrix(label_list, predict)
accu = accuracy_score(label_list, predict)
print('test')
print(res)
print(accu)