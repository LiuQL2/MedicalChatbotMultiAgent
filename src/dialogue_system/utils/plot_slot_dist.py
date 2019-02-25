# -*- coding:utf-8 -*-
"""
用来画症状的分布直方图。
"""

import matplotlib.pyplot as plt
import pickle


class DistributionPloter(object):
    def __init__(self, goal_set_file):
        self.goal_set = pickle.load(open(goal_set_file, 'rb'))
        self.symptom2id, self.id2disease, self.symptom_dist_by_disease = self.__distribution__()
        self.disease_to_english = {
            '小儿腹泻': 'Infantile diarrhea',
            '小儿支气管炎': 'Children’s bronchitis',
            '小儿消化不良': 'Children functional dyspepsia',
            '上呼吸道感染': 'Upper respiratory infection'
        }

    def __distribution__(self):
        symptom2id = dict()
        id2disease = dict()
        disease2id = dict()

        for goal in self.goal_set['train'] + self.goal_set['test'] + self.goal_set['validate']:
            id = len(disease2id)
            disease2id.setdefault(goal['disease_tag'], id)
            for symptom in goal['goal']['explicit_inform_slots'].keys():
                id = len(symptom2id)
                symptom2id.setdefault(symptom, id)
            for symptom in goal['goal']['implicit_inform_slots'].keys():
                id = len(symptom2id)
                symptom2id.setdefault(symptom, id)

        symptom_dist_by_disease = {}
        for goal in self.goal_set['train'] + self.goal_set['test'] + self.goal_set['validate']:
            symptom_dist_by_disease.setdefault(goal['disease_tag'], [0] * len(symptom2id))
            for symptom in goal['goal']['explicit_inform_slots'].keys():
                symptom_dist_by_disease[goal['disease_tag']][symptom2id[symptom]] += 1
            for symptom in goal['goal']['implicit_inform_slots'].keys():
                symptom_dist_by_disease[goal['disease_tag']][symptom2id[symptom]] += 1

        for key, v in disease2id.items():
            id2disease[v] = key
            print(key, v)
        return symptom2id, id2disease,  symptom_dist_by_disease

    def plot(self, save_file):
        colors = ['#2f79c0', '#278b18', '#ff5186', '#8660a4', '#D49E0F', '#a8d40f']
        print(self.symptom2id)
        bottom = [0]* len(self.symptom2id)

        disease = self.id2disease[0]
        symptom_dist = self.symptom_dist_by_disease[disease]
        plt.bar(range(len(symptom_dist)), symptom_dist, label=self.disease_to_english[disease], fc=colors[0])
        for index in range(1, len(self.id2disease)):
            disease = self.id2disease[index]
            symptom_dist = self.symptom_dist_by_disease[disease]
            print(disease,len(symptom_dist),  symptom_dist)
            plt.bar(range(len(symptom_dist)), symptom_dist, bottom=bottom, label=self.disease_to_english[disease], fc=colors[index])
            # plt.bar(range(len(symptom_dist)), symptom_dist, bottom=bottom, label=disease, tick_label=self.symptom2id.keys(), fc=colors[index])
            bottom = [bottom[i] + symptom_dist[i] for i in range(len(self.symptom2id))]
        plt.legend()
        plt.savefig(save_file)
        plt.show()


if __name__ == '__main__':
    goal_set_file = './../../data/real_world/goal_set.p'
    ploter = DistributionPloter(goal_set_file)
    save_file = './../../data/real_world/symptom_dist.pdf'
    ploter.plot(save_file)