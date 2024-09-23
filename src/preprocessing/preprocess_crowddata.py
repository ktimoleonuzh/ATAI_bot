# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 20:05:03 2022

@author: Nadia Timoleon
"""
import pandas as pd
import numpy as np
from src.utils import load_data_config, load_pickle
from statsmodels.stats import inter_rater as irr

data_config = load_data_config()
crowd_data = load_pickle(data_config['paths']['crowd_data'])

def track_fast_deceivers(crowd_data):
    min_WorkTime = 10
    dismiss = crowd_data[(crowd_data['WorkTimeInSeconds'] <= min_WorkTime)]
    fast_deceivers = [work_id for work_id in dismiss['WorkerId']]
    return fast_deceivers

def track_unreliable(crowd_data):
    dismiss = crowd_data[~(crowd_data['LifetimeApprovalRate'].isin(['70%', '80%', '85%', '98%', '99%']))]
    unreliable = [work_id for work_id in dismiss['WorkerId']]
    return unreliable

def clean_data(crowd_data):
    fast_deceivers = track_fast_deceivers(crowd_data)
    unreliable = track_unreliable(crowd_data)
    malicious_workers = set(fast_deceivers+unreliable)
    clean_crowd_data = crowd_data[~crowd_data['WorkerId'].isin(malicious_workers)]
    return clean_crowd_data

def aggregate_votes(clean_crowd_data):
    grp = clean_crowd_data.groupby(['HITId'])
    dict_list = list()
    aggr_ans_dict = dict()
    fleiss_kappa_dict = fleiss_kapa(clean_crowd_data)
    for key, group in grp:
        subj = np.unique(group['Input1ID'])[0]
        pred = np.unique(group['Input2ID'])[0]
        obj = np.unique(group['Input3ID'])[0]
        triple = (subj, pred, obj)
        correct_votes = group[group['AnswerID'] == 1]
        incorrect_votes = group[group['AnswerID'] == 2]
        HITTypeId = np.unique(group['HITTypeId'])[0]
        fk = fleiss_kappa_dict[HITTypeId]
        if len(correct_votes) > len(incorrect_votes):
            answer_id = 1
        elif len(correct_votes) < len(incorrect_votes):
            answer_id = 2
        correction, fix_pos = correct_triple(group, triple)
        group_dict = {
            "HITId": key,
            "AnswerId": answer_id,
            "Triple": triple,
            "Distribution": [len(correct_votes), len(incorrect_votes)],
            "HITTypeId": HITTypeId,
            "FleissKappa": fk,
            "Correction": correction,
            "FixPosition": fix_pos
        }
        dict_list.append(group_dict)
    aggr_ans_dict['crowddata'] = dict_list
    return aggr_ans_dict

def correct_triple(group, triple):
    corr = group.loc[:,['FixPosition', 'FixValue']].dropna()
    if corr.empty:
        correction = None
        fix_pos = None
    else:
        correction = corr.iloc[0,:].tolist()
        fix_pos = correction[0]
        fix_val = correction[1]
        if fix_pos == 'Subject':
            correction = (fix_val, triple[1], triple[2])
        elif fix_pos == 'Predicate':
            correction = (triple[0], fix_val, triple[2])
        elif fix_pos == 'Object':
            correction = (triple[0], triple[1], fix_val)
        else:
            correction = None
    return correction, fix_pos

def fleiss_kapa(clean_crowd_data):    
    fleiss_kappa_dict = dict()
    grp = clean_crowd_data.groupby(['HITTypeId'])
    for batch_key, batch in grp:
        ans_arr = [[0]*3 for i in range(batch.groupby('HITId').ngroups)]
        for idx, (_, task) in enumerate(batch.groupby('HITId')):
            ans_arr[idx] = [ans_id for ans_id in task['AnswerID']]
        data, cats = irr.aggregate_raters(ans_arr)
        fleiss_kappa_dict[batch_key] = irr.fleiss_kappa(data, method='fleiss')
    return fleiss_kappa_dict

def preprocess(crowd_data):
    clean_crowd_data = clean_data(crowd_data)
    aggr_ans_dict = aggregate_votes(clean_crowd_data)
    return aggr_ans_dict
