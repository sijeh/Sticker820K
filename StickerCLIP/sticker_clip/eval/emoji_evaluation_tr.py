# -*- coding: utf-8 -*-
'''
This script computes the recall scores given the ground-truth annotations and predictions.
'''

import json
import sys
import os
import string
import numpy as np
import time

NUM_K = 10


def read_submission(submit_path):

    submission_dict = {}
    with open(submit_path, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            try:
                pred_obj = json.loads(line)
            except:
                raise Exception('Cannot parse this line into json object: {}'.format(line))
            if "image_id" not in pred_obj:
                raise Exception('There exists one line not containing image_id: {}'.format(line))
            qid = pred_obj["image_id"]
            if "text_ids" not in pred_obj:
                raise Exception('There exists one line not containing the predicted text_ids: {}'.format(line))
            text_ids = pred_obj["text_ids"]
            if not isinstance(text_ids, list):
                raise Exception('The text_ids field of image_id {} is not a list, please check your schema'.format(qid))
            
            submission_dict[qid] = text_ids # here we save the list of product ids

    return submission_dict


def dump_2_json(info, path):
    with open(path, 'w', encoding="utf-8") as output_json_file:
        json.dump(info, output_json_file)


def report_error_msg(detail, showMsg, out_p):
    error_dict=dict()
    error_dict['errorDetail']=detail
    error_dict['errorMsg']=showMsg
    error_dict['score']=0
    error_dict['scoreJson']={}
    error_dict['success']=False
    dump_2_json(error_dict,out_p)

def report_score(r1, r5, r10, out_p):
    result = dict()
    result['success']=True
    mean_recall = (r1 + r5 + r10) / 3.0
    result['score'] = mean_recall * 100
    result['scoreJson'] = {'score': mean_recall * 100, 'mean_recall': mean_recall * 100, 'r1': r1 * 100, 'r5': r5 * 100, 'r10': r10 * 100}
    dump_2_json(result,out_p)



if __name__ == '__main__':
    submit_path = sys.argv[1]
    output_path = sys.argv[2]

    print("Read user submit file from %s" % submit_path)

    predictions = read_submission(submit_path)

        # compute score for each text
    r1_stat, r5_stat, r10_stat = 0, 0, 0

    count = 0
    for qid in predictions.keys():
        if any([qid in predictions[qid][:1]]):
            r1_stat += 1
        if any([qid in predictions[qid][:5]]):
            r5_stat += 1
        if any([qid in predictions[qid][:10]]):
            r10_stat += 1
        count += 1

    r1, r5, r10 = r1_stat * 1.0 / count, r5_stat * 1.0 / count, r10_stat * 1.0 / count

    report_score(r1, r5, r10, output_path)
    print("The evaluation finished successfully.")