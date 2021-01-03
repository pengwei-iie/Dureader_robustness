""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def f1_score(prediction, ground_truth):
    prediction_tokens = prediction[0]
    ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return ((prediction) == (ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    # for ground_truth in ground_truths:
    score = metric_fn(prediction, ground_truths)
    scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for i in range(len(dataset)):
        total += 1
        ground_truths = dataset[i]['answers']
        prediction = predictions[i]['answers']
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction[0], ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


if __name__ == '__main__':
    expected_version = '2020'
    parser = argparse.ArgumentParser(
        description='Evaluation for Dureader ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    dataset = []
    predictions = []
    with open(args.dataset_file, 'r', encoding='utf-8') as dataset_file:
        for line in dataset_file.readlines():
            dataset_json = json.loads(line)
            dataset.append(dataset_json)
    with open(args.prediction_file, 'r', encoding='utf-8') as prediction_file:
        for line in prediction_file.readlines():
            dataset_json = json.loads(line)
            predictions.append(dataset_json)
    print(json.dumps(evaluate(dataset, predictions)))
