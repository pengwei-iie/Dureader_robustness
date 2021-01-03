import sys
import json
from collections import Counter
import jieba

def precision_recall_f1(prediction, ground_truth):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def recall(prediction, ground_truth):
    """
    This function calculates and returns the recall
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of recall
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[1]


def f1_score(prediction, ground_truth):
    """
    This function calculates and returns the f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of f1
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[2]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        metric_fn: metric function pointer which calculates scores according to corresponding logic.
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    scores_for_ground_truths = []
    # for ground_truth in ground_truths:
    score = metric_fn(prediction, ground_truths)
    scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def find_best_question_match(doc, question, with_score=False):
    """
    For each document, find the paragraph that matches best to the question.
    Args:
        doc: The document object.
        question: The question tokens.
        with_score: If True then the match score will be returned,
            otherwise False.
    Returns:
        The index of the best match paragraph, if with_score=False,
        otherwise returns a tuple of the index of the best match paragraph
        and the match score of that paragraph.
    """
    most_related_para = -1
    max_related_score = 0
    most_related_para_len = 0
    for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
        if len(question) > 0:
            related_score = metric_max_over_ground_truths(recall,
                    para_tokens,
                    question)
        else:
            related_score = 0

        if related_score > max_related_score \
                or (related_score == max_related_score \
                and len(para_tokens) < most_related_para_len):
            most_related_para = p_idx
            max_related_score = related_score
            most_related_para_len = len(para_tokens)
    if most_related_para == -1:
        most_related_para = 0
    if with_score:
        return most_related_para, max_related_score
    return most_related_para


def find_fake_answer(sample):
    """
    For each document, finds the most related paragraph based on recall,
    then finds a span that maximize the f1_score compared with the gold answers
    and uses this span as a fake answer span
    Args:
        sample: a sample in the dataset
    Returns:
        None
    Raises:
        None
    """
    # sample['answer_docs'] = []
    sample['answer_spans'] = []
    sample['fake_answers'] = []
    sample['match_scores'] = []
    sample['answers'] = sample['qas'][0]['answers'][0]['text']
    sample['question_id'] = sample['qas'][0]['id']
    sample['question'] = sample['qas'][0]['question']
    sample['segmented_answers'] = jieba.lcut(sample['qas'][0]['answers'][0]['text'])

    best_match_score = 0
    best_match_d_idx, best_match_span = -1, [-1, -1]
    best_fake_answer = None

    answer_tokens = set([token for token in sample['segmented_answers']])

    most_related_para_tokens = jieba.lcut(sample['context'])
    sample['doc_tokens'] = most_related_para_tokens
    for start_tidx in range(len(most_related_para_tokens)):
        if most_related_para_tokens[start_tidx] not in answer_tokens:
            continue
        for end_tidx in range(len(most_related_para_tokens) - 1, start_tidx - 1, -1):
            span_tokens = most_related_para_tokens[start_tidx: end_tidx + 1]
            if len(sample['answers']) > 0:
                match_score = metric_max_over_ground_truths(f1_score, span_tokens,
                                                            sample['segmented_answers'])
            else:
                match_score = 0
            if match_score == 0:
                break
            if match_score > best_match_score:
                # best_match_d_idx = d_idx
                best_match_span = [start_tidx, end_tidx]
                best_match_score = match_score
                best_fake_answer = ''.join(span_tokens)
    if best_match_score > 0:
        # sample['answer_docs'].append(best_match_d_idx)
        sample['answer_spans'].append(best_match_span)
        sample['fake_answers'].append(best_fake_answer)
        sample['match_scores'].append(best_match_score)


# 跑的时候用下面的，直接run.sh就可以了
# if __name__ == '__main__':
#     samples = json.load(sys.stdin)['data'][0]['paragraphs']
#     for sample in samples:
#         find_fake_answer(sample)
#         print(json.dumps(sample, ensure_ascii=False))


# debug的时候用下面这个
if __name__ == '__main__':
    with open('../data_2020/devset/dev.json', 'r', encoding='utf-8') as infile:
        samples = json.load(infile)
        for sample in samples['data'][0]['paragraphs']:
            find_fake_answer(sample)
            print(json.dumps(sample, ensure_ascii=False))
