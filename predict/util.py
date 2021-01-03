import json
import args
import torch
import pickle
from tqdm import tqdm


def creat_examples(filename_1, result):
    examples = []
    samples = json.load(open(filename_1, 'r', encoding='utf-8'))['data'][0]['paragraphs']
    for source in samples:
        if len(source['qas']) == 0:
            continue
        for i in range(len(source['qas'])):
            clean_doc = source['context']
            source['doc_tokens'] = []
            source['doc_tokens'].append({'doc_tokens': clean_doc})
            source['question_id'] = source['qas'][i]['id']
            source['question'] = source['qas'][i]['question']

            example = ({
                'id': source['question_id'],
                'question_text': source['question'].strip(),
                'doc_tokens': source['doc_tokens'],
                'answers': source['qas'][0]['answers'][0]['text']})
            examples.append(example)
    print("{} questions in total".format(len(examples)))
    with open(result, 'wb') as fw:
        pickle.dump(examples, fw)


if __name__ == "__main__":
    creat_examples(filename_1='../data_2020/devset/dev.json',
                   result=args.predict_example_files)
