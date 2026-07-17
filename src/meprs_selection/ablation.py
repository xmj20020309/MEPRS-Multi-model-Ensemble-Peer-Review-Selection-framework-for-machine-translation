import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np


model_dict = {
    "gpt-3.5-turbo":"G35",
    "gpt-4o":"G4o",
    "claude-3-opus":"C3",
    "claude-3.5-sonnet":"C35",
    "gemini-pro":"GP"
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "dataset"


def process_sub_lists_avg(sub_lists):
    num_candidates = len(sub_lists[0])
    num_items = len(sub_lists[0][0])
    summed_result = [[0.0 for _ in range(num_items)] for _ in range(num_candidates)]
    for sub in sub_lists:

        for i in range(num_candidates):
            for j in range(num_items):
                summed_result[i][j] += sub[i][j]


    return summed_result

def read_scores(path):
    values = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                values.append(float(line.split()[0]))
    return values

def dataset_file(args, filename):
    return Path(args.data_dir) / f"{args.src_lan}-{args.tgt_lan}-new" / filename

def output_results(forward_scores, backward_scores):
    max_backward_index = []
    for i in range(len(backward_scores[0])):
        max_val = max(item[i] for item in backward_scores)
        index = [item[i] for item in backward_scores].index(max_val)
        max_backward_index.append(index)

    final_scores = []
    for i in range(len(forward_scores[0])):
        final_scores.append(forward_scores[max_backward_index[i]][i])

    print(round((sum(final_scores) / len(final_scores)) * 100, 2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Command-line script to use')
    parser.add_argument('--src_lan', type=str, default='', help='source language')
    parser.add_argument('--tgt_lan', type=str, default='', help='target language')
    parser.add_argument('--forward', type=str, default='', help='forward translation method')
    parser.add_argument('--metric', type=str, default='', help='selection metric')
    parser.add_argument('--models', nargs='+', help='LLMs')
    parser.add_argument('--dimensions', nargs='+', help='dimensions')
    parser.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR), help='dataset directory')

    args = parser.parse_args()

    print(args.src_lan, args.tgt_lan)

    forward_scores = []
    for model in args.models:
        score_file_forward = dataset_file(args, model_dict[model] + "_" + args.forward + "." + args.metric)
        values = read_scores(score_file_forward)
        forward_scores.append(values)
    max_forward_index = []
    for i in range(len(forward_scores[0])):
        max_val = max(item[i] for item in forward_scores)
        index = [item[i] for item in forward_scores].index(max_val)
        max_forward_index.append(index)
    for dimension in args.dimensions:
        backward_scores_list = []

        for model_eval in args.models:
            backward_scores = []
            for model_predict in args.models:
                score_file_backward = dataset_file(
                    args, model_dict[model_predict] + "_" + args.forward + "_" + model_dict[model_eval] + "." + dimension
                )
                values = read_scores(score_file_backward)
                backward_scores.append(values)
            backward_scores_list.append(backward_scores)
        print(dimension, end=" ")
        backward_scores_avg = process_sub_lists_avg(backward_scores_list)
        output_results(forward_scores, backward_scores_avg)

