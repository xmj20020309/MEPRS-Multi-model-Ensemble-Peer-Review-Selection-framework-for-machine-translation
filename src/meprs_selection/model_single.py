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

def process_sub_lists_zscore(sub_lists):
    num_candidates = len(sub_lists[0])
    num_items = len(sub_lists[0][0])
    summed_result = [[0.0 for _ in range(num_items)] for _ in range(num_candidates)]
    for sub in sub_lists:
        transposed = list(zip(*sub))
        normalized_transposed = []

        for col in transposed:
            col_data = list(col)
            mean = sum(col_data) / len(col_data)
            variance = sum([(x - mean) ** 2 for x in col_data]) / len(col_data)
            std_dev = variance ** 0.5

            if std_dev == 0:
                normalized_col = [0.0 for _ in col_data]
            else:
                normalized_col = [(x - mean) / std_dev for x in col_data]
            normalized_transposed.append(normalized_col)

        normalized_sub = list(zip(*normalized_transposed))
        normalized_sub = [list(row) for row in normalized_sub]
        for i in range(num_candidates):
            for j in range(num_items):
                summed_result[i][j] += normalized_sub[i][j]

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

    print(round((sum(final_scores) / len(final_scores)) * 100, 2), end=" ")

def compute(model, t, models_eval):
    forward_scores = []
    for cnt in range(t):
        score_file_forward = dataset_file(args, model_dict[model] + "_" + args.forward + "_" + str(cnt) + "." + args.metric)
        values = read_scores(score_file_forward)
        forward_scores.append(values)

    metric_scores_list = []

    for model_eval in models_eval:
        backward_scores = []
        for cnt in range(t):
            score_file_backward = dataset_file(
                args, model_dict[model] + "_" + args.forward + "_" + str(cnt) + "_" + model_dict[model_eval] + ".score"
            )
            values = read_scores(score_file_backward)
            backward_scores.append(values)
        metric_scores_list.append(backward_scores)

    backward_scores_zscore = process_sub_lists_zscore(metric_scores_list)
    output_results(forward_scores, backward_scores_zscore)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Command-line script to use')
    parser.add_argument('--src_lan', type=str, default='', help='source language')
    parser.add_argument('--tgt_lan', type=str, default='', help='target language')
    parser.add_argument('--forward', type=str, default='', help='forward translation method')
    parser.add_argument('--metric', type=str, default='', help='selection metric')
    parser.add_argument('--models', nargs='+', help='LLMs')
    parser.add_argument('--times', type=int, default='1', help='generation times')
    parser.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR), help='dataset directory')
    args = parser.parse_args()

    print(args.src_lan, args.tgt_lan)
    for model in args.models:
        print(model)
        for i in range(1,6):
            for j in range(1,6):
                compute(model, i, args.models[:j])
            print("")

