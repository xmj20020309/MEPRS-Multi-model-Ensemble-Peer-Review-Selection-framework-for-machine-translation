import argparse
import json
from collections import Counter
from numpy import *


model_dict = {
    "gpt-3.5-turbo":"G35",
    "gpt-4o":"G4o",
    "claude-3-opus":"C3",
    "claude-3.5-sonnet":"C35",
    "gemini-pro":"GP"
}

def process_sub_lists_zscore(sub_lists):
    summed_result = [[0.0 for _ in range(200)] for _ in range(len(sub_lists[0]))]
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
        for i in range(len(sub_lists[0])):
            for j in range(200):
                summed_result[i][j] += normalized_sub[i][j]

    return summed_result

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
        score_file_forward = "datasets/" + args.src_lan + "-" + args.tgt_lan + "-new/" + model_dict[
            model] + "_" + args.forward + "_" + str(cnt) + "." + args.metric
        values = []
        with open(score_file_forward, 'r') as file:
            for line in file:
                values.append(float(line.split()[0]))
        forward_scores.append(values)

    metric_scores_list = []

    for model_eval in models_eval:
        backward_scores = []
        for cnt in range(t):
            score_file_backward = "datasets/" + args.src_lan + "-" + args.tgt_lan + "-new/" + model_dict[
                model] + "_" + args.forward +  "_" + str(cnt) + "_" + model_dict[model_eval] + ".score"
            values = []
            with open(score_file_backward, 'r') as file:
                for line in file:
                    values.append(float(line.split()[0]))
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
    args = parser.parse_args()

    print(args.src_lan, args.tgt_lan)
    for model in args.models:
        print(model)
        for i in range(1,6):
            for j in range(1,6):
                compute(model, i, args.models[:j])
            print("")