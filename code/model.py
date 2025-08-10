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


def process_sub_lists_avg(sub_lists):
    summed_result = [[0.0 for _ in range(200)] for _ in range(5)]
    for sub in sub_lists:

        for i in range(5):
            for j in range(200):
                summed_result[i][j] += sub[i][j]


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

    print(round((sum(final_scores) / len(final_scores)) * 100, 2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Command-line script to use')
    parser.add_argument('--src_lan', type=str, default='', help='source language')
    parser.add_argument('--tgt_lan', type=str, default='', help='target language')
    parser.add_argument('--forward', type=str, default='', help='forward translation method')
    parser.add_argument('--metric', type=str, default='', help='selection metric')
    parser.add_argument('--models', nargs='+', help='LLMs')
    args = parser.parse_args()

    print(args.src_lan, args.tgt_lan)

    forward_scores = []
    for model in args.models:
        score_file_forward = "datasets/" + args.src_lan + "-" + args.tgt_lan + "-new/" + model_dict[
            model] + "_" + args.forward + "." + args.metric
        values = []
        with open(score_file_forward, 'r') as file:
            for line in file:
                values.append(float(line.split()[0]))
        print(model_dict[model], round(mean(values)*100, 2))
        forward_scores.append(values)
    max_forward_index = []
    for i in range(len(forward_scores[0])):
        max_val = max(item[i] for item in forward_scores)
        index = [item[i] for item in forward_scores].index(max_val)
        max_forward_index.append(index)

    backward_scores_list = []

    for model_eval in args.models:
        backward_scores = []
        for model_predict in args.models:
            score_file_backward = "datasets/" + args.src_lan + "-" + args.tgt_lan + "-new/" + model_dict[
                model_predict] + "_" + args.forward + "_" + model_dict[model_eval] + ".score"
            values = []
            with open(score_file_backward, 'r') as file:
                for line in file:
                    values.append(float(line.split()[0]))
            backward_scores.append(values)
        backward_scores_list.append(backward_scores)

        output_results(forward_scores, backward_scores)

    backward_scores_avg = process_sub_lists_avg(backward_scores_list)
    print("Avg:")
    output_results(forward_scores, backward_scores_avg)
