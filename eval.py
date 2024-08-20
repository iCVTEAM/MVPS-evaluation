import os
from time import time
import argparse
import numpy as np
import pandas as pd
from evaluation import MVPSEvaluation


default_mvps_path = '/path/to/the/folder/MVPS'


if __name__ == '__main__':
    time_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('mask_path', type=str, help='Path to the mask folder containing the sequences folders')
    parser.add_argument('--mvps_path', type=str, help='Path to the MVPS dataset folder', default=default_mvps_path)
    parser.add_argument('--set', type=str, help='Subset to evaluate the results', default='val')
    parser.add_argument('--include_first_last', action='store_true', help='Include the first and the last frame')
    args = parser.parse_args()
    csv_name_global = f'global_results-{args.set}.csv'
    csv_name_per_sequence = f'per-sequence_results-{args.set}.csv'

    csv_name_global_path = os.path.join(args.mask_path, csv_name_global)
    csv_name_per_sequence_path = os.path.join(args.mask_path, csv_name_per_sequence)
    if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
        print('Using precomputed results')
        table_g = pd.read_csv(csv_name_global_path)
        table_seq = pd.read_csv(csv_name_per_sequence_path)
    else:
        print('Evaluating sequences')
        dataset_eval = MVPSEvaluation(mvps_root=args.mvps_path, subset=args.set, include_first_last=args.include_first_last)
        metrics_res = dataset_eval.evaluate(args.mask_path)
        J, F = metrics_res['J'], metrics_res['F']

        final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
        g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
        g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]), np.mean(F["D"])])
        g_res = np.reshape(g_res, [1, len(g_res)])
        table_g = pd.DataFrame(data=g_res, columns=g_measures)
        with open(csv_name_global_path, 'w') as f:
            table_g.to_csv(f, index=False, float_format="%.6f")
        print(f'Global results saved in {csv_name_global_path}')

        # Generate a dataframe for the per sequence results
        seq_measures = ['Sequence'] + g_measures[1:]
        seq_names = metrics_res['seq_names']
        table_seq = pd.DataFrame(data=list(zip(seq_names, J['M'], J['R'], J['D'], F['M'], F['R'], F['D'])), columns=seq_measures)
        with open(csv_name_per_sequence_path, 'w') as f:
            table_seq.to_csv(f, index=False, float_format="%.6f")
        print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

    # Print the results
    print(f"--------------------------- Global results for {args.set} ---------------------------")
    print(table_g.to_string(index=False))
    print(f"\n---------- Per sequence results for {args.set} ----------")
    print(table_seq.to_string(index=False))
    total_time = time() - time_start
    print('\nTotal time:' + str(total_time))
