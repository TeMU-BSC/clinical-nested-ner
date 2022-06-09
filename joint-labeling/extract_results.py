"""Script to extract model results from experiments"""
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict


def read_metrics(metric_names, results_file: Path):
    dir_name = results_file.parent.__str__()
    metrics = {"run_dir": dir_name}
    with open(results_file) as rf:
        results = json.load(rf)
    for metric_name in metric_names:
        metrics[metric_name] = results[metric_name]
    return metrics


if __name__ == '__main__':

    # TODO: add arguments
    runs_dir = "/home/ccasimiro/ccasimiro/clinical-nested-ner/language-models/cte-amd/runs/"
    path = Path(runs_dir)
    results_filename = "all_results.json"
    results_files = list(path.rglob(results_filename))
    metric_names = ['eval_accuracy', 'eval_f1', 'predict_accuracy', 'predict_f1',
                    'eval_precision', 'eval_recall', 'predict_precision', 'predict_recall']
    metrics_all = []
    for results_file in results_files:
        metrics_all.append(read_metrics(metric_names, results_file))
    df = pd.DataFrame.from_dict(metrics_all)
    # Expand the path parts (they represent experiments configurations),
    # concatenate to the dataframe and remove the original splitted path
    df = pd.concat([df, df['run_dir'].str.split('/', expand=True)],
                   axis=1).drop(columns=[('run_dir')])
    df = df.groupby([8, 9, 11], as_index=False).agg(
        {metric_name: ['mean', 'std'] for metric_name in metric_names})
    df.to_csv(path.joinpath('results_joint_labeling.csv'), index=False)
