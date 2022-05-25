from seqeval.metrics.sequence_labeling import get_entities
from nestednereval.metrics import standard_metric
from nestednereval.metrics import inner_metric
from nestednereval.metrics import outer_metric
from nestednereval.metrics import flat_metric
from nestednereval.metrics import nested_metric
from nestednereval.metrics import nesting_metric
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import os
import sys

if __name__ == '__main__':
    preds_dir = sys.argv[1]
    output_file = open('metrics.txt', 'w')
    wl_entity_types = ["Disease", "Medication", "Body_Part",
                       "Abbreviation", "Family_Member", "Finding", "Procedure"]
    clinical_trials_entity_types = ["DISO", "CHEM", "PROC", "ANAT"]
    datasets = ["wl", "clinical_trials"]
    seeds = [1, 2, 3]
    lrs = ['1e-5', '1e-6', '5e-5', '5e-6']

    for model in tqdm(os.listdir(f'{preds_dir}'), desc="Generating nested metrics....................."):
        for dataset in datasets:
            
            metrics = {lr: defaultdict(list) for lr in lrs}
           

            for seed in seeds:
                
                for lr in lrs:
                    
                    path = f'{preds_dir}/{model}/{dataset}/seed-{seed}/lr-{lr}/preds_with_labels.conll'
                    if os.path.isfile(path):
                        prediction_file = open(path, 'r', encoding='utf-8').read()
                        sentences = prediction_file.split('\n\n')[:-1]
                        data = []
                        for sent in sentences:

                            entity_types = wl_entity_types if dataset == 'wl' else clinical_trials_entity_types
                            gold_entities_dict = {k: [] for k in entity_types}
                            pred_entities_dict = {k: [] for k in entity_types}

                            for line in sent.splitlines():
                                entity_types = wl_entity_types if dataset == 'wl' else clinical_trials_entity_types
                                for type in entity_types:
                                    if f'B-{type}' in line.split()[1].split('+'):
                                        gold_entities_dict[type].append(
                                            f'B-{type}')
                                    elif f'I-{type}' in line.split()[1].split('+'):
                                        gold_entities_dict[type].append(
                                            f'I-{type}')
                                    else:
                                        gold_entities_dict[type].append('O')
                                    if f'B-{type}' in line.split()[2].split('+'):
                                        pred_entities_dict[type].append(
                                            f'B-{type}')
                                    elif f'I-{type}' in line.split()[2].split('+'):
                                        pred_entities_dict[type].append(
                                            f'I-{type}')
                                    else:
                                        pred_entities_dict[type].append('O')
                            real_entities = []
                            for k, v in gold_entities_dict.items():
                                real_entities.extend(get_entities(v))
                            pred_entities = []
                            for k, v in pred_entities_dict.items():
                                pred_entities.extend(get_entities(v))
                            data.append(
                                {"real": real_entities, "pred": pred_entities})

                        p, r, f, support = standard_metric(data)
                        metrics[lr]["standard"].append({"precision": p, "recall": r, "f1": f})

                        p, r, f, support = flat_metric(data)
                        metrics[lr]["flat"].append({"precision": p, "recall": r, "f1": f})

                        p, r, f, support = inner_metric(data)
                        metrics[lr]["inner"].append({"precision": p, "recall": r, "f1": f})

                        p, r, f, support = outer_metric(data)
                        metrics[lr]["outer"].append({"precision": p, "recall": r, "f1": f})

                        p, r, f, support = nested_metric(data)
                        metrics[lr]["nested"].append({"precision": p, "recall": r, "f1": f})

                        p, r, f, support = nesting_metric(data)
                        metrics[lr]["nesting"].append({"precision": p, "recall": r, "f1": f})

       

            for k1, v1 in metrics.items():
             
                print(f"Dataset: {dataset}, Transformer model: {model}, Learning Rate: {k1}.\n")
                output_file.write(f"Dataset: {dataset}, Transformer model: {model}, Learning Rate: {k1}.\n")
                for k2, v2 in v1.items():
                    
                    d = {"precision": [], "recall": [], "f1": []}

                    for seed_metrics in v2:
                        for k3, v3 in seed_metrics.items():
                            d[k3].append(v3)

                    
                    output_file.write(
                                    f'{k2} metric - precision: {round(np.mean(d["precision"])*100, 2)} ({round(np.std(d["precision"]), 3)}), recall: {round(np.mean(d["recall"])*100, 2)} ({round(np.std(d["recall"]), 3)}), f1-score: {round(np.mean(d["f1"])*100, 2)} ({round(np.std(d["f1"]), 3)})\n')
                output_file.write('\n')

            
           
                       
