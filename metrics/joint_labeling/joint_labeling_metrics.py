from seqeval.metrics.sequence_labeling import get_entities
from nestednereval.metrics import standard_metric
from nestednereval.metrics import inner_metric
from nestednereval.metrics import outer_metric
from nestednereval.metrics import flat_metric
from nestednereval.metrics import nested_metric
from nestednereval.metrics import nesting_metric
import os
import sys
from tqdm import tqdm

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

                        output_file.write(
                            f'Dataset: {dataset}, Model: {model}, Seed: {seed}, Learning_rate: {lr} \n')
                        p, r, f, support = standard_metric(data)
                        output_file.write(
                            f'Standard metric - Precision: {p}, Recall: {r}, F1 score: {f}, Support: {support}\n')
                        p, r, f, support = flat_metric(data)
                        output_file.write(
                            f'Flat metric - Precision: {p}, Recall: {r}, F1 score: {f}, Support: {support}\n')
                        p, r, f, support = inner_metric(data)
                        output_file.write(
                            f'Inner metric - Precision: {p}, Recall: {r}, F1 score: {f}, Support: {support}\n')
                        p, r, f, support = outer_metric(data)
                        output_file.write(
                            f'Outer metric - Precision: {p}, Recall: {r}, F1 score: {f}, Support: {support}\n')
                        p, r, f, support = nested_metric(data)
                        output_file.write(
                            f'Nested metric - Precision: {p}, Recall: {r}, F1 score: {f}, Support: {support}\n')
                        p, r, f, support = nesting_metric(data)
                        output_file.write(
                            f'Nesting metric - Precision: {p}, Recall: {r}, F1 score: {f}, Support: {support}\n')
                        output_file.write('\n')
