from nested_entities_utils import read_iob2_prediction_file, merge_predictions
from nested_ner_metrics import standard_metric, flat_metric, inner_metric, outer_metric, nested_metric, nesting_metric
import numpy as np 

if __name__=='__main__':
    models = ['flair', 'flert', 'roberta', 'spanish-bert']
    datasets = ['wl', 'clinical_trials', 'pharmaconer']

    for model in models:
        for dataset in datasets:
            if dataset == 'pharmaconer':
                entity_types = ['NORMALIZABLES', 'NO_NORMALIZABLES', 'PROTEINAS', 'UNCLEAR']
            if dataset == 'clinical_trials':
                entity_types = ['ANAT', 'CHEM', 'DISO', 'PROC']
            if dataset == 'wl':
                entity_types = ['Disease', 'Medication', 'Body_Part', 'Abbreviation', 'Finding', 'Procedure', 'Family_Member']
            
            chunks = []

            for entity in entity_types:
                entity_chunks = read_iob2_prediction_file(f"mlc-{model}/{dataset}_{entity}/test.tsv") 
                chunks.append(entity_chunks)

            entities = merge_predictions(chunks)  
            
            print("=========================================================================================")
            _, _, f1, support = standard_metric(entities)
            print(f'{model} - {dataset} - Standard metric - F1-score: {np.round(f1*100,2)}, support: {support}')
            
            if dataset in ('clinical_trials', 'wl'):

                _, _, flat_f1, support = flat_metric(entities)
                print(f'{model} - {dataset} - Flat f1-score: {np.round(flat_f1*100,2)}, support: {support}')
                _, _, nested_f1, support = nested_metric(entities)
                print(f'{model} - {dataset} - Nested f1-score: {np.round(nested_f1*100,2)}, support: {support}')
                _, _, inner_f1, support = inner_metric(entities)
                print(f'{model} - {dataset} - Inner f1-score: {np.round(inner_f1*100,2)}, support: {support}')
                _, _, outer_f1, support = outer_metric(entities)
                print(f'{model} - {dataset} - Outer f1-score: {np.round(outer_f1*100,2)}, support: {support}')
                _, _, nesting_f1, support = nesting_metric(entities)
                print(f'{model} - {dataset} - Nesting f1-score: {np.round(nesting_f1*100,2)}, support: {support}')
            print("=========================================================================================")    
                
            # Con lo anterior puedo calcular las métricas, en el archivo métricas utilizar la función get nestings



    
  
