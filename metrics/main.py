from nestednereval.utils import read_iob2_prediction_file, merge_predictions
from nestednereval.metrics import nested_ner_metrics

if __name__=='__main__':
    #models = ['flair', 'flert', 'roberta', 'spanish-bert']
    models = ['flair']
    #datasets = ['wl', 'clinical_trials']
    datasets = ['clinical_trials']

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

            print()
            print(f'Nested metrics of model: {model} in dataset: {dataset}.\n')
            nested_ner_metrics(entities) 
            print() 