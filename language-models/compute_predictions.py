from transformers import pipeline
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import shutil
import json
import re
import os


if __name__ == '__main__':
    runs_dir = "/home/ccasimiro/ccasimiro/clinical-nested-ner/language-models/cte-amd/runs/"
    path = Path(runs_dir)
    pretrained_models = [os.path.dirname(model) for model in path.rglob('*pytorch_model.bin')]
    testsets = ['wl', 'clinical_trials']

    for model in tqdm(pretrained_models):
        print(model)
        pipe = pipeline('token-classification', model, device=0)
        model_name = re.findall('.*runs/(.*)', model)[0]

        with open(os.path.join(model, 'preds_with_labels.conll'), 'w') as fn:
            for testset in tqdm(testsets):
                print(testset)
                dataset = load_dataset('/home/ccasimiro/ccasimiro/clinical-nested-ner/datasets/load_ner_iob2.py', testset, 
                                       cache_dir='/home/ccasimiro/ccasimiro/clinical-nested-ner/language-models/.cache')
                
                for words, labels in tqdm(zip(dataset['test']['tokens'], dataset['test']['ner_tags'])):
                    sentence = ' '.join(words)
                    prediction = pipe(sentence)
                    start_word = 0
                    for word, label in zip(words, labels):
                        label_name =  dataset['test'].features['ner_tags'].feature.names[label]

                        start_word = sentence.find(word, start_word)
                        end_word = start_word + len(word)
                        for el in prediction:
                            if el['end'] <= end_word:
                                pred = el['entity']
                                start_word = el['end']
                            else:
                                break

                        fn.write(f"{word}\t{label_name}\t{pred}\n")
