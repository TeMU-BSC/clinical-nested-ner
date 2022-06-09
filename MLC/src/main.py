from unicodedata import bidirectional
from flair_datasets import NERCorpus
from models import NERTagger
from embeddings import Embeddings
from trainers import NERTrainer
from argparse import ArgumentParser
import yaml
import torch 
import flair
import os
import sys

if __name__=='__main__':
    # TODO: add argparser to input list arguments
    parser = ArgumentParser()
    parser.add_argument('run_dir', type=str, help='Output directory to store the experiment')
    parser.add_argument('models', nargs='+', help='Models to train')
    parser.add_argument('datasets', nargs='+', help='Dataset to train models')
    parser.add_argument('seeds', nargs='+', help='Seeds to use')
    args = parser.parse_args()

    available_gpu = torch.cuda.is_available()
    if available_gpu:
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        flair.device = torch.device('cuda')
    else:
        flair.device = torch.device('cpu')


    for seed in args.seeds:
        seed = int(seed)
        flair.set_seed(seed)
        torch.cuda.empty_cache()

        for dataset in args.datasets:
            actual_path = os.getcwd()
            directory = os.fsencode(f'../data/{dataset}/')
        
            for file in os.listdir(directory):
                entity_type = os.fsdecode(file)

                corpus = NERCorpus(dataset).create_corpus(entity_type)
                tag_dictionary = corpus.make_label_dictionary(label_type = 'ner')
            
                #for model in ('clinical-flair', 'biomedical-roberta'):
                # for model in ('biomedical-roberta', 'clinical-flair'):
                for model in args.models:
                    output_path = f'{args.run_dir}/dataset-{dataset}/entity-{entity_type}/model-{os.path.basename(model)}/seed-{seed}/'
                    os.makedirs(output_path, exist_ok=True)
                
                    
                    if 'clinical-flair' in model:
                        embeddings = Embeddings(embeddings_path=model, 
                                                contextual_embeddings_type='clinical-flair', 
                                                layers='None').create_embeddings()
                    
                    else:
                        for layers in ('last-4', 'all'):
                            # Embeddings
                            
                            embeddings = Embeddings(embeddings_path=model, 
                                                    contextual_embeddings_type='transformers',
                                                    layers=layers).create_embeddings()

                            # Create Sequence Labeling Model
                            tagger = NERTagger(embeddings = embeddings,
                                use_crf = True,
                                hidden_size = 256,
                                tag_dictionary = tag_dictionary
                            ).create_tagger()

                            # Create Sequence Labeling Trainer
                            trainer = NERTrainer(corpus = corpus,
                                tagger = tagger,
                                epochs = 100,
                                learning_rate = 0.1,
                                mini_batch_size = 16,
                                output_path = output_path
                            ).train()

                    # Create Sequence Labeling Model
                        tagger = NERTagger(embeddings = embeddings,
                            use_crf = True,
                            hidden_size = 256,
                            tag_dictionary = tag_dictionary
                        ).create_tagger()

                        # Create Sequence Labeling Trainer
                        trainer = NERTrainer(corpus = corpus,
                            tagger = tagger,
                            epochs = 100,
                            learning_rate = 0.1,
                            mini_batch_size = 16,
                            output_path = output_path
                        ).train()
