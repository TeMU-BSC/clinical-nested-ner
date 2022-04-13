from unicodedata import bidirectional
from datasets import NERCorpus
from models import NERTagger
from embeddings import Embeddings
from trainers import NERTrainer
import yaml
import torch 
import flair
import os

if __name__=='__main__':
   
    torch.cuda.empty_cache()
    # Read configuration file
    with open('../config.yaml') as file:
        config = yaml.safe_load(file)


    # Device
    available_gpu = torch.cuda.is_available()
    if available_gpu:
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        flair.device = torch.device('cuda')
    else:
        flair.device = torch.device('cpu')

    # Create corpus
    corpus_name = config['dataset']
    task = ''
    
    actual_path = os.getcwd()

    directory = os.fsencode(f'../data/{corpus_name}/')
  
    for file in os.listdir(directory):
        entity_type = os.fsdecode(file)

        corpus = NERCorpus(corpus_name).create_corpus(entity_type)
        print(corpus)

        # print the first Sentence in the training split
        print(corpus.train[0].to_tagged_string('ner'))
        tag_dictionary = corpus.make_label_dictionary(label_type = 'ner')
        print(tag_dictionary)
        # Embeddings
        embeddings = Embeddings(config['pretrained_embeddings_path'], config['contextual_embeddings_type']).create_embeddings()
        
        
        # Create Sequence Labeling Model
        tagger = NERTagger(embeddings = embeddings,
            encoder = config['ner_hyperparameters']['encoder'],
            encoder_layers = config['ner_hyperparameters']['encoder_layers'],
            use_crf = config['ner_hyperparameters']['use_crf'],
            hidden_size = config['ner_hyperparameters']['hidden_size'],
            dropout = config['ner_hyperparameters']['dropout'],
            tag_dictionary = tag_dictionary
        ).create_tagger()

        # Create Sequence Labeling Trainer
        trainer = NERTrainer(corpus = corpus,
            tagger = tagger,
            epochs = config['ner_training']['max_epochs'],
            learning_rate = config['ner_training']['learning_rate'],
            mini_batch_size = config['ner_training']['mini_batch_size'],
            optimizer = config['ner_training']['optimizer'],
            output_path = '{}_{}_{}_output/'.format(corpus_name, entity_type, config['contextual_embeddings_type'])
        ).train()