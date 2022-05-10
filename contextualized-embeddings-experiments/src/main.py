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
    available_gpu = torch.cuda.is_available()
    if available_gpu:
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        flair.device = torch.device('cuda')
    else:
        flair.device = torch.device('cpu')


    for seed in (1, 2, 3):
        flair.set_seed(seed)
        torch.cuda.empty_cache()

        for dataset in ('clinical_trials', 'pharmaconer', 'wl'):
            actual_path = os.getcwd()
            directory = os.fsencode(f'../data/{dataset}/')
        
            for file in os.listdir(directory):
                entity_type = os.fsdecode(file)

                corpus = NERCorpus(dataset).create_corpus(entity_type)
                tag_dictionary = corpus.make_label_dictionary(label_type = 'ner')
            
                #for model in ('clinical-flair', 'biomedical-roberta'):
                for model in ('biomedical-roberta', 'clinical-flair'):
                   
                    if model=='biomedical-roberta':

                        for layers in ('last-4', 'all'):
                        # Embeddings
                            embeddings = Embeddings('embeddings/cwlce.vec', model, layers).create_embeddings()
                        
                        


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
                                output_path = 'corpus_{}_entity_type_{}_model_{}_seed_{}_layers_{}/'.format(dataset, entity_type, model, seed, layers)
                            ).train()
                    
                    else:
                        embeddings = Embeddings('embeddings/cwlce.vec', model, layers='None').create_embeddings()
                    
                    


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
                            output_path = 'corpus_{}_entity_type_{}_model_{}_seed_{}/'.format(dataset, entity_type, model, seed)
                        ).train()
