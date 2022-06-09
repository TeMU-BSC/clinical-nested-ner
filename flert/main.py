from flair.data import Corpus
from flair.datasets import ColumnCorpus
import torch 
from flair.embeddings import FlairEmbeddings, StackedEmbeddings, TokenEmbeddings, TransformerWordEmbeddings, CharacterEmbeddings
import numpy as np
from typing import List
from gensim.models import KeyedVectors
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import re
import flair
from flair.data import Sentence, Token
from torch.optim.lr_scheduler import OneCycleLR
import os


class W2vWordEmbeddings(TokenEmbeddings):

    def __init__(self, embeddings, static_embeddings, device, binary):
        super().__init__()
        self.name = embeddings
        self.static_embeddings = static_embeddings
        self.device = device
        self.precomputed_word_embeddings = KeyedVectors.load_word2vec_format(embeddings, binary=binary)
        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for i, sentence in enumerate(sentences):
            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token
                if token.text in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[token.text]
                elif token.text.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[token.text.lower()]
                elif re.sub('\d', '#', token.text.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub('\d', '#', token.text.lower())]
                elif re.sub('\d', '0', token.text.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub('\d', '0', token.text.lower())]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype='float')
                word_embedding = torch.FloatTensor(word_embedding).to(self.device)
                token.set_embedding(self.name, word_embedding)
        return sentences

available_gpu = torch.cuda.is_available()
if available_gpu:
    flair.device = torch.device('cuda')
else:
    flair.device = torch.device('cpu')

for seed in [1, 2, 3]:
        flair.set_seed(seed)
        torch.cuda.empty_cache()

        for dataset in ['wl']:
            actual_path = os.getcwd()
            directory = os.fsencode(f'data/{dataset}/')
        
            for file in os.listdir(directory):
                entity_type = os.fsdecode(file)
                print(entity_type)
                corpus = ColumnCorpus(data_folder = f'data/{dataset}/{entity_type}',  column_format = {0: 'text', 1: 'ner'}, train_file = f'{entity_type}_train.iob2', test_file = f'{entity_type}_test.iob2', dev_file = f'{entity_type}_dev.iob2')





                # 1. get the corpus


                # 2. what label do we want to predict?
                label_type = 'ner'

                # 3. make the label dictionary from the corpus
                label_dict = corpus.make_label_dictionary(label_type=label_type)
                print(label_dict)

                # 4. initialize embedding stack with Flair and GloVe
                embedding_types = [
                    TransformerWordEmbeddings(model="dccuchile/bert-base-spanish-wwm-cased",
                                                    layers="-1",
                                                    subtoken_pooling="first",
                                                    fine_tune=True,
                                                    use_context=True,
                                                    )]

                embeddings = StackedEmbeddings(embeddings=embedding_types)


                # 5. initialize sequence tagger

                tagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=label_dict,
                                        tag_type='ner',
                                        use_crf=False,
                                        use_rnn=False,
                                        reproject_embeddings=False,
                                        )


                # 6. initialize trainer
                trainer = ModelTrainer(tagger, corpus)

                #trainer.train('', mini_batch_size=16, max_epochs=100)
                # 7. start training
                trainer.fine_tune(f'{dataset}_{entity_type}_{seed}_flert_beto',
                                learning_rate=5.0e-6,
                                mini_batch_size=4,
                                mini_batch_chunk_size=1,
                                max_epochs=20,
                                scheduler=OneCycleLR,
                                embeddings_storage_mode='none',
                                weight_decay=0.,)