from flair.embeddings import TokenEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from flair.data import Sentence, Token
from gensim.models import KeyedVectors
from typing import List
import re 
import numpy as np
import torch
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

class Embeddings:
    def __init__(self, embeddings_path, contextual_embeddings_type, layers) -> None:
        self.embeddings_path = embeddings_path
        self.contextual_embeddings_type = contextual_embeddings_type
        self.layers = layers
    
    def create_embeddings(self) -> StackedEmbeddings:
        embedding_types: List[FlairEmbeddings] = []
        
        if self.contextual_embeddings_type == 'clinical-flair':
            embedding_types.append(FlairEmbeddings(os.path.join(self.embeddings_path, 'es-clinical-forward.pt')))
            embedding_types.append(FlairEmbeddings(os.path.join(self.embeddings_path, 'es-clinical-backward.pt')))
        
        else:
            if self.layers=='all':
                embedding_types.append(TransformerWordEmbeddings(self.embeddings_path,
                layers = 'all', 
                layer_mean = True, 
                subtoken_pooling = 'first'))
            else:
                embedding_types.append(TransformerWordEmbeddings(self.embeddings_path,
                layers = '-1,-2,-3,-4', 
                layer_mean = False, 
                subtoken_pooling = 'first'))

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings = embedding_types)
        return embeddings