from flair.data import Corpus
from flair.datasets import ColumnCorpus

class NERCorpus:
    def __init__(self, corpus_name) -> None:
        self.corpus_name = corpus_name
        
    def create_corpus(self, entity_type) -> Corpus:
        corpus: Corpus = ColumnCorpus(data_folder = '../data/{}/{}/'.format(self.corpus_name, entity_type),  
                                                column_format = {0: 'text', 1: 'ner'},
                                                train_file = '{}_train.iob2'.format(entity_type),
                                                test_file = '{}_test.iob2'.format(entity_type),
                                                dev_file = '{}_dev.iob2'.format(entity_type))
        return corpus