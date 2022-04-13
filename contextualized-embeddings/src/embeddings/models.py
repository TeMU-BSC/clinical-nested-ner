from unicodedata import bidirectional
from flair.models import SequenceTagger

class NERTagger:
    def __init__(self, embeddings, encoder, encoder_layers, use_crf, hidden_size, dropout, tag_dictionary) -> None:
        self.embeddings = embeddings
        self.encoder = encoder
        self.encoder_layers = encoder_layers
        self.use_crf = use_crf
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.tag_dictionary = tag_dictionary

        
        
    def create_tagger(self) -> SequenceTagger:
        tagger: SequenceTagger = SequenceTagger(
                                    rnn_type = self.encoder,
                                    hidden_size = self.hidden_size,
                                    dropout = self.dropout,
                                    embeddings = self.embeddings,
                                    tag_dictionary = self.tag_dictionary,
                                    use_crf = self.use_crf,
                                    rnn_layers = self.encoder_layers,
                                    tag_type = 'ner'
                                )
        return tagger