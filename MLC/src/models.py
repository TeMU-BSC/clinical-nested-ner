from unicodedata import bidirectional
from flair.models import SequenceTagger

class NERTagger:
    def __init__(self, embeddings, use_crf, hidden_size, tag_dictionary) -> None:
        self.embeddings = embeddings
        self.use_crf = use_crf
        self.hidden_size = hidden_size
        self.tag_dictionary = tag_dictionary

        
        
    def create_tagger(self) -> SequenceTagger:
        tagger: SequenceTagger = SequenceTagger(
                                    hidden_size = self.hidden_size,
                                    embeddings = self.embeddings,
                                    tag_dictionary = self.tag_dictionary,
                                    use_crf = self.use_crf,
                                    tag_type = 'ner'
                                )
        return tagger