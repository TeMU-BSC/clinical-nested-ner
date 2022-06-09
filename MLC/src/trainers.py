from flair.trainers import ModelTrainer
from torch.optim import Adam, SGD

class NERTrainer:
    def __init__(self, corpus, tagger, epochs, learning_rate, mini_batch_size, output_path):  
        self.corpus = corpus
        self.tagger = tagger
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.output_path = output_path

    def train(self):

        trainer: ModelTrainer = ModelTrainer(
                model = self.tagger, 
                corpus = self.corpus)

        print(trainer)
        
        trainer.train(
            base_path = '{}'.format(self.output_path),
            learning_rate = self.learning_rate,
            train_with_dev = True,
            embeddings_storage_mode = 'none',
            mini_batch_size = self.mini_batch_size,
            max_epochs = self.epochs
            ) 