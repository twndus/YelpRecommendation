from abc import abstractmethod

class BaseTrainer:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def inference(self):
        pass

    @abstractmethod
    def load_best_model(self):
        pass