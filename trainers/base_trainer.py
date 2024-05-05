import torch
import numpy as np

from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Module, BCELoss
from torch.optim import Optimizer, Adam, AdamW

from tqdm import tqdm
from loguru import logger
from omegaconf.dictconfig import DictConfig

from ..metric import (
    precision_at_k, recall_at_k, map_at_k, ndcg_at_k
)

class BaseTrainer:
    def __init__(self, args: DictConfig) -> None:
        self.args: DictConfig = args
        self.device: torch.device = self._device(self.args.device)
        self.model: Module = self._model(self.args.model).to(self.device)
        self.optimizer: Optimizer = self._optimizer(self.args.optimizer, self.model, self.args.lr)
        self.loss: BCELoss = self._loss(self.args.loss)

    def _device(self, device_name: str) -> torch.device:
        if device_name.lower() in ('cpu', 'cuda',):
            return torch.device(device_name.lower())
        else:
            logger.error(f"Not supported device: {device_name}")
            return torch.device('cpu')

    def _model(self, model_name: str) -> Module:
        if model_name.lower() in ('test',):
            return type("TestModel", (Module,), {"forward": (lambda self, x: x)})()
        else:
            logger.error(f"Not implemented model: {model_name}")
            raise NotImplementedError(f"Not implemented model: {model_name}")
    
    def _optimizer(self, optimizer_name: str, model: Module, learning_rate: float) -> Optimizer:
        if optimizer_name.lower() == 'adam':
            return Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'adamw':
            return AdamW(model.parameters(), lr=learning_rate)
        else:
            logger.error(f"Optimizer Not Exists: {optimizer_name}")
            raise NotImplementedError(f"Optimizer Not Exists: {optimizer_name}")
    
    def _loss(self, loss_name: str):
        if loss_name.lower() == 'bce':
            return BCELoss()
        else:
            logger.error(f"Loss Not Exists: {loss_name}")
            raise NotImplementedError(f"Loss Not Exists: {loss_name}")
    
    def run(self, train_dataloader: DataLoader, valid_dataloader: DataLoader):
        logger.info(f"[Trainer] run...")

        best_valid_loss: float = 1e+6
        best_valid_precision_at_k: float = .0
        best_valid_recall_at_k: float = .0
        best_valid_map_at_k: float = .0
        best_valid_ndcg_at_k: float = .0
        best_epoch: int = 0
        endurance: int = 0

        # train
        for epoch in range(self.args.epochs):
            train_loss: float = self.train(train_dataloader)
            (valid_loss,
             valid_precision_at_k,
             valid_recall_at_k,
             valid_map_at_k,
             valid_ndcg_at_k) = self.validate(valid_dataloader)
            logger.info(f'''[Trainer] epoch: {epoch} > train loss: {train_loss} / 
                        valid loss: {valid_loss} / 
                        precision@K : {valid_precision_at_k} / 
                        Recall@K: {valid_recall_at_k} / 
                        MAP@K: {valid_map_at_k} / 
                        NDCG@K: {valid_ndcg_at_k}''')
            
            # update model
            if best_valid_loss > valid_loss:
                logger.info(f"[Trainer] update best model...")
                best_valid_loss = valid_loss
                best_valid_precision_at_k = valid_precision_at_k
                best_recall_k = valid_recall_at_k
                best_valid_ndcg_at_k = valid_ndcg_at_k
                best_valid_map_at_k = valid_map_at_k
                best_epoch = epoch
                endurance = 0

                # TODO: add mlflow

                torch.save(self.model.state_dict(), f'{self.best_model_dir}/best_model.pt')
            else:
                endurance += 1
                if endurance > self.args.patience: 
                    logger.info(f"[Trainer] ealry stopping...")
                    break

    def train(self, train_dataloader: DataLoader) -> float:
        raise NotImplementedError("[ERROR] Trainer train method is not implemented...")
        # logger.info(f"[Trainer] train...")
        # self.model.train()

        # total_loss: float = .0

        # for i, data in enumerate(tqdm(train_dataloader)):
        #     X: Tensor = data['X'].to(self.device)
        #     y: Tensor = data['y'].to(self.device)
        #     pred: Tensor = self.model(X)
        #     batch_loss: Tensor = self.loss(pred, y) # loss.forward(input, target)

        #     self.optimizer.zero_grad()
        #     batch_loss.backward()
        #     self.optimizer.step()

        #     total_loss += batch_loss.item() # require item call...
        
        # return total_loss
    
    def validate(self, valid_dataloader: DataLoader) -> tuple[float]:
        raise NotImplementedError("[ERROR] Trainer validate method is not implemented...")
        # self.model.eval()

        # valid_loss: float = .0
        # valid_precision_at_k: float = .0
        # valid_recall_at_k: float = .0
        # valid_map_at_k: float = .0
        # valid_ndcg_at_k: float = .0

        # for _, data in enumerate(tqdm(valid_dataloader)):
        #     X: Tensor = data['X'].to(self.device)
        #     y: Tensor = data['y'].to(self.device)
        #     pred: Tensor = self.model(X)
        #     batch_loss: Tensor = self.loss(pred, y)

        #     valid_loss += batch_loss.item()

        # valid_loss /= len(valid_dataloader)

        # valid_precision_at_k, valid_recall_at_k, valid_map_at_k, valid_ndcg_at_k = self.evaluate()

        # return valid_loss, valid_precision_at_k, valid_recall_at_k, valid_map_at_k, valid_ndcg_at_k
    
    def evaluate(self, k: int=20) -> tuple[float]:
        raise NotImplementedError("[ERROR] Trainer evaluate method is not implemented...")

    def inference(self):
        pass

    def load_best_model(self):
        logger.info(f"[Trainer] Load best model...")
        self.model.load_state_dict(torch.load(f'{self.best_model_dir}/best_model.pt'))
    