import torch

from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Module, BCELoss
from torch.optim import Optimizer, Adam, AdamW

from tqdm import tqdm
from loguru import logger

from ..metric import (
    precision_at_k, recall_at_k, map_at_k, ndcg_at_k
)

class BaseTrainer:
    def __init__(self, args) -> None:
        self.args = args
        self.device: torch.device = torch.device(self.args.device)
        self.model: Module = self._model(self.args.model).to(self.device)
        self.optimizer: Optimizer = self._optimizer(self.args.optimizer, self.model, self.args.lr)
        self.loss: BCELoss = self._loss(self.args.loss)

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
            logger.info(f"[Trainer] epoch: {epoch} > train loss: {train_loss} 
                        / valid loss: {valid_loss} 
                        / precision@K : {valid_precision_at_k} 
                        / Recall@K: {valid_recall_at_k} 
                        / MAP@K: {valid_map_at_k} 
                        / NDCG@K: {valid_ndcg_at_k}")
            
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
        logger.info(f"[Trainer] train...")
        self.model.train()

        total_loss: float = .0

        for i, data in enumerate(tqdm(train_dataloader)):
            X: Tensor = data['X'].to(self.device)
            y: Tensor = data['y'].to(self.device)
            pred: Tensor = self.model(X)
            batch_loss: Tensor = self.loss(pred, y) # loss.forward(input, target)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item() # require item call...
        
        return total_loss

    def validate(self, valid_dataloader: DataLoader) -> tuple[float]:
        logger.info(f"[Trainer] validate...")
        self.model.eval()

        valid_loss: float = .0
        valid_precision_at_k: float = .0
        valid_recall_at_k: float = .0
        valid_map_at_k: float = .0
        valid_ndcg_at_k: float = .0

        total_X = []

        for _, data in enumerate(tqdm(valid_dataloader)):
            X: Tensor = data['X'].to(self.device)
            y: Tensor = data['y'].to(self.device)
            pred: Tensor = self.model(X)
            batch_loss: Tensor = self.loss(pred, y)

            positive_index = torch.where(data['y'][:,0]==1)
            total_X.append(data['X'][positive_index])

            valid_loss += batch_loss.item()

        valid_loss /= len(valid_dataloader)

        # total_X = np.concatenate(total_X, axis=0)

        if self.valid_actual is None:
            self.valid_actual = self.actual_interaction_dict(total_X) # valid 평가시엔 valid actual로

        valid_precision_at_k, valid_recall_at_k, valid_map_at_k, valid_ndcg_at_k = self.evaluate()

        return valid_loss, valid_precision_at_k, valid_recall_at_k, valid_map_at_k, valid_ndcg_at_k
    
    def evaluate(self, k: int=20) -> tuple[float]:
        logger.info("[Trainer] evaluating....")
        self.model.eval()
        eval_precision_at_k: float = .0
        eval_recall_at_k: float = .0
        eval_map_at_k: float = .0
        eval_ndcg_at_k: float = .0

        num_users = self.cat_features_size['user']
        num_items = self.cat_features_size['item']
        offset = len(self.num_features)

        prediction = []
        
        logger.info("[EVAL]Predict all users and items interaction....")
        users = self.total_interaction[:, offset].unique().detach().cpu().numpy()
        for idx, user in enumerate(tqdm(users)):

            start_idx, end_idx = idx * num_items, (idx+1) * num_items
            user_X = self.total_interaction[start_idx:end_idx, :]
            user_items = user_X.detach().cpu().numpy()[:, offset+1]
            user_mask = torch.tensor([0 if item.item() in self.train_actual[int(user)] else 1 for item in user_items], dtype=int)

            user_pred = self.model(user_X.float()).detach().cpu()
            user_pred = user_pred.squeeze(1) * user_mask # train interaction 제외
            
            # find high prob index
            # high_index = np.argpartition(user_pred.numpy(), -k)[-k:]
            high_index = np.argsort(user_pred.numpy())[-k:]
            # find high prob item by index
            user_recom = user_items[high_index[::-1]]

            prediction.append(user_recom)

        assert len(prediction) == self.args.evaluate_size, f"prediction's length should be same as num_users({self.args.evaluate_size}): {len(prediction)}"

        eval_precision_at_k = precision_at_k(list(self.valid_actual.values()), prediction, k)
        eval_recall_at_k = recall_at_k(list(self.valid_actual.values()), prediction, k)
        eval_map_at_k = map_at_k(list(self.valid_actual.values()), prediction, k)
        eval_ndcg_at_k = ndcg_at_k(list(self.valid_actual.values()), prediction, k)

        return eval_precision_at_k, eval_recall_at_k, eval_map_at_k, eval_ndcg_at_k

    def inference(self):
        '''
        Umm.. is it need..?
        '''
        pass

    def load_best_model(self):
        pass