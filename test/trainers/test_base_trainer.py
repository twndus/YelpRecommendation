from omegaconf import DictConfig

from ...trainers.base_trainer import BaseTrainer

def test_base_trainer_model():
    args: DictConfig = DictConfig({
        'device': 'cpu',
        'model': 'a',
    })

    try:
        BaseTrainer(args)
    except NotImplementedError as e: 
        assert e.args[0] == "Not implemented model: a"