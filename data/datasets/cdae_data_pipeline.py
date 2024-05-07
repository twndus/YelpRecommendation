import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

from .data_pipeline import DataPipeline

class CDAEDataPipeline(DataPipeline):

    def __init__(self, cfg):
        super().__init__(cfg)

    def split(self, df):
        '''
           train_data: ((user-specific vector, item sets, neg_data), ...)
        '''
        logger.info(f'start random user split...')

        item_list = df.columns[1:].astype(int)
        train_data, valid_data, test_data = {}, {}, {}

        for i, row in tqdm(df.iterrows(), total=df.shape[0]):

            # train+valid+test mask
            user_id = int(row['user_id'])
            user_history = np.argwhere(row[1:]).reshape(-1)

            # split positive samples
            np.random.shuffle(user_history)
            train_samples, test_samples = np.split(user_history, [int(0.8*len(user_history))])
            train_samples, valid_samples = np.split(train_samples, [int(0.9*len(train_samples))])

            train_mask = np.zeros_like(row[1:], dtype=np.int32)
            train_mask[train_samples] = 1

            valid_mask = np.zeros_like(row[1:], dtype=np.int32)
            valid_mask[valid_samples] = 1

            test_mask = np.zeros_like(row[1:], dtype=np.int32)
            test_mask[test_samples] = 1

            train_valid_mask = np.zeros_like(row[1:], dtype=np.int32)
            train_valid_mask[np.union1d(train_samples, valid_samples)] = 1

            train_data[user_id] = {
                'input_mask': train_mask,
            }

            valid_data[user_id] = {
                'input_mask': train_mask,
                'loss_mask': valid_mask,
            }

            test_data[user_id] = {
                'input_mask': test_mask,
                'train_valid_mask': train_valid_mask,
            }
        logger.info("done")
        return train_data, valid_data, test_data

    def preprocess(self) -> pd.DataFrame:
        '''
           output: pivot table (row: user, col: user-specific vector + item set, values: binary preference) 
        '''
        logger.info("start preprocessing...")
        # load df
        df = self._load_df()
        # transform df into training set
        training_set = self._transform_into_training_set(df)
        logger.info("done")
        return training_set 

    def _load_df(self):
        logger.info("load df...")
        return pd.read_csv(os.path.join(self.cfg.data_dir, 'yelp_interactions.tsv'), sep='\t', index_col=False)

    def _transform_into_training_set(self, df):
        logger.info("transform df into training set...")
        item_inputs = df.pivot_table(index='user_id', columns=['business_id'], values=['rating'])
        training_set = item_inputs.droplevel(0, 1)

        # rating -> binary preference
        training_set = training_set.fillna(0)
        training_set = training_set.mask(training_set > 0, 1)

        training_set = training_set.reset_index()
        training_set.index.name = None

        return training_set
