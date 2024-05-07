import os

import pandas as pd
from loguru import logger

from .data_pipeline import DataPipeline

class CDAEDataPipeline(DataPipeline):

    def __init__(self, cfg):
        super().__init__(cfg)

    def split(self, data):
        # observed data 만 나눔
        train_data, valid_data, test_data = None, None, None
        return train_data, valid_data, test_data

    def preprocess(self) -> pd.DataFrame:
        '''
           output: pivot table (row: user, col: user-specific vector + item set, values: binary preference) 
        '''
        # load df
        df = self._load_df()
        # transform df into training set
        training_set = self._transform_into_training_set(df)
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
