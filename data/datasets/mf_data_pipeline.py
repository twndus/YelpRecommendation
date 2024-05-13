import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from loguru import logger

from .data_pipeline import DataPipeline

class MFDataPipeline(DataPipeline):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_items = None
        self.num_users = None

    def split(self, df):
        '''
           train_data: ((user_id, item_id, rating), ...)
        '''
        logger.info(f'start random user split...')
        train_df, valid_df, test_df = [], [], []

        for _, user_df in df.groupby('user_id'):
            if self.cfg.loss_name == 'pointwise':
                user_train_df, user_test_df = train_test_split(user_df, test_size=.2, stratify=user_df['rating'])
                user_train_df, user_valid_df = train_test_split(user_train_df, test_size=.25, stratify=user_train_df['rating'])
            else:
                user_train_df, user_test_df = train_test_split(user_df, test_size=.2)
                user_train_df, user_valid_df = train_test_split(user_train_df, test_size=.25)
            train_df.append(user_train_df)
            valid_df.append(user_valid_df)
            test_df.append(user_test_df)

        train_df = pd.concat(train_df).reset_index()
        valid_df = pd.concat(valid_df).reset_index()
        test_df = pd.concat(test_df).reset_index()

        train_pos_df = train_df.groupby('user_id').agg({'business_id': [('pos_items', list)]}).droplevel(0, 1)
        valid_pos_df = valid_df.groupby('user_id').agg({'business_id': [('pos_items', list)]}).droplevel(0, 1)
        train_valid_pos_df = pd.concat([train_df, valid_df], axis=0).groupby('user_id').agg({'business_id': [('pos_items', list)]}).droplevel(0, 1)
        test_pos_df = test_df.groupby('user_id').agg({'business_id': [('pos_items', list)]}).droplevel(0, 1)

        train_data = pd.merge(train_df, train_pos_df, left_on='user_id', right_on='user_id', how='left')
        valid_data = pd.merge(valid_df, valid_pos_df, left_on='user_id', right_on='user_id', how='left')
        valid_eval_data = pd.merge(valid_pos_df, train_pos_df.rename(columns={'pos_items': 'mask_items'}), left_on='user_id', right_on='user_id', how='left')
        test_eval_data = pd.merge(test_pos_df, train_valid_pos_df.rename(columns={'pos_items': 'mask_items'}), left_on='user_id', right_on='user_id', how='left')
        
        return train_data, valid_data, valid_eval_data, test_eval_data

    def preprocess(self) -> pd.DataFrame:
        '''
           output: pivot table (row: user, col: user-specific vector + item set, values: binary preference) 
        '''
        logger.info("start preprocessing...")
        # load df
        df = self._load_df()
        # set num items and num users
        self._set_num_items_and_num_users(df)
        # negative sampling
        if self.cfg.loss_name == 'pointwise':
            df = self._negative_sampling(df, self.cfg.neg_times)
        logger.info("done")
        return df 

    def _load_df(self):
        logger.info("load df...")
        return pd.read_csv(os.path.join(self.cfg.data_dir, 'yelp_interactions.tsv'), sep='\t', index_col=False)

    def _set_num_items_and_num_users(self, df):
        self.num_items = df.business_id.nunique()
        self.num_users = df.user_id.nunique()

    def _negative_sampling(self, df: pd.DataFrame, neg_times: 5) -> pd.DataFrame:
        logger.info(f"negative sampling...")
        logger.info(f"before neg sampling: {df.shape}")
        all_items = df.business_id.unique()
        
        df['rating'] = 1
        neg_data = []
        for _, user_df in df.groupby('user_id'): 
            user_id = user_df.user_id.values[0]
            pos_items = user_df.business_id.unique()
            neg_items = []
            while len(neg_items) < len(pos_items)*neg_times:
                neg_item = np.random.choice(all_items)
                if (neg_item in pos_items) or (neg_item in neg_items): continue
                neg_items.append(neg_item)
            neg_data.extend([[user_id, neg_item, 0] for neg_item in neg_items])
        
        df = pd.concat([df, pd.DataFrame(neg_data, columns=df.columns)], axis=0)
        df = df.sample(frac=1).reset_index(drop=True) 
        logger.info(f"after neg sampling: {df.shape}")
        logger.info(f"done...")
        return df
