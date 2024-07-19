import os
from loguru import logger
import pandas as pd
from .data_pipeline import DataPipeline

class S3RecDataPipeline(DataPipeline):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_users = None
        self.num_items = None


    def split(self, df: pd.DataFrame):
        # train X: [:-3] y: -3
        train_df_X = df.behaviors.apply(lambda row: row[: -3]).rename('X')
        train_df_Y = df.behaviors.apply(lambda row: row[-3]).rename('y')

        # valid X: [:-2] y: -2 
        valid_df_X = df.behaviors.apply(lambda row: row[: -2]).rename('X')
        valid_df_Y = df.behaviors.apply(lambda row: row[-2]).rename('y')

        # test X: [:-1] y: -1 
        test_df_X = df.behaviors.apply(lambda row: row[: -1]).rename('X')
        test_df_Y = df.behaviors.apply(lambda row: row[-1]).rename('y')

        # pre-padding for input sequence X
        train_df_X = self._adjust_seq_len(train_df_X)
        valid_df_X = self._adjust_seq_len(valid_df_X)
        test_df_X = self._adjust_seq_len(test_df_X)

        return pd.concat([df, train_df_X, train_df_Y], axis=1),\
            pd.concat([df, valid_df_X, valid_df_Y], axis=1),\
            pd.concat([df, test_df_X, test_df_Y], axis=1)


    def _adjust_seq_len(self, df):
        def _adjust_seq_len_by_user(row):
            if len(row) > self.cfg.max_seq_len:
                row = row[-self.cfg.max_seq_len:]
            elif len(row) < self.cfg.max_seq_len:
                row = [-1] * (self.cfg.max_seq_len - len(row)) + row
            # item 0: pad, item starts from 1
            return [e+1 for e in row]
        
        df = df.apply(_adjust_seq_len_by_user)
        return df
    
    
    def preprocess(self) -> pd.DataFrame:
        '''
           output: pivot table (row: user, col: user-specific vector + item set, values: binary preference) 
        '''
        logger.info("start preprocessing...")
        # load df
        df = self._load_df()
        # set num items and num users
        self._set_num_items_and_num_users(df)

        # group by user_id
        df = df.groupby(['user_id']).agg({'business_id': [('behaviors', list)]}).droplevel(0, 1)

        # load attributes
        self.item2attributes = self._load_attributes()

        logger.info("done")
        return df 


    def _load_df(self):
        logger.info("load df...")
        return pd.read_csv(os.path.join(self.cfg.data_dir, 'yelp_interactions.tsv'), sep='\t', index_col=False)
    

    def _load_attributes(self):
        logger.info("load item2attributes...")
        df = pd.read_json(os.path.join(self.cfg.data_dir, 'yelp_item2attributes.json')).transpose()
        self.attributes_count = df.categories.explode().nunique()

        return df.drop(columns=['statecity']).transpose().to_dict()
    

    def _set_num_items_and_num_users(self, df):
        self.num_items = df.business_id.nunique()
        self.num_users = df.user_id.nunique()
