import os
from loguru import logger
import pandas as pd
from .data_pipeline import DataPipeline

class S3RecDataPipeline(DataPipeline):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_users = None
        self.num_items = None

    def split(self):
        return None
    
    def preprocess(self) -> pd.DataFrame:
        '''
           output: pivot table (row: user, col: user-specific vector + item set, values: binary preference) 
        '''
        logger.info("start preprocessing...")
        # load df
        df = self._load_df()
        # set num items and num users
        self._set_num_items_and_num_users(df)

        # user 별로 item sequence 뽑아야돼
        # train_pos_df = train_df.groupby('user_id').agg({'business_id': [('pos_items', list)]}).droplevel(0, 1)
        df = df.groupby(['user_id']).agg({'business_id': [('behaviors', list)]}).droplevel(0, 1)
        # logger.info(f"after groupby: {df.head()}")

        logger.info("done")
        return df 

    def _load_df(self):
        logger.info("load df...")
        return pd.read_csv(os.path.join(self.cfg.data_dir, 'yelp_interactions.tsv'), sep='\t', index_col=False)
    

    def _set_num_items_and_num_users(self, df):
        self.num_items = df.business_id.nunique()
        self.num_users = df.user_id.nunique()
