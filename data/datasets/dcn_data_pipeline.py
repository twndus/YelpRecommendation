import os

import pandas as pd
from loguru import logger

from .mf_data_pipeline import MFDataPipeline

class DCNDatapipeline(MFDataPipeline):
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

        # load item attributes
        self.item2attributes = self._load_attributes()

        logger.info("done")

        return df

    def _load_attributes(self):
        logger.info("load item2attributes...")
        df = pd.read_json(os.path.join(self.cfg.data_dir, 'yelp_item2attributes.json')).transpose()
        self.attributes_count = [df.categories.explode().nunique(), df.statecity.nunique()]

        # The item category #0 is reserved for null embedding.
        # Pad the category sequence to ensure a fixed input length for all items.
        df.categories = self._pad_sequences_in_df(df.categories, df.categories.apply(len).max())
        df.categories = df.categories.apply(lambda x: [y+1 for y in x])

        return df.transpose().to_dict()

    def _pad_sequences_in_df(self, series, max_len, padding_value=-1):
        def pad_sequence(seq, max_len, padding_value):
            return seq + [padding_value] * (max_len - len(seq)) if len(seq) < max_len else seq
        
        padded_sequences = series.apply(lambda x: pad_sequence(x, max_len, padding_value))
        return padded_sequences
