import os, json

import pandas as pd

from sklearn.preprocessing import OrdinalEncoder

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

class YelpPreprocessPipe:

    def __init__(self, cfg):
        
        self.cfg = cfg
        os.makedirs(self.cfg.result_dir, exist_ok=True)

        self.user2id = None
        self.id2user = None

        self.item2id = None 
        self.id2item = None

        logger.info('YelpPreprocessPipe instanciated')

    def run(self):
        review_df = self._read_yelp_json('review')
        logger.info(f'원본 리뷰 데이터: {review_df.shape}')

        review_df = self._filter_by_min_interactions(review_df, self.cfg.min_interactions)
        logger.info(f'필터링 후 데이터: {review_df.shape}')

        review_df: pd.DataFrame = self._id_mapping(review_df)
        # logger.info(f"review df dtypes: {review_df.dtypes}")
        review_df = review_df.sort_values(['date'])
        # logger.info(f"after order by: {review_df[review_df.user_id == review_df.iloc[0].user_id].head()}")
        review_df = review_df[['user_id', 'business_id', 'stars']].rename(columns={'stars':'rating'})
        self._save_interactions(review_df)

#        user_df = self._read_yelp_json('user')
#        users2attributes = self._create_users2attributes(user_df, review_df)
#        self._save_entities2attributes(users2attributes, 'user')

        item_df = self._read_yelp_json('business')
        items2attributes = self._create_items2attributes(item_df, review_df)
        self._save_entities2attributes(items2attributes, 'item')

    def _read_yelp_json(self, datatype, query: str=None):
        logger.info(f"load {datatype} raw data ...")

        reader = pd.read_json(
            f'{self.cfg.data_dir}/yelp_academic_dataset_{datatype}.json', 
            orient="records", lines=True, chunksize=10000)

        target_dfs = []
        for raw_df in reader:
            if datatype == 'review':
                target_df = raw_df.query(
                    f"`date` >= '{self.cfg.start_date}' and `date` <= '{self.cfg.end_date}'")
            else:
                target_df = raw_df
            if target_df.shape[0] > 0:
                target_dfs.append(target_df)

        logger.info(f"done...")
        return pd.concat(target_dfs)

    def _filter_by_min_interactions(self, df, min_interactions=5):
        logger.info(f"filter users and items having {min_interactions} or more interactions ...")
        user_ids_under_5, business_ids_under_5 = [0], [0]

        while len(user_ids_under_5) > 0 or len(business_ids_under_5) > 0:
            user_ids_under_5 = df.user_id.value_counts()[df.user_id.value_counts() < 5].index
            business_ids_under_5 = df.business_id.value_counts()[df.business_id.value_counts() < 5].index

            df = df[~df.user_id.isin(user_ids_under_5)]
            df = df[~df.business_id.isin(business_ids_under_5)]

        logger.info(f"done...")
        return df

    def _id_mapping(self, df):
        logger.info(f"map user_id and business_id to new numeric ids...")

        self.user2id = {user_id: id_ for id_, user_id in enumerate(df['user_id'].unique())}
        self.id2user = {id_: user_id for user_id, id_ in self.user2id.items()}

        self.item2id = {item_id: id_ for id_, item_id in enumerate(df['business_id'].unique())}
        self.id2item = {id_: item_id for item_id, id_ in self.item2id.items()}

        df['user_id'] = df['user_id'].map(self.user2id)
        df['business_id'] = df['business_id'].map(self.item2id)
       
        logger.info(f"done...")
        return df

    def _create_users2attributes(self, user_df, review_df):
        logger.info(f"create user2attributes...")
        users2attributes = None

        user_df = user_df[user_df.user_id.isin(self.user2id.keys())]
        user_df['user_id'] = user_df['user_id'].map(self.user2id)

        logger.info(f"done...")
        return users2attributes

    def _create_items2attributes(self, item_df, review_df):
        logger.info(f"create item2attributes...")
        items2attributes = None

        item_df = item_df[item_df.business_id.isin(self.item2id.keys())]
        item_df['business_id'] = item_df['business_id'].map(self.item2id)

        # categories 810
        categories = item_df.categories.str.split(', ').explode().unique()
        categories2id = {category: id_ for id_, category in enumerate(categories)}
        id2categories = {id_:category for category, id_ in categories2id.items()}

        # statecity 462
        item_df['statecity'] = item_df.state + '_' + item_df.city
        statecities = item_df.statecity.unique()
        statecities2id = {statecity: id_ for id_, statecity in enumerate(statecities)}
        id2statecities = {id_:statecity for statecity, id_ in statecities2id.items()}

        # encoding
        item_df['category_encoded'] = item_df.categories.str.split(', ').apply(lambda x: [categories2id[y] for y in x])
        item_df['statecity_encoded'] = item_df.statecity.apply(lambda x: statecities2id[x])

        items2attributes = {
            int(row['business_id']): {'categories': row['category_encoded'], 'statecity': row['statecity_encoded']}\
            for i, row in item_df.iterrows()
        }

        logger.info(f"done...")
        return items2attributes

    def _save_interactions(self, interactions):
        logger.info(f"save interactions...")
        interactions.to_csv(os.path.join(self.cfg.result_dir, 'yelp_interactions.tsv'), index=False, sep='\t')
        logger.info(f"done...")

    def _save_entities2attributes(self, entities2attributes, entity_name):
        logger.info(f"save {entity_name}2attributes...")

        filename = os.path.join(self.cfg.result_dir, f'yelp_{entity_name}2attributes.json')
        with open(filename, 'w') as f:
            json.dump(entities2attributes, f)
        
        logger.info(f"done...")

@hydra.main(version_base=None, config_path="../configs", config_name="data_preprocess")
def main(cfg: OmegaConf):
    ypp = YelpPreprocessPipe(cfg)
    ypp.run()


if __name__ == '__main__':
    main()
