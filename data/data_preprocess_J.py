import os
import json
import pandas as pd
from tqdm import tqdm

class YelpData:
    def __init__(self, 
            data_name: str='yelp_academic_dataset_review.json',
            target_year: int=2019) -> None:
        super().__init__()
        self.data: pd.DataFrame = self._loaded(data_name, target_year)

    def _loaded(self, data_name: str, target_year: int) -> pd.DataFrame:
        saved_dir = os.path.join(os.path.dirname(__file__), data_name.replace('.json', f'_{target_year}.csv'))
        if os.path.exists(saved_dir):
            return pd.read_csv(saved_dir)
        
        data_dir = os.path.join(os.path.dirname(__file__), data_name)
        with open(data_dir, 'r') as file:
            data = list()
            for line in tqdm(file):
                json_data = json.loads(line)
                year = json_data['date'][:4]
                if int(year) == target_year:
                    data.append(json_data)
        df: pd.DataFrame = pd.DataFrame(data)
        df.to_csv(saved_dir, index=False)
        return df
    
    def _filtered(self, data: pd.DataFrame, threshold: int=5) -> pd.DataFrame:
        # calculate interaction count per user and per item
        count_per_user, count_per_item = self._count_per_user_and_item(data, threshold)

        while (count_per_user.sum() > 0) or (count_per_item.sum() > 0):
            # remove user under threshold
            data = data[data['user_id'].isin(count_per_user.keys()) == False]
            # remove item under threshold
            data = data[data['business_id'].isin(count_per_item.keys()) == False]
            count_per_user, count_per_item = self._count_per_user_and_item(data, threshold)
        
        return data
    
    def _count_per_user_and_item(self, data: pd.DataFrame, threshold: int) -> tuple[pd.Series]:
        count_per_user = data['user_id'].value_counts()
        count_per_item = data['business_id'].value_counts()

        count_per_user = count_per_user[count_per_user < threshold]
        count_per_item = count_per_item[count_per_item < threshold]

        return count_per_user, count_per_item
    
    def _encoded(self, data: pd.DataFrame) -> pd.DataFrame:
        encoded: pd.DataFrame = data[['user_id', 'business_id', 'stars']]
        
        # encode user_id
        user2idx = {value:idx for idx, value in enumerate(set(encoded['user_id']))}
        encoded['user_id'] = encoded['user_id'].map(user2idx)

        # encode business_id
        business2idx = {value:idx for idx, value in enumerate(set(encoded['business_id']))}
        encoded['business_id'] = encoded['business_id'].map(business2idx)

        return encoded
