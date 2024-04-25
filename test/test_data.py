import pytest
from ..data.data_preprocess_J import YelpData

def test_instantiate_YelpData():
    yelp_2019 = YelpData()
    assert len(yelp_2019.data) == 907_284

def test_filter_YelpData():
    yelp_2019 = YelpData()
    filtered = yelp_2019._filtered(yelp_2019.data)
    assert len(filtered) == 207_952