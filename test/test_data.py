from ..data.data_preprocess_J import YelpData

def test_instantiate_YelpData():
    yelp_2019 = YelpData()
    assert len(yelp_2019.data) == 907_284

def test_filter_YelpData():
    yelp_2019 = YelpData()
    filtered = yelp_2019._filtered(yelp_2019.data)
    assert len(filtered) == 207_952

def test_encode_YelpData():
    yelp_2019 = YelpData()
    encoded = yelp_2019._encoded(yelp_2019.data)
    assert encoded['user_id'].dtype == 'int64'
    assert encoded['business_id'].dtype == 'int64'
