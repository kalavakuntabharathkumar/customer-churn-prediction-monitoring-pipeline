from src.data import make_dataset

def test_synthetic_data_contract():
    df = make_dataset(500, 1)
    assert len(df) == 500
    assert set(df.churn.unique()).issubset({0,1})
    assert df.isna().sum().sum() == 0
