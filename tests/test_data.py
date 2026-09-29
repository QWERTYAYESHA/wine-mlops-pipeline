from src.data import load_and_split_data


def test_wine_data_pipeline():
    X_train, X_test, y_train, y_test = load_and_split_data()

    assert X_train.shape[1] == 13
    assert X_test.shape[1] == 13

    assert len(X_train) == 142
    assert len(X_test) == 36

    assert len(y_train) == 142
    assert len(y_test) == 36

    assert not X_train.dtype == object
    assert not X_test.dtype == object
