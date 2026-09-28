from sklearn.datasets import load_wine


def test_wine_dataset():
    wine = load_wine()

    assert wine.data.shape == (178, 13)
    assert len(wine.target) == 178
    assert len(set(wine.target)) == 3
