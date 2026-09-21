import pandas as pd

from microevolution.vowel_space import date_centroids, endpoint_change


def test_centroids_are_chronological_medians_and_change_is_interpreted():
    tokens = pd.DataFrame([
        {"effective_date": pd.Timestamp("2024-02-01"), "f1_hz": 620, "f2_hz": 1400, "token_id": "c"},
        {"effective_date": pd.Timestamp("2020-01-01"), "f1_hz": 500, "f2_hz": 1200, "token_id": "a"},
        {"effective_date": pd.Timestamp("2020-01-01"), "f1_hz": 520, "f2_hz": 1240, "token_id": "b"},
    ])

    centroids = date_centroids(tokens)
    assert list(centroids["f1_hz"]) == [510, 620]
    assert list(centroids["tokens"]) == [2, 1]
    assert endpoint_change(centroids) == {
        "first_date": "2020-01-01", "last_date": "2024-02-01",
        "first_tokens": 2, "last_tokens": 1,
        "delta_f1": 110.0, "delta_f2": 180.0,
        "height": "lower", "backness": "fronter",
    }


def test_change_requires_two_dates():
    tokens = pd.DataFrame([
        {"effective_date": pd.Timestamp("2020-01-01"), "f1_hz": 500, "f2_hz": 1200, "token_id": "a"},
    ])
    assert endpoint_change(date_centroids(tokens)) is None
