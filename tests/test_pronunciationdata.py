from sonorous.pronunciationdata import load_pronunciations


def test_load_data():
    """This is partially just an integregation test to make sure the CMU Dict data
    doesn't change from under my feet.
    """
    df = load_pronunciations()
    assert len(df) == 125801
    cat_row = df.loc["cat"]
    assert cat_row.pronunciation == ("K", "AE1", "T")
    assert cat_row.length == 3


def test_load_data_no_stress():
    df = load_pronunciations(include_stress=False)
    assert len(df) == 125801
    cat_row = df.loc["cat"]
    assert cat_row.pronunciation == ("K", "AE", "T")
    assert cat_row.length == 3
