from eeg_emotion.splits import make_subject_split


def test_subject_split_is_disjoint_and_complete():
    ids = list(range(1, 33))
    split = make_subject_split(ids, seed=42)
    train = set(split.train)
    validation = set(split.validation)
    test = set(split.test)

    assert (len(train), len(validation), len(test)) == (24, 4, 4)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert train | validation | test == set(ids)

