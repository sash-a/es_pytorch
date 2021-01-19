import numpy as np

from src.utils.rankers import MultiObjectiveRanker, CenteredRanker


def test_moo_weighted_rank():
    evals = 10
    objectives = 2

    x = np.reshape(np.arange(evals * objectives), (-1, 2))

    centered_ranker = CenteredRanker()
    moo_ranker = MultiObjectiveRanker(centered_ranker, 0.5)  # mean

    ranked = moo_ranker._rank(x)
    expected = []
    for fits in x.T:
        expected += [centered_ranker._rank(fits)]
    assert (expected == ranked).all()

    moo_ranker.w = 0.1  # 0.1:0.9
    ranked = moo_ranker._rank(x)
    res = []
    for i, fits in enumerate(x.T):
        res.append(centered_ranker._rank(fits))
    expected = res[0] * 0.1 + res[1] * 0.9
    assert (expected == ranked).all()
