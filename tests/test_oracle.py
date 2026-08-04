from atlas.oracle import CachedOracle


def test_cached_oracle_counts_unique_queries():
    oracle = CachedOracle(lambda seq: sum(seq) % 2)

    assert oracle.query((1, 0)) == 1
    assert oracle.query((1, 0)) == 1
    assert oracle.query_count == 1
    assert oracle.cache_hits == 1
