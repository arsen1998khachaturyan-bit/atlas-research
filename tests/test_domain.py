from atlas.domain import all_binary_sequences


def test_binary_domain_size():
    assert len(all_binary_sequences(4)) == 31
