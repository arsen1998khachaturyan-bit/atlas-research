from atlas.domain import all_binary_sequences
from atlas.hypothesis import behavior_matrix
from atlas.learners import (
    ActiveInformationGainLearner,
    PassiveLearner,
    RandomLearner,
)
from atlas.learners.common import fidelity
from atlas.programs import Program


def test_learners_recover_simple_hypothesis():
    domain = all_binary_sequences(4)

    programs = [
        Program("always_false", 1, lambda s: 0),
        Program("parity", 2, lambda s: sum(s) % 2),
        Program("contains_one", 2, lambda s: int(1 in s)),
    ]

    outputs = behavior_matrix(programs, domain)
    task_index = 1

    learners = [
        PassiveLearner(domain, seed=1),
        RandomLearner(domain, seed=1),
        ActiveInformationGainLearner(domain),
    ]

    for learner in learners:
        result = learner.fit(outputs, task_index)
        assert fidelity(outputs, result.recovered_index, task_index) == 1.0
