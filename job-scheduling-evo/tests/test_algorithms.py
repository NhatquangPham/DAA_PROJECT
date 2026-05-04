"""
Integration tests — kiểm tra toàn bộ pipeline cho tất cả thuật toán.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problem.generator import generate_random_instance
from src.algorithms.pso import PSOScheduler, PSOConfig
from src.algorithms.ga import GAScheduler, GAConfig
from src.algorithms.greedy import GreedyScheduler
from src.algorithms.backtracking import BacktrackingScheduler
from src.algorithms.dp import DPScheduler
from src.algorithms.branch_bound import BranchBoundScheduler


TINY_INSTANCE = generate_random_instance(4, 2, seed=42)


class TestGreedyScheduler:
    def test_lpt_runs(self):
        r = GreedyScheduler(TINY_INSTANCE, "LPT").run()
        assert r.best_makespan > 0
        assert r.elapsed_time >= 0

    def test_spt_runs(self):
        r = GreedyScheduler(TINY_INSTANCE, "SPT").run()
        assert r.best_makespan > 0

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            GreedyScheduler(TINY_INSTANCE, "INVALID")

    def test_algorithm_name(self):
        r = GreedyScheduler(TINY_INSTANCE, "LPT").run()
        assert "Greedy" in r.algorithm_name


class TestBacktrackingScheduler:
    def test_tiny_instance(self):
        inst = generate_random_instance(3, 2, seed=42)
        r = BacktrackingScheduler(inst, time_limit=10.0).run()
        assert r.best_makespan >= inst.lower_bound

    def test_algorithm_name(self):
        inst = generate_random_instance(3, 2, seed=0)
        r = BacktrackingScheduler(inst, time_limit=5.0).run()
        assert r.algorithm_name == "Backtracking"


class TestDPScheduler:
    def test_small_instance(self):
        inst = generate_random_instance(5, 2, seed=42)
        r = DPScheduler(inst, time_limit=30.0).run()
        assert r.best_makespan > 0

    def test_too_large_fallback(self):
        inst = generate_random_instance(25, 3, seed=42)
        r = DPScheduler(inst).run()
        assert r.best_makespan > 0


class TestBranchBoundScheduler:
    def test_tiny_instance(self):
        inst = generate_random_instance(3, 2, seed=42)
        r = BranchBoundScheduler(inst, time_limit=10.0).run()
        assert r.best_makespan >= inst.lower_bound

    def test_algorithm_name(self):
        inst = generate_random_instance(3, 2, seed=0)
        r = BranchBoundScheduler(inst, time_limit=5.0).run()
        assert "Branch" in r.algorithm_name


class TestAllAlgorithmsConsistency:
    """
    Kiểm tra tính nhất quán: tất cả thuật toán trả về makespan ≥ lower bound.
    """

    def test_all_above_lower_bound(self):
        inst = generate_random_instance(4, 2, seed=123)
        lb = inst.lower_bound

        schedulers = [
            PSOScheduler(inst, PSOConfig(n_particles=10, max_iter=20, seed=0)),
            GAScheduler(inst, GAConfig(pop_size=20, generations=20, seed=0)),
            GreedyScheduler(inst, "LPT"),
            GreedyScheduler(inst, "SPT"),
            BacktrackingScheduler(inst, time_limit=5.0),
        ]

        for sched in schedulers:
            result = sched.run()
            assert result.best_makespan >= lb, (
                f"{result.algorithm_name} trả về makespan ({result.best_makespan}) "
                f"thấp hơn lower bound ({lb})"
            )

    def test_greedy_faster_than_pso(self):
        """Greedy luôn nhanh hơn PSO (deterministic O(n log n))."""
        inst = generate_random_instance(20, 5, seed=42)
        g_result = GreedyScheduler(inst, "LPT").run()
        pso_result = PSOScheduler(inst, PSOConfig(n_particles=20, max_iter=50, seed=42)).run()
        assert g_result.elapsed_time < pso_result.elapsed_time
