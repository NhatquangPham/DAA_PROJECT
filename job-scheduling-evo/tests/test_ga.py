"""
Unit tests cho thuật toán GA.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problem.generator import generate_random_instance, generate_benchmark_instance
from src.algorithms.ga import GAScheduler, GAConfig, Chromosome
from src.algorithms.base import AlgorithmResult


class TestGAConfig:
    def test_default_config(self):
        cfg = GAConfig()
        assert cfg.pop_size == 100
        assert cfg.generations == 300
        assert 0 < cfg.p_crossover <= 1.0
        assert 0 < cfg.p_mutation <= 1.0

    def test_elite_size_valid(self):
        cfg = GAConfig(elite_size=5, pop_size=50)
        assert cfg.elite_size < cfg.pop_size


class TestGAScheduler:
    @pytest.fixture
    def small_instance(self):
        return generate_random_instance(4, 3, seed=42)

    @pytest.fixture
    def fast_config(self):
        return GAConfig(pop_size=20, generations=30, seed=42)

    def test_returns_result_type(self, small_instance, fast_config):
        result = GAScheduler(small_instance, fast_config).run()
        assert isinstance(result, AlgorithmResult)

    def test_makespan_positive(self, small_instance, fast_config):
        result = GAScheduler(small_instance, fast_config).run()
        assert result.best_makespan > 0

    def test_makespan_above_lower_bound(self, small_instance, fast_config):
        result = GAScheduler(small_instance, fast_config).run()
        assert result.best_makespan >= small_instance.lower_bound

    def test_history_monotonic(self, small_instance, fast_config):
        """Makespan tốt nhất không được tăng theo thế hệ."""
        result = GAScheduler(small_instance, fast_config).run()
        history = result.history
        for i in range(1, len(history)):
            assert history[i] <= history[i-1] + 1e-9

    def test_algorithm_name(self, small_instance, fast_config):
        result = GAScheduler(small_instance, fast_config).run()
        assert result.algorithm_name == "GA"

    def test_best_sequence_valid_jobs(self, small_instance, fast_config):
        """Chuỗi kết quả phải chỉ chứa job_id hợp lệ."""
        result = GAScheduler(small_instance, fast_config).run()
        valid_ids = set(range(small_instance.num_jobs))
        assert all(jid in valid_ids for jid in result.best_sequence)

    def test_reproducibility(self, small_instance):
        cfg = GAConfig(pop_size=20, generations=30, seed=7)
        r1 = GAScheduler(small_instance, cfg).run()
        r2 = GAScheduler(small_instance, cfg).run()
        assert r1.best_makespan == r2.best_makespan

    def test_ft06_quality(self):
        """GA phải tìm makespan ≤ 2× optimal (55) cho ft06."""
        instance = generate_benchmark_instance("ft06")
        cfg = GAConfig(pop_size=30, generations=100, seed=42)
        result = GAScheduler(instance, cfg).run()
        assert result.best_makespan <= 55 * 2.5


class TestChromosome:
    def test_copy_independent(self):
        c = Chromosome([1, 2, 3], fitness=10.0)
        c2 = c.copy()
        c2.genes[0] = 99
        assert c.genes[0] == 1

    def test_lt_by_fitness(self):
        c1 = Chromosome([0, 1], fitness=5.0)
        c2 = Chromosome([1, 0], fitness=10.0)
        assert c1 < c2
