"""
Unit tests cho thuật toán PSO.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problem.generator import generate_random_instance, generate_benchmark_instance
from src.algorithms.pso import PSOScheduler, PSOConfig
from src.algorithms.base import AlgorithmResult


class TestPSOConfig:
    def test_default_config(self):
        cfg = PSOConfig()
        assert cfg.n_particles == 50
        assert cfg.max_iter == 200
        assert 0 < cfg.w_min < cfg.w_max <= 1.0
        assert cfg.c1 > 0 and cfg.c2 > 0

    def test_custom_config(self):
        cfg = PSOConfig(n_particles=20, max_iter=50, seed=0)
        assert cfg.n_particles == 20
        assert cfg.max_iter == 50


class TestPSOScheduler:
    @pytest.fixture
    def small_instance(self):
        return generate_random_instance(4, 3, seed=42)

    @pytest.fixture
    def fast_config(self):
        return PSOConfig(n_particles=10, max_iter=20, seed=42)

    def test_returns_result_type(self, small_instance, fast_config):
        scheduler = PSOScheduler(small_instance, fast_config)
        result = scheduler.run()
        assert isinstance(result, AlgorithmResult)

    def test_makespan_positive(self, small_instance, fast_config):
        result = PSOScheduler(small_instance, fast_config).run()
        assert result.best_makespan > 0

    def test_makespan_above_lower_bound(self, small_instance, fast_config):
        result = PSOScheduler(small_instance, fast_config).run()
        assert result.best_makespan >= small_instance.lower_bound

    def test_elapsed_time_positive(self, small_instance, fast_config):
        result = PSOScheduler(small_instance, fast_config).run()
        assert result.elapsed_time > 0

    def test_history_decreasing(self, small_instance, fast_config):
        """Makespan tốt nhất không được tăng theo thời gian."""
        result = PSOScheduler(small_instance, fast_config).run()
        history = result.history
        for i in range(1, len(history)):
            assert history[i] <= history[i-1] + 1e-9, (
                f"History không đơn điệu tại vòng {i}: {history[i-1]} → {history[i]}"
            )

    def test_history_length(self, small_instance, fast_config):
        result = PSOScheduler(small_instance, fast_config).run()
        assert len(result.history) == fast_config.max_iter + 1

    def test_algorithm_name(self, small_instance, fast_config):
        result = PSOScheduler(small_instance, fast_config).run()
        assert result.algorithm_name == "PSO"

    def test_reproducibility(self, small_instance):
        cfg = PSOConfig(n_particles=10, max_iter=20, seed=99)
        r1 = PSOScheduler(small_instance, cfg).run()
        r2 = PSOScheduler(small_instance, cfg).run()
        assert r1.best_makespan == r2.best_makespan

    def test_ft06_quality(self):
        """PSO phải tìm makespan ≤ 2× optimal (55) cho ft06."""
        instance = generate_benchmark_instance("ft06")
        cfg = PSOConfig(n_particles=30, max_iter=100, seed=42)
        result = PSOScheduler(instance, cfg).run()
        assert result.best_makespan <= 55 * 2.0, (
            f"PSO makespan ({result.best_makespan}) quá cao so với optimal (55)"
        )
