"""
Unit tests cho mô hình hóa bài toán JSP.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problem.job import Task, Job, Machine, JSPInstance
from src.problem.generator import generate_random_instance, generate_benchmark_instance
from src.problem.encoder import Encoder, decode_sequence_to_makespan


class TestTask:
    def test_repr(self):
        t = Task(job_id=0, machine_id=1, processing_time=5)
        assert "J0" in repr(t)
        assert "M1" in repr(t)

    def test_reset(self):
        t = Task(0, 0, 3)
        t.start_time = 5
        t.end_time = 8
        t.reset()
        assert t.start_time is None
        assert t.end_time is None


class TestJob:
    def test_total_processing_time(self):
        tasks = [Task(0, 0, 3), Task(0, 1, 4), Task(0, 2, 5)]
        job = Job(0, tasks)
        assert job.total_processing_time == 12

    def test_num_operations(self):
        tasks = [Task(0, m, 1) for m in range(4)]
        job = Job(0, tasks)
        assert job.num_operations == 4

    def test_reset_propagates(self):
        tasks = [Task(0, 0, 3), Task(0, 1, 2)]
        job = Job(0, tasks)
        job.tasks[0].start_time = 0
        job.reset()
        assert job.tasks[0].start_time is None


class TestJSPInstance:
    def test_lower_bound(self):
        instance = generate_random_instance(4, 3, seed=0)
        lb = instance.lower_bound
        assert lb > 0

    def test_reset(self):
        instance = generate_random_instance(3, 2, seed=1)
        seq = list(range(instance.num_jobs)) * instance.num_machines
        decode_sequence_to_makespan(seq, instance)
        instance.reset()
        for job in instance.jobs:
            for task in job.tasks:
                assert task.start_time is None


class TestGenerator:
    def test_random_instance_shape(self):
        inst = generate_random_instance(5, 3, seed=42)
        assert inst.num_jobs == 5
        assert inst.num_machines == 3
        assert len(inst.jobs) == 5
        for job in inst.jobs:
            assert job.num_operations == 3

    def test_reproducibility(self):
        inst1 = generate_random_instance(6, 3, seed=99)
        inst2 = generate_random_instance(6, 3, seed=99)
        for j in range(6):
            for k in range(3):
                assert inst1.jobs[j].tasks[k].processing_time == inst2.jobs[j].tasks[k].processing_time

    def test_benchmark_ft06(self):
        inst = generate_benchmark_instance("ft06")
        assert inst.num_jobs == 6
        assert inst.num_machines == 6
        assert inst.name == "ft06"

    def test_benchmark_invalid(self):
        with pytest.raises(ValueError):
            generate_benchmark_instance("nonexistent")


class TestEncoder:
    def test_spv_length(self):
        import numpy as np
        pos = np.random.rand(12)
        seq = Encoder.spv_decode(pos, n_jobs=4, n_machines=3)
        assert len(seq) == 12

    def test_spv_values(self):
        import numpy as np
        pos = np.array([0.3, 0.1, 0.9, 0.5, 0.7, 0.2])
        seq = Encoder.spv_decode(pos, n_jobs=3, n_machines=2)
        assert all(0 <= v < 3 for v in seq)

    def test_makespan_positive(self):
        inst = generate_random_instance(4, 3, seed=42)
        seq = Encoder.random_permutation(4, 3)
        makespan = decode_sequence_to_makespan(seq, inst)
        assert makespan > 0

    def test_makespan_lower_bound(self):
        inst = generate_random_instance(4, 3, seed=42)
        seq = Encoder.random_permutation(4, 3)
        makespan = decode_sequence_to_makespan(seq, inst)
        assert makespan >= inst.lower_bound
