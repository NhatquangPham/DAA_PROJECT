"""
Particle Swarm Optimization (PSO) cho bài toán Job Shop Scheduling.

Cài đặt PSO với:
- Mã hóa SPV (Smallest Position Value) để xử lý không gian rời rạc
- Trọng số quán tính giảm tuyến tính (linearly decreasing inertia weight)
- Giới hạn vận tốc (velocity clamping)
- Greedy seeding cho khởi tạo quần thể
"""
from __future__ import annotations
import time
import tracemalloc
import copy
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from src.problem.job import JSPInstance
from src.problem.encoder import Encoder, decode_sequence_to_makespan
from .base import BaseScheduler, AlgorithmResult


@dataclass
class PSOConfig:
    """
    Cấu hình tham số cho PSO.

    Attributes:
        n_particles : Kích thước quần thể (số hạt)
        max_iter    : Số vòng lặp tối đa
        w_max       : Trọng số quán tính ban đầu (pha khám phá)
        w_min       : Trọng số quán tính cuối (pha khai thác)
        c1          : Hệ số nhận thức (cognitive) — hút về pBest
        c2          : Hệ số xã hội (social) — hút về gBest
        v_max_ratio : Giới hạn vận tốc = v_max_ratio × (ub - lb)
        greedy_ratio: Tỷ lệ hạt được khởi tạo bằng greedy seeding
        seed        : Hạt giống ngẫu nhiên
    """
    n_particles: int = 50
    max_iter: int = 200
    w_max: float = 0.9
    w_min: float = 0.4
    c1: float = 2.0
    c2: float = 2.0
    v_max_ratio: float = 0.5
    greedy_ratio: float = 0.2
    seed: Optional[int] = 42


class Particle:
    """Đại diện cho một hạt (particle) trong đàn PSO."""

    def __init__(self, dim: int, lb: float, ub: float, rng: np.random.Generator) -> None:
        self.dim = dim
        self.lb = lb
        self.ub = ub

        # Khởi tạo vị trí và vận tốc ngẫu nhiên
        self.position: np.ndarray = rng.uniform(lb, ub, dim)
        self.velocity: np.ndarray = rng.uniform(-1.0, 1.0, dim) * (ub - lb) * 0.1

        # Bộ nhớ cá nhân
        self.pbest_position: np.ndarray = self.position.copy()
        self.pbest_fitness: float = float("inf")

    def update_velocity(
        self,
        gbest_position: np.ndarray,
        w: float,
        c1: float,
        c2: float,
        v_max: float,
        rng: np.random.Generator,
    ) -> None:
        """
        Cập nhật vận tốc theo phương trình PSO chuẩn:
        v(t+1) = w·v(t) + c1·r1·(pBest - x(t)) + c2·r2·(gBest - x(t))
        """
        r1 = rng.random(self.dim)
        r2 = rng.random(self.dim)

        cognitive = c1 * r1 * (self.pbest_position - self.position)
        social = c2 * r2 * (gbest_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

        # Giới hạn vận tốc (velocity clamping)
        self.velocity = np.clip(self.velocity, -v_max, v_max)

    def update_position(self) -> None:
        """Cập nhật vị trí và giới hạn trong [lb, ub]."""
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, self.lb, self.ub)

    def evaluate(self, instance: JSPInstance) -> float:
        """
        Đánh giá fitness (makespan) của hạt hiện tại.
        Sử dụng SPV để chuyển vị trí → hoán vị → makespan.
        """
        instance.reset()
        sequence = Encoder.spv_decode(self.position, instance.num_jobs, instance.num_machines)
        makespan = decode_sequence_to_makespan(sequence, instance)
        return float(makespan)

    def update_pbest(self, fitness: float) -> None:
        """Cập nhật vị trí tốt nhất cá nhân nếu fitness tốt hơn."""
        if fitness < self.pbest_fitness:
            self.pbest_fitness = fitness
            self.pbest_position = self.position.copy()


class PSOScheduler(BaseScheduler):
    """
    Bộ giải PSO cho bài toán Job Shop Scheduling.

    Quy trình:
    1. Khởi tạo đàn (swarm) với greedy seeding
    2. Đánh giá fitness ban đầu
    3. Lặp max_iter vòng:
       a. Cập nhật w tuyến tính
       b. Cập nhật vận tốc và vị trí từng hạt
       c. Đánh giá fitness
       d. Cập nhật pBest và gBest
    4. Trả về kết quả tốt nhất
    """

    def __init__(self, instance: JSPInstance, config: Optional[PSOConfig] = None) -> None:
        super().__init__(instance)
        self.config = config or PSOConfig()

    def run(self) -> AlgorithmResult:
        """Chạy PSO và trả về kết quả."""
        tracemalloc.start()
        t_start = time.perf_counter()

        result = self._run_pso()

        elapsed = time.perf_counter() - t_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result.elapsed_time = elapsed
        result.memory_kb = peak / 1024
        return result

    def _run_pso(self) -> AlgorithmResult:
        cfg = self.config
        inst = self.instance
        rng = np.random.default_rng(cfg.seed)

        dim = inst.num_jobs * inst.num_machines
        lb, ub = 0.0, 1.0
        v_max = cfg.v_max_ratio * (ub - lb)

        # Khởi tạo quần thể
        swarm = self._init_swarm(dim, lb, ub, rng)

        # Đánh giá fitness ban đầu
        gbest_fitness = float("inf")
        gbest_position = swarm[0].position.copy()
        gbest_sequence: List[int] = []

        for p in swarm:
            fitness = p.evaluate(inst)
            p.update_pbest(fitness)
            if fitness < gbest_fitness:
                gbest_fitness = fitness
                gbest_position = p.position.copy()
                gbest_sequence = Encoder.spv_decode(p.position, inst.num_jobs, inst.num_machines)

        history = [gbest_fitness]

        # Vòng lặp chính
        for iteration in range(cfg.max_iter):
            # Giảm trọng số quán tính tuyến tính
            w = cfg.w_max - (cfg.w_max - cfg.w_min) * iteration / cfg.max_iter

            for p in swarm:
                p.update_velocity(gbest_position, w, cfg.c1, cfg.c2, v_max, rng)
                p.update_position()
                fitness = p.evaluate(inst)
                p.update_pbest(fitness)

                if fitness < gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = p.position.copy()
                    gbest_sequence = Encoder.spv_decode(p.position, inst.num_jobs, inst.num_machines)

            history.append(gbest_fitness)

        return AlgorithmResult(
            algorithm_name="PSO",
            best_makespan=gbest_fitness,
            best_sequence=gbest_sequence,
            elapsed_time=0.0,
            history=history,
            is_optimal=False,
            n_iterations=cfg.max_iter,
        )

    def _init_swarm(self, dim: int, lb: float, ub: float, rng: np.random.Generator) -> List[Particle]:
        """
        Khởi tạo quần thể với greedy seeding.
        - greedy_ratio × n_particles hạt dựa trên heuristic greedy
        - Phần còn lại: khởi tạo ngẫu nhiên hoàn toàn
        """
        cfg = self.config
        n_greedy = max(1, int(cfg.n_particles * cfg.greedy_ratio))
        swarm = []

        # Greedy seeding: tạo vị trí gần với nghiệm greedy
        greedy_seq = Encoder.greedy_permutation(self.instance)
        greedy_pos = self._sequence_to_position(greedy_seq, dim, rng)

        for _ in range(n_greedy):
            p = Particle(dim, lb, ub, rng)
            noise = rng.normal(0, 0.05, dim)
            p.position = np.clip(greedy_pos + noise, lb, ub)
            swarm.append(p)

        # Khởi tạo ngẫu nhiên phần còn lại
        for _ in range(cfg.n_particles - n_greedy):
            swarm.append(Particle(dim, lb, ub, rng))

        return swarm

    @staticmethod
    def _sequence_to_position(sequence: List[int], dim: int, rng: np.random.Generator) -> np.ndarray:
        """
        Chuyển đổi chuỗi công việc → vector vị trí gần đúng trong [0, 1].
        Dùng thứ tự xuất hiện để xác định giá trị vị trí tương đối.
        """
        position = np.zeros(dim)
        n = len(sequence)
        for i, val in enumerate(sequence[:dim]):
            position[i] = (i + rng.uniform(0, 0.5)) / n
        rng.shuffle(position)
        return np.clip(position, 0.0, 1.0)
