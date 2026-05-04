"""
Genetic Algorithm (GA) cho bài toán Job Shop Scheduling.

Cài đặt GA với:
- Mã hóa hoán vị (permutation encoding)
- Tournament Selection (chọn lọc giải đấu)
- Order Crossover – OX (lai ghép thứ tự)
- Swap Mutation + Inversion Mutation (đột biến)
- Elitism (tinh hoa)
- Greedy Seeding (khởi tạo có hướng dẫn)
"""
from __future__ import annotations
import time
import tracemalloc
import random
import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.problem.job import JSPInstance
from src.problem.encoder import decode_sequence_to_makespan, Encoder
from .base import BaseScheduler, AlgorithmResult


@dataclass
class GAConfig:
    """
    Cấu hình tham số cho GA.

    Attributes:
        pop_size        : Kích thước quần thể
        generations     : Số thế hệ tối đa
        p_crossover     : Xác suất lai ghép
        p_mutation      : Xác suất đột biến
        elite_size      : Số cá thể tinh hoa giữ lại mỗi thế hệ
        tournament_k    : Kích thước giải đấu (Tournament Selection)
        greedy_ratio    : Tỷ lệ khởi tạo bằng greedy seeding
        seed            : Hạt giống ngẫu nhiên
    """
    pop_size: int = 100
    generations: int = 300
    p_crossover: float = 0.85
    p_mutation: float = 0.05
    elite_size: int = 2
    tournament_k: int = 3
    greedy_ratio: float = 0.2
    seed: Optional[int] = 42


class Chromosome:
    """Đại diện cho một cá thể (chromosome) trong quần thể GA."""
    __slots__ = ("genes", "fitness")

    def __init__(self, genes: List[int], fitness: float = float("inf")) -> None:
        self.genes = genes
        self.fitness = fitness

    def copy(self) -> "Chromosome":
        return Chromosome(list(self.genes), self.fitness)

    def __lt__(self, other: "Chromosome") -> bool:
        return self.fitness < other.fitness

    def __repr__(self) -> str:
        return f"Chromosome(fitness={self.fitness:.2f})"


class GAScheduler(BaseScheduler):
    """
    Bộ giải GA cho bài toán Job Shop Scheduling.

    Quy trình:
    1. Khởi tạo quần thể (một phần greedy seeding)
    2. Đánh giá fitness cho toàn bộ quần thể
    3. Lặp `generations` vòng:
       a. Giữ lại elite_size cá thể tinh hoa
       b. Tournament selection để chọn cha mẹ
       c. OX crossover để tạo con cái
       d. Swap/Inversion mutation
       e. Thay thế quần thể cũ
    4. Trả về cá thể tốt nhất
    """

    def __init__(self, instance: JSPInstance, config: Optional[GAConfig] = None) -> None:
        super().__init__(instance)
        self.config = config or GAConfig()

    def run(self) -> AlgorithmResult:
        """Chạy GA và trả về kết quả."""
        tracemalloc.start()
        t_start = time.perf_counter()

        result = self._run_ga()

        elapsed = time.perf_counter() - t_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result.elapsed_time = elapsed
        result.memory_kb = peak / 1024
        return result

    def _run_ga(self) -> AlgorithmResult:
        cfg = self.config
        inst = self.instance
        rng = random.Random(cfg.seed)

        # Khởi tạo quần thể
        population = self._init_population(rng)
        self._evaluate_all(population, inst)

        best = min(population).copy()
        history = [best.fitness]

        for _ in range(cfg.generations):
            # Giữ lại tinh hoa
            elite = sorted(population)[:cfg.elite_size]

            new_pop: List[Chromosome] = [e.copy() for e in elite]

            while len(new_pop) < cfg.pop_size:
                p1 = self._tournament_select(population, cfg.tournament_k, rng)
                p2 = self._tournament_select(population, cfg.tournament_k, rng)

                if rng.random() < cfg.p_crossover:
                    c1_genes, c2_genes = self._ox_crossover(p1.genes, p2.genes, rng)
                else:
                    c1_genes, c2_genes = list(p1.genes), list(p2.genes)

                c1 = Chromosome(self._mutate(c1_genes, cfg.p_mutation, rng))
                c2 = Chromosome(self._mutate(c2_genes, cfg.p_mutation, rng))

                new_pop.extend([c1, c2])

            population = new_pop[:cfg.pop_size]
            self._evaluate_all(population, inst)

            gen_best = min(population)
            if gen_best.fitness < best.fitness:
                best = gen_best.copy()

            history.append(best.fitness)

        return AlgorithmResult(
            algorithm_name="GA",
            best_makespan=best.fitness,
            best_sequence=best.genes,
            elapsed_time=0.0,
            history=history,
            is_optimal=False,
            n_iterations=cfg.generations,
        )

    # ------------------------------------------------------------------
    # Khởi tạo quần thể
    # ------------------------------------------------------------------

    def _init_population(self, rng: random.Random) -> List[Chromosome]:
        """Khởi tạo quần thể với greedy seeding."""
        cfg = self.config
        inst = self.instance
        n_greedy = max(1, int(cfg.pop_size * cfg.greedy_ratio))

        population: List[Chromosome] = []

        # Greedy seeding
        greedy_genes = Encoder.greedy_permutation(inst)
        for _ in range(n_greedy):
            genes = list(greedy_genes)
            # Thêm nhiễu bằng swap ngẫu nhiên
            for _ in range(max(1, len(genes) // 10)):
                i, j = rng.sample(range(len(genes)), 2)
                genes[i], genes[j] = genes[j], genes[i]
            population.append(Chromosome(genes))

        # Khởi tạo ngẫu nhiên
        all_jobs = list(range(inst.num_jobs)) * inst.num_machines
        for _ in range(cfg.pop_size - n_greedy):
            genes = list(all_jobs)
            rng.shuffle(genes)
            population.append(Chromosome(genes))

        return population

    # ------------------------------------------------------------------
    # Đánh giá fitness
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_all(population: List[Chromosome], instance: JSPInstance) -> None:
        """Tính makespan cho tất cả cá thể trong quần thể."""
        for chromo in population:
            instance.reset()
            chromo.fitness = float(decode_sequence_to_makespan(chromo.genes, instance))

    # ------------------------------------------------------------------
    # Chọn lọc — Tournament Selection
    # ------------------------------------------------------------------

    @staticmethod
    def _tournament_select(
        population: List[Chromosome],
        k: int,
        rng: random.Random,
    ) -> Chromosome:
        """
        Chọn lọc giải đấu: chọn ngẫu nhiên k cá thể, trả về cá thể tốt nhất.
        Không thay thế (sampling without replacement).
        """
        contestants = rng.sample(population, min(k, len(population)))
        return min(contestants)

    # ------------------------------------------------------------------
    # Lai ghép — Order Crossover (OX)
    # ------------------------------------------------------------------

    @staticmethod
    def _ox_crossover(
        parent1: List[int],
        parent2: List[int],
        rng: random.Random,
    ) -> Tuple[List[int], List[int]]:
        """
        Order Crossover (OX) — bảo toàn thứ tự tương đối của gen.

        Thuật toán OX:
        1. Chọn ngẫu nhiên đoạn [start, end] từ parent1
        2. Sao chép đoạn đó vào con cái ở đúng vị trí
        3. Điền phần còn lại theo thứ tự xuất hiện trong parent2

        Lưu ý: Áp dụng trực tiếp trên multiset (mỗi job có thể xuất hiện nhiều lần).
        """
        n = len(parent1)
        start, end = sorted(rng.sample(range(n), 2))

        def _ox_one(p1: List[int], p2: List[int]) -> List[int]:
            child = [None] * n
            child[start:end] = p1[start:end]

            segment = p1[start:end]
            remaining = []
            for gene in p2[end:] + p2[:end]:
                remaining.append(gene)

            # Điền vào các vị trí trống theo thứ tự
            fill_positions = list(range(end, n)) + list(range(0, start))
            for pos, gene in zip(fill_positions, remaining):
                child[pos] = gene

            # Fallback: điền None còn lại
            fill_idx = 0
            for i in range(n):
                if child[i] is None:
                    while fill_idx < len(remaining) and remaining[fill_idx] in segment:
                        fill_idx += 1
                    child[i] = remaining[fill_idx] if fill_idx < len(remaining) else p1[i]
                    fill_idx += 1

            return child

        child1 = _ox_one(parent1, parent2)
        child2 = _ox_one(parent2, parent1)
        return child1, child2

    # ------------------------------------------------------------------
    # Đột biến — Swap Mutation + Inversion Mutation
    # ------------------------------------------------------------------

    @staticmethod
    def _mutate(genes: List[int], p_mutation: float, rng: random.Random) -> List[int]:
        """
        Đột biến theo xác suất p_mutation.
        Chọn ngẫu nhiên giữa Swap và Inversion mutation.
        """
        if rng.random() >= p_mutation:
            return genes

        n = len(genes)
        result = list(genes)

        if rng.random() < 0.5:
            # Swap Mutation: hoán đổi 2 vị trí ngẫu nhiên
            i, j = rng.sample(range(n), 2)
            result[i], result[j] = result[j], result[i]
        else:
            # Inversion Mutation: đảo ngược đoạn ngẫu nhiên
            i, j = sorted(rng.sample(range(n), 2))
            result[i:j+1] = reversed(result[i:j+1])

        return result
