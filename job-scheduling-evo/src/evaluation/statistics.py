"""
Kiểm định thống kê để so sánh các thuật toán.

Sử dụng kiểm định Wilcoxon Signed-Rank Test (phi tham số)
để so sánh hai thuật toán không đòi hỏi giả định phân phối chuẩn.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from scipy import stats


@dataclass
class WilcoxonTest:
    """
    Kết quả kiểm định Wilcoxon Signed-Rank Test.

    Giả thuyết:
    - H₀: Không có sự khác biệt có ý nghĩa thống kê giữa hai thuật toán
    - H₁: Có sự khác biệt có ý nghĩa thống kê

    Attributes:
        algorithm_a     : Tên thuật toán A
        algorithm_b     : Tên thuật toán B
        statistic       : Giá trị thống kê kiểm định
        p_value         : P-value
        alpha           : Mức ý nghĩa thống kê (thường 0.05)
        reject_h0       : True nếu bác bỏ H₀ (p_value < alpha)
        conclusion      : Kết luận bằng ngôn ngữ tự nhiên
    """
    algorithm_a: str
    algorithm_b: str
    statistic: float
    p_value: float
    alpha: float = 0.05

    @property
    def reject_h0(self) -> bool:
        return self.p_value < self.alpha

    @property
    def conclusion(self) -> str:
        if self.reject_h0:
            return (
                f"Có sự khác biệt có ý nghĩa thống kê giữa {self.algorithm_a} và "
                f"{self.algorithm_b} (p={self.p_value:.4f} < α={self.alpha}). Bác bỏ H₀."
            )
        return (
            f"Không có sự khác biệt có ý nghĩa thống kê giữa {self.algorithm_a} và "
            f"{self.algorithm_b} (p={self.p_value:.4f} ≥ α={self.alpha}). Không bác bỏ H₀."
        )

    def __repr__(self) -> str:
        return (
            f"WilcoxonTest({self.algorithm_a} vs {self.algorithm_b}): "
            f"statistic={self.statistic:.4f}, p={self.p_value:.4f} → "
            f"{'Reject H₀' if self.reject_h0 else 'Fail to reject H₀'}"
        )


@dataclass
class StatisticalSummary:
    """Tổng hợp kiểm định thống kê giữa nhiều cặp thuật toán."""
    tests: List[WilcoxonTest]

    def print_report(self) -> None:
        print("=" * 70)
        print("KIỂM ĐỊNH THỐNG KÊ WILCOXON SIGNED-RANK TEST")
        print("=" * 70)
        for test in self.tests:
            print(f"\n{test.algorithm_a} vs {test.algorithm_b}:")
            print(f"  Statistic : {test.statistic:.4f}")
            print(f"  P-value   : {test.p_value:.4f}")
            print(f"  Kết luận  : {test.conclusion}")
        print("=" * 70)


def wilcoxon_test(
    samples_a: List[float],
    samples_b: List[float],
    name_a: str,
    name_b: str,
    alpha: float = 0.05,
) -> WilcoxonTest:
    """
    Thực hiện Wilcoxon Signed-Rank Test giữa hai mẫu.

    Args:
        samples_a   : Danh sách makespan của thuật toán A
        samples_b   : Danh sách makespan của thuật toán B
        name_a      : Tên thuật toán A
        name_b      : Tên thuật toán B
        alpha       : Mức ý nghĩa thống kê

    Returns:
        WilcoxonTest với đầy đủ thông tin kiểm định
    """
    if len(samples_a) != len(samples_b):
        raise ValueError("Hai mẫu phải có cùng kích thước.")
    if len(samples_a) < 5:
        raise ValueError("Cần ít nhất 5 lần chạy để kiểm định thống kê.")

    stat, p_value = stats.wilcoxon(samples_a, samples_b, zero_method="wilcox")

    return WilcoxonTest(
        algorithm_a=name_a,
        algorithm_b=name_b,
        statistic=float(stat),
        p_value=float(p_value),
        alpha=alpha,
    )
