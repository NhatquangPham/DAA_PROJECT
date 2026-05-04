# Thuật toán Di truyền (Genetic Algorithm — GA)

## 1. Tổng quan và lịch sử

Thuật toán Di truyền (GA) được **John Holland** đề xuất vào những năm **1970** tại Đại học Michigan, lấy cảm hứng từ lý thuyết tiến hóa Darwin:

| Khái niệm Sinh học | Tương quan trong GA |
|-------------------|---------------------|
| Cá thể (Individual) | Một nghiệm ứng viên |
| Nhiễm sắc thể (Chromosome) | Chuỗi dữ liệu mã hóa nghiệm |
| Gen (Gene) | Một biến quyết định |
| Quần thể (Population) | Tập hợp N nghiệm tại một thế hệ |
| Độ thích nghi (Fitness) | Giá trị định lượng chất lượng nghiệm |
| Chọn lọc tự nhiên | Chọn cá thể tốt làm cha mẹ |
| Lai ghép (Crossover) | Kết hợp hai nhiễm sắc thể |
| Đột biến (Mutation) | Thay đổi ngẫu nhiên một số gen |

**Điểm khác biệt cốt lõi**: GA là phương pháp tìm kiếm **song song** — khám phá đồng thời nhiều vùng không gian nghiệm, thay vì đi theo một con đường tuần tự duy nhất.

---

## 2. Mô hình toán học

### 2.1 Định nghĩa hình thức

Cho bài toán tối ưu hóa $\min_{x \in X} f(x)$, GA tìm kiếm trong $X$ qua quần thể $P = \{x_1, x_2, \ldots, x_N\}$.

**Hàm fitness** (độ thích nghi):

$$F(x) = \frac{1}{1 + f(x) + \lambda \cdot \sum_k w_k v_k(x)}$$

Trong đó $v_k(x)$ là mức độ vi phạm ràng buộc thứ $k$, $\lambda > 0$ là hệ số phạt.

### 2.2 Định lý Lược đồ (Schema Theorem — Holland, 1975)

Số lượng cá thể tương ứng với lược đồ $H$ tại thế hệ $t+1$:

$$m(H, t+1) \geq m(H, t) \cdot \frac{f(H)}{\bar{f}} \cdot \left[1 - p_c \cdot \frac{\delta(H)}{L-1} - o(H) \cdot p_m\right]$$

**Ý nghĩa**: Các lược đồ ngắn, có bậc thấp, và độ thích nghi cao sẽ phát triển **theo cấp số nhân** qua các thế hệ. GA tích lũy các "khối xây dựng" (building blocks) tốt.

### 2.3 Độ phức tạp

$$T_{GA} = O(G \cdot N \cdot (O(f) + L))$$

Với $G$ = số thế hệ, $N$ = kích thước quần thể, $L$ = độ dài nhiễm sắc thể, $O(f)$ = chi phí đánh giá fitness.

---

## 3. Các thành phần chính

### 3.1 Mã hóa (Encoding) cho JSP

Sử dụng **mã hóa hoán vị** (permutation encoding):

- Nhiễm sắc thể = chuỗi công việc với độ dài $n \times m$ (mỗi job xuất hiện đúng $m$ lần)
- Ví dụ với 3 jobs, 2 machines: `[0, 2, 1, 0, 1, 2]`
- Giải mã: duyệt chuỗi theo thứ tự, mỗi job_id chỉ đến thao tác tiếp theo của nó

### 3.2 Chọn lọc — Tournament Selection

Chọn ngẫu nhiên $k$ cá thể, trả về cá thể có fitness tốt nhất:

```
TOURNAMENT_SELECT(P, k):
    S ← chọn ngẫu nhiên k cá thể từ P (không lặp)
    RETURN arg min_{x ∈ S} f(x)
```

**Ưu điểm**: Không nhạy cảm với scaling của fitness, tham số $k$ điều chỉnh áp lực chọn lọc.

### 3.3 Lai ghép — Order Crossover (OX)

OX bảo toàn thứ tự tương đối của gen:

```
OX(P1, P2):
    1. Chọn ngẫu nhiên đoạn [start, end] từ P1
    2. Sao chép đoạn P1[start:end] vào vị trí tương ứng trong con cái
    3. Điền phần còn lại theo thứ tự xuất hiện trong P2 (bắt đầu từ end)
    RETURN child
```

**Ví dụ**:
```
P1 = [3, 1, 2, 0, 4]   (đoạn [1, 2] được giữ)
P2 = [1, 4, 0, 3, 2]
                   ↓
Child = [_, 1, 2, _, _] → điền từ P2: [4, 1, 2, 0, 3]
```

### 3.4 Đột biến — Swap & Inversion

**Swap Mutation** (xác suất 50%): Hoán đổi 2 vị trí ngẫu nhiên
```
[3, 1, 2, 0, 4] → [3, 4, 2, 0, 1]  (hoán đổi vị trí 1 và 4)
```

**Inversion Mutation** (xác suất 50%): Đảo ngược đoạn ngẫu nhiên
```
[3, 1, 2, 0, 4] → [3, 0, 2, 1, 4]  (đảo ngược đoạn [1, 3])
```

### 3.5 Elitism (Tinh hoa)

Giữ lại $k$ cá thể tốt nhất từ thế hệ trước → đảm bảo **fitness không bao giờ giảm**.

---

## 4. Pseudocode GA cho JSP

```
Đầu vào: instance JSP, pop_size, generations, pc, pm, elite_size
Đầu ra: chuỗi công việc tốt nhất, makespan

1. Khởi tạo quần thể P = {c_1, ..., c_N}
   - 20% từ greedy seeding (SPT-based)
   - 80% ngẫu nhiên

2. Đánh giá fitness: f(c_i) = makespan(decode(c_i)) cho mọi c_i

3. Lặp (g = 1, ..., generations):
   a. Elite = top elite_size cá thể từ P
   b. new_P ← Elite
   c. WHILE |new_P| < pop_size:
       p1 ← TOURNAMENT_SELECT(P, k)
       p2 ← TOURNAMENT_SELECT(P, k)
       IF random() < pc:
           c1, c2 ← OX_CROSSOVER(p1, p2)
       ELSE:
           c1, c2 ← copy(p1), copy(p2)
       c1 ← MUTATE(c1, pm)
       c2 ← MUTATE(c2, pm)
       new_P ← new_P ∪ {c1, c2}
   d. P ← new_P[:pop_size]
   e. Đánh giá fitness cho P
   f. Cập nhật best nếu tìm được nghiệm tốt hơn

4. Trả về: best chromosome, best makespan
```

---

## 5. Phân tích tham số

| Tham số | Giá trị khuyến nghị | Ảnh hưởng |
|---------|--------------------|----|
| `pop_size` | 50–200 | Lớn hơn → đa dạng hơn, chậm hơn |
| `generations` | 100–500 | Nhiều hơn → tốt hơn, tốn thời gian |
| `p_crossover` | 0.7–0.95 | Cao → khai thác nhiều hơn |
| `p_mutation` | 0.01–0.1 | Thấp → ổn định, cao → đa dạng |
| `elite_size` | 1–5 | Tăng → hội tụ nhanh, giảm đa dạng |
| `tournament_k` | 2–5 | Lớn hơn → áp lực chọn lọc cao hơn |

---

## 6. Ưu và nhược điểm

### Ưu điểm
- ✅ Tìm kiếm song song — tránh cực trị cục bộ hiệu quả
- ✅ Linh hoạt — dễ tùy chỉnh toán tử cho bài toán cụ thể
- ✅ Không cần gradient (black-box)
- ✅ Có cơ sở lý thuyết (Schema Theorem)
- ✅ Hoạt động tốt trên không gian không liên tục, phi lồi

### Nhược điểm
- ❌ Nhiều siêu tham số cần điều chỉnh
- ❌ Hội tụ chậm hơn PSO trên nhiều bài toán
- ❌ Chi phí tính toán cao (đánh giá toàn quần thể mỗi thế hệ)
- ❌ Không đảm bảo tối ưu toàn cục

---

## 7. Tài liệu tham khảo

1. Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
2. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
3. Davis, L. (1985). Applying adaptive algorithms to epistatic domains. *IJCAI*, 85, 162–164.
4. Pezzella, F., et al. (2008). A GA for Flexible JSP. *Computers & Operations Research*, 35(10), 3202–3212.
