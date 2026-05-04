# Thuật toán Tối ưu hóa Bầy đàn (PSO)

## 1. Nguồn gốc và cảm hứng sinh học

Thuật toán **Particle Swarm Optimization (PSO)** được đề xuất lần đầu bởi **James Kennedy và Russell Eberhart** vào năm **1995**. Ý tưởng xuất phát từ việc quan sát hành vi tập thể trong tự nhiên:

- **Đàn chim tìm thức ăn**: Mỗi con chim bay theo hướng của mình, đồng thời điều chỉnh dựa trên vị trí tốt nhất nó từng đến và vị trí tốt nhất của cả đàn. Không cần lãnh đạo tập trung.
- **Đàn cá di cư**: Mỗi cá thể phản ứng với trạng thái cục bộ và thông tin từ hàng xóm, tạo nên hành vi tập thể thông minh từ quy tắc đơn giản.

PSO là ví dụ điển hình của **trí tuệ bầy đàn (Swarm Intelligence)** — lớp thuật toán tiến hóa mô phỏng trí thông minh tập thể xuất hiện từ sự tương tác của các cá thể đơn giản.

---

## 2. Mô hình toán học

### 2.1 Không gian tìm kiếm

Bài toán tối ưu hóa D-chiều:

$$\min_{x \in S} f(x), \quad S = \{x \in \mathbb{R}^D \mid lb_j \leq x_j \leq ub_j\}$$

### 2.2 Quần thể và Hạt

Quần thể (swarm) gồm N hạt: $P = \{P_1, P_2, \ldots, P_N\}$

Mỗi hạt $P_i$ tại thời điểm $t$ được đặc trưng bởi:
- **Vị trí**: $x_i(t) \in S$ — nghiệm ứng viên
- **Vận tốc**: $v_i(t)$ — hướng và tốc độ di chuyển
- **Vị trí tốt nhất cá nhân (pBest)**: $pBest_i$ — vị trí tốt nhất $P_i$ từng đến

### 2.3 Phương trình cập nhật

$$v_{i,d}^{(t+1)} = w \cdot v_{i,d}^{(t)} + c_1 r_1 \left(pBest_{i,d} - x_{i,d}^{(t)}\right) + c_2 r_2 \left(gBest_d - x_{i,d}^{(t)}\right)$$

$$x_{i,d}^{(t+1)} = x_{i,d}^{(t)} + v_{i,d}^{(t+1)}$$

| Ký hiệu | Tên gọi | Ý nghĩa |
|---------|---------|---------|
| $w$ | Trọng số quán tính | Duy trì hướng di chuyển cũ. $w$ lớn → khám phá; $w$ nhỏ → khai thác |
| $c_1$ | Hệ số nhận thức | Kéo hạt về kinh nghiệm tốt nhất của bản thân (pBest) |
| $c_2$ | Hệ số xã hội | Kéo hạt về kinh nghiệm tốt nhất của đàn (gBest) |
| $r_1, r_2$ | Số ngẫu nhiên | $r_1, r_2 \sim U(0,1)$ — duy trì tính đa dạng |

### 2.4 Trọng số quán tính giảm tuyến tính

$$w(t) = w_{\max} - \frac{w_{\max} - w_{\min}}{T_{\max}} \cdot t$$

Với $w_{\max} = 0.9$, $w_{\min} = 0.4$: Pha đầu khám phá rộng, pha cuối khai thác sâu.

### 2.5 Giới hạn vận tốc

$$v_{i,d} \leftarrow \text{clip}(v_{i,d}, -V_{\max}, V_{\max}), \quad V_{\max} = k \cdot (ub - lb)$$

---

## 3. Kỹ thuật SPV cho bài toán tổ hợp

PSO nguyên bản được thiết kế cho không gian liên tục. Để giải bài toán JSP (không gian rời rạc), ta dùng kỹ thuật **Smallest Position Value (SPV)**:

**Cho vector thực** $x = (x_1, x_2, \ldots, x_n)$, xây dựng hoán vị $\pi$ bằng cách sắp thứ tự tăng dần:

$$x_{\pi(1)} \leq x_{\pi(2)} \leq \cdots \leq x_{\pi(n)}$$

**Ví dụ**:
```
Position : [0.7, 0.1, 0.9, 0.3, 0.5, 0.2]
Argsort  :  [1,   5,   3,   4,   0,   2]   (chỉ số theo giá trị tăng dần)
SPV Seq  :  [1%3, 5%3, 3%3, 4%3, 0%3, 2%3] = [1, 2, 0, 1, 0, 2]
                                              (job_id = index % n_jobs)
```

---

## 4. Pseudocode PSO cho JSP

```
Đầu vào: instance JSP, n_particles, max_iter, w_max, w_min, c1, c2
Đầu ra: nghiệm tốt nhất (chuỗi công việc, makespan)

1. Khởi tạo:
   FOR i = 1 TO n_particles:
       x_i ← vector ngẫu nhiên trong [0, 1]^D (D = n_jobs × n_machines)
       v_i ← vector vận tốc nhỏ ngẫu nhiên
       pBest_i ← x_i
       Đánh giá f(x_i) = makespan(SPV(x_i))
   gBest ← x_i có f(x_i) nhỏ nhất

2. Lặp (t = 1, ..., max_iter):
   w ← w_max - (w_max - w_min) × t / max_iter
   FOR i = 1 TO n_particles:
       r1, r2 ← ngẫu nhiên trong [0, 1]
       v_i ← w·v_i + c1·r1·(pBest_i - x_i) + c2·r2·(gBest - x_i)
       v_i ← clip(v_i, -v_max, v_max)
       x_i ← x_i + v_i
       x_i ← clip(x_i, 0, 1)
       
       fitness ← makespan(SPV(x_i))
       IF fitness < f(pBest_i):
           pBest_i ← x_i
       IF fitness < f(gBest):
           gBest ← x_i

3. Trả về: sequence = SPV(gBest), makespan = f(gBest)
```

---

## 5. Độ phức tạp

| Thành phần | Độ phức tạp |
|-----------|------------|
| Khởi tạo | O(N × D) |
| Đánh giá 1 hạt | O(n × m) |
| 1 vòng lặp | O(N × n × m) |
| **Tổng** | **O(T × N × n × m)** |

Với T = max_iter, N = n_particles, n = n_jobs, m = n_machines.

---

## 6. Ưu và nhược điểm

### Ưu điểm
- ✅ Ít tham số cần điều chỉnh (w, c1, c2)
- ✅ Hội tụ nhanh — chia sẻ thông tin qua gBest
- ✅ Dễ cài đặt và song song hóa
- ✅ Không cần gradient (black-box optimization)
- ✅ Hiệu quả cho không gian tìm kiếm lớn

### Nhược điểm
- ❌ Có thể hội tụ sớm vào cực trị cục bộ
- ❌ Hiệu năng phụ thuộc nhiều vào tham số
- ❌ Không đảm bảo tối ưu toàn cục
- ❌ Tốn bộ nhớ O(N × D) cho toàn đàn

---

## 7. Tài liệu tham khảo

1. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *ICNN'95*, 4, 1942–1948.
2. Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. *IEEE CEC*, 69–73.
3. Liu, H., et al. (2007). A particle swarm approach to quadratic assignment problems. *Advances in Swarm Intelligence*, 170–179.
