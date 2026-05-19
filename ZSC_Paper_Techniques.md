# Zero-Shot Object Counting (ZSC)
### Jingyi Xu et al. | Stony Brook University & EPFL | arXiv 2303.02001

---

## Paper làm gì?

Đếm số lượng object thuộc 1 class bất kỳ trong ảnh, **chỉ cần tên class** (không cần exemplar boxes do người annotate). Ví dụ: cho ảnh chợ trái cây + text "strawberry" → đếm ra 38 quả dâu.

---

## Bài toán

| Setting | Input | Cần người annotate? |
|---------|-------|---------------------|
| Class-specific counting | Ảnh + model riêng (e.g., crowd counter) | Cần train data chuyên biệt |
| Few-shot counting | Ảnh + vài bounding boxes exemplar | **Cần** người vẽ box lúc test |
| Exemplar-free (RepRPN) | Ảnh | Không, nhưng **không chọn được class** |
| **Zero-shot counting (đề xuất)** | **Ảnh + tên class** | **Không cần** |

---

## Pipeline tổng quan

```
Input: Ảnh + tên class (e.g., "grape")
                │
    ┌───────────┴──────────────┐
    ▼                          ▼
Step 1: Tạo Class Prototype    Step 2: Sample patches từ ảnh
(Conditional VAE + CLIP)       (random M patches, nhiều sizes)
    │                          │
    └──────────┬───────────────┘
               ▼
Step 3: Chọn Class-relevant Patches
(K-nearest neighbors của prototype)
               │
               ▼
Step 4: Chọn Best Exemplars
(Error Predictor → chọn top-s lowest error)
               │
               ▼
Step 5: Đếm bằng Base Counting Model
(Exemplar-based counter → density map → count)
```

---

## Kỹ thuật đáng chú ý

### 1. Class Prototype Generation bằng Conditional VAE

**Vấn đề:** Cho tên class "grape" → làm sao biết feature vector của "grape" trông như thế nào trong visual feature space?

**Giải pháp:** Train 1 Conditional VAE:
- **Input:** CLIP text embedding của class name (semantic vector `a`)
- **Output:** Synthesize visual features giả lập cho class đó
- **Prototype:** Lấy mean của nhiều features sinh ra → `p_y = mean(G(z, y))` với z ~ N(0,I)

```
"grape" → CLIP text encoder → semantic vector a
    → Conditional VAE decoder G(z, a) → nhiều features giả lập
    → Mean → Class prototype p_y
```

**Tại sao hay:** Không cần ảnh thật của class. Chỉ cần tên class → sinh ra feature đại diện. VAE train trên MS-COCO, generalize tới classes chưa thấy.

**Loss VAE:**
```
L_V = KL(q(z|x,a) || p(z|a)) - E[log p(x|z,a)]
```
- KL divergence: regularize latent space
- Reconstruction: sinh lại feature chính xác

---

### 2. Class-relevant Patch Selection bằng K-NN

**Vấn đề:** Trong ảnh có nhiều objects, chọn patches nào chứa đúng class cần đếm?

**Giải pháp:**
1. Random sample M=450 patches nhiều kích thước từ ảnh
2. Extract ImageNet features cho mỗi patch → `f_i`
3. Tính L2 distance: `d_i = ||f_i - p_y||_2`
4. Chọn k=10 patches gần prototype nhất (K-NN)

**Tại sao hay:** Đơn giản nhưng hiệu quả. ImageNet feature space đã discriminative sẵn → patches gần prototype = patches chứa đúng class.

---

### 3. Error Predictor — Chọn "good exemplar" vs "bad exemplar"

**Observation quan trọng (Figure 2):**
- **Good exemplar** → feature map có pattern lặp lại đều (dots tại vị trí objects) → density map tốt
- **Bad exemplar** → feature map random, không pattern → density map sai

**Giải pháp:** Train 1 network R dự đoán counting error:
```
Input:  F(I) (image feature map) + S (similarity map từ exemplar)
Output: ε = |predicted_count - GT_count|   (counting error)
```

Khi inference: chạy Error Predictor cho mỗi candidate patch → chọn top-3 patches có **error thấp nhất** → dùng làm exemplar.

**Loss:**
```
L = MSE(R(F(I), S), ε)    # Regress counting error
```

**Tại sao hay:** Thay vì heuristic (e.g., objectness score), train model học trực tiếp "patch nào đếm tốt nhất". Ablation cho thấy Error Predictor tốt hơn objectness score.

---

### 4. Base Counting Model (Similarity-based)

```
Ảnh I → Feature extractor F → F(I)         [d × h × w]
Exemplar B → Feature extractor F → F(B)     [d × h_B × w_B]
    → Global Average Pooling → b            [d]

Similarity map: S_ij = w_ij^T · b           (dot product mỗi vị trí)

[F(I); S] → Counter C → Density map D
Count = Σ D(i,j)
```

**Loss:** `L_count = ||D - D*||_2²` (L2 giữa predicted và GT density map)

---

## Kết quả

| Method | Cần exemplar? | Test MAE ↓ | Test RMSE ↓ |
|--------|---------------|-----------|-------------|
| BMNet+ (GT boxes) | Có (GT) | 14.62 | 91.83 |
| RepRPN (exemplar-free) | Không | 27.45 | 129.69 |
| BMNet+ (RPN boxes) | Không (auto) | 34.52 | 132.64 |
| **ZSC (paper này)** | **Không (chỉ tên class)** | **22.09** | **115.17** |

- Vượt RepRPN (method exemplar-free trước đó) **5.36 MAE**
- Gap với GT exemplars (human-annotated) chỉ khoảng 7 MAE
- Patches chọn bởi ZSC còn dùng được cho FamNet, BMNet → method tổng quát

---

## Ablation

| Prototype | Error Predictor | Test MAE |
|-----------|-----------------|----------|
| ✗ | ✗ | 31.37 (random patches) |
| ✓ | ✗ | 25.30 (-6.07) |
| ✗ | ✓ | 23.80 (-7.57) |
| ✓ | ✓ | **22.09** (-9.28) |

Cả 2 components đều contribute, kết hợp tốt hơn.

---

## Kỹ thuật có thể tham khảo cho AA-CLIP

| Kỹ thuật ZSC | Ứng dụng tiềm năng |
|---|---|
| **Conditional VAE sinh class prototype** | Có thể sinh pseudo anomaly features thay vì dùng hand-crafted prompts |
| **Error Predictor đánh giá "chất lượng"** | Dùng tương tự cho confidence weighting — đánh giá chất lượng pseudo mask |
| **Similarity map từ exemplar-image correlation** | Giống patch-text similarity trong AA-CLIP, có thể refine cách tạo pseudo mask |
| **K-NN selection trong feature space** | Chọn training samples chất lượng cao, hoặc chọn pseudo mask đáng tin cậy |

---

## Hạn chế

- Chỉ test trên FSC-147 (1 dataset)
- Cần pre-trained base counter + VAE + error predictor → pipeline phức tạp
- Random patch sampling → không guarantee cover hết objects
- Performance vẫn gap so với human-annotated exemplars (~7 MAE)
