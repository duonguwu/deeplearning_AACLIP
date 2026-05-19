# Đề cương nghiên cứu
## Mask-Free AA-CLIP: Zero-Shot Anomaly Detection với LoRA Adapter và Pseudo Supervision

---

## 1. Giới thiệu đề tài

### 1.1. Bối cảnh

Phát hiện bất thường (Anomaly Detection) là bài toán quan trọng trong nhiều lĩnh vực: kiểm tra chất lượng sản phẩm công nghiệp (phát hiện vết xước, lỗi hàn, linh kiện hỏng), y tế (phát hiện khối u, tổn thương mô), và giám sát hạ tầng. Điểm khó của AD là dữ liệu bất thường hiếm, đa dạng, và khó dự đoán trước  không thể thu thập đủ mẫu lỗi cho mọi trường hợp.

Gần đây, các phương pháp dựa trên CLIP (Contrastive Language-Image Pretraining) cho zero-shot anomaly detection đã cho kết quả ấn tượng. Trong đó, **AA-CLIP** (CVPR 2025) đạt state-of-the-art bằng cách giải quyết vấn đề "Anomaly Unawareness"  hiện tượng CLIP không phân biệt được ngữ nghĩa "bình thường" và "bất thường" trong text space. AA-CLIP sử dụng two-stage adaptation với residual adapters để tạo text anchors tách biệt (normal vs anomaly) rồi align visual features theo các anchors này.

### 1.2. Vấn đề cần giải quyết

Mặc dù AA-CLIP đạt kết quả tốt, phương pháp này có **hai hạn chế chính**:

**Hạn chế 1  Phụ thuộc ground-truth segmentation masks:**
AA-CLIP sử dụng pixel-level masks (Dice loss + Focal loss) trong huấn luyện. Đây là loại annotation có chi phí rất cao:
- Trong công nghiệp: cần kỹ thuật viên đánh dấu từng pixel vùng lỗi trên ảnh sản phẩm
- Trong y tế: cần bác sĩ chuyên khoa vẽ boundary tổn thương trên ảnh CT/MRI
- Chi phí annotate 1 ảnh pixel-level có thể gấp 10-50 lần so với chỉ gán nhãn image-level (normal/anomaly)

**Hạn chế 2  Overfitting khi dùng nhiều dữ liệu:**
Kết quả AA-CLIP cho thấy image-level AUROC ở full-shot (82.5) **thấp hơn** 64-shot (83.1). Nguyên nhân có thể do adapter full-rank (W ∈ R^(d×d)) có quá nhiều parameters, dễ memorize training patterns thay vì học general anomaly awareness.

---

## 2. Động lực

### 2.1. Tại sao cần giảm phụ thuộc pixel masks?

Trong thực tế, khi triển khai AD cho 1 nhà máy hoặc 1 bệnh viện mới:
- **Image-level labels** (ảnh này có lỗi hay không) → dễ có, rẻ, nhanh
- **Pixel-level masks** (chính xác vùng lỗi nào) → đắt, chậm, cần chuyên gia

Nếu AA-CLIP hoạt động tốt **mà không cần pixel masks**, nó sẽ thực tế hơn rất nhiều cho ứng dụng thực tế.

### 2.2. Tại sao LoRA phù hợp cho bài toán mask-free?

Khi thay GT masks bằng pseudo masks (sinh tự động từ model), pseudo labels sẽ chứa noise. Trong machine learning, khi training data noisy:
- Model **ít parameters** → học patterns chính, bỏ qua noise (underfitting nhẹ nhưng robust)
- Model **nhiều parameters** → memorize cả noise → overfit

AA-CLIP dùng full-rank adapter: W ∈ R^(1024×1024) = **~1M params/layer**. LoRA (rank=8): chỉ **~16K params/layer** (giảm 64 lần). LoRA có ít capacity hơn → ít khả năng memorize noise từ pseudo masks → phù hợp hơn cho weakly-supervised setting.

### 2.3. Tại sao dùng K-NN distance thay threshold cứng?

Một cách tiếp cận đơn giản để tạo pseudo mask là chọn top-k% patches có anomaly score cao nhất. Tuy nhiên, k% bao nhiêu là hợp lý? Threshold cố định không phù hợp cho mọi ảnh vì:
- Ảnh có anomaly lớn (30% diện tích) → cần threshold thấp
- Ảnh có anomaly nhỏ (2% diện tích) → cần threshold cao
- Ảnh normal (0% anomaly) → không nên tạo pseudo mask

Thay vào đó, dùng **K-NN distance tới text anchors T_N và T_A** (đã có sẵn từ AA-CLIP): tính khoảng cách mỗi patch tới T_N và T_A, chỉ gán label cho patches có khoảng cách rõ ràng (gần T_A hoặc gần T_N), bỏ qua vùng mơ hồ. Cách này **adaptive theo từng ảnh**, không cần tune threshold.

---

## 3. Mục tiêu đề tài

### 3.1. Mục tiêu nghiên cứu

Xây dựng **Mask-Free AA-CLIP**  một biến thể của AA-CLIP có thể huấn luyện **chỉ với image-level labels** (normal/anomaly) mà không cần pixel-level segmentation masks, đồng thời duy trì hiệu quả zero-shot anomaly detection ở mức cạnh tranh với AA-CLIP gốc.

### 3.2. Nội dung nghiên cứu

**Nội dung 1:** Tái hiện AA-CLIP gốc làm baseline, đánh giá trên các benchmark tiêu chuẩn.

**Nội dung 2:** Thiết kế cơ chế sinh pseudo masks từ patch-text similarity, sử dụng K-NN distance-based selection để chọn vùng tin cậy cao làm pseudo labels.

**Nội dung 3:** Thay thế full-rank residual adapter bằng LoRA adapter, đánh giá tác động lên robustness khi train với pseudo labels noisy.

**Nội dung 4:** Bổ sung consistency loss để ổn định quá trình huấn luyện khi dùng pseudo supervision.

**Nội dung 5:** Thực nghiệm và ablation study so sánh từng component, đánh giá trên thiết lập zero-shot.

---

## 4. Phương pháp nghiên cứu

### 4.1. Giữ nguyên từ AA-CLIP gốc

Các thành phần cốt lõi được **giữ nguyên** vì đã được chứng minh hiệu quả:

- **Two-stage training:** Stage 1 adapt text encoder, Stage 2 adapt visual encoder.
- **Anomaly-aware text anchors:** T_N (normal), T_A (anomaly) với Disentangle Loss.
- **Multi-level patch aggregation:** Features từ layer {6, 12, 18, 24}.
- **Residual fusion:** x_enhanced = λ·x_adapted + (1-λ)·x_original (giữ CLIP knowledge).

### 4.2. Thay đổi 1  LoRA thay Full-rank Adapter

**AA-CLIP gốc:**
```
x_residual = Norm(Act(W · x))        # W ∈ R^(d×d), ~1M params/layer
x_enhanced = λ · x_residual + (1-λ) · x
```

**Đề xuất:**
```
x_residual = Norm(Act(B · A · x))     # A ∈ R^(r×d), B ∈ R^(d×r), r << d
x_enhanced = λ · x_residual + (1-λ) · x
```

Với d=1024, r=8: giảm từ ~1M → ~16K params/layer. Vẫn giữ residual fusion (λ=0.1) để bảo vệ CLIP knowledge. LoRA đã được chứng minh hiệu quả trong fine-tuning LLMs và vision transformers, đặc biệt trong low-data và noisy-data settings.

### 4.3. Thay đổi 2 — Sinh Pseudo Mask thay Ground-Truth Mask

Thay vì dùng GT segmentation mask trong loss, đề tài sinh **pseudo mask tự động** để làm supervision cho segmentation. Dưới đây là các phương pháp sinh pseudo mask được khảo sát, phân tích, và lựa chọn.

#### 4.3.1. Khảo sát các phương pháp sinh Pseudo Mask

**Phương pháp A — Patch-Text Similarity Map**

- **Định nghĩa:** Tính cosine similarity giữa visual patch features và text anchors (T_N, T_A) để xác định mức độ "bất thường" của từng vùng trong ảnh. Vùng có similarity cao với T_A (anomaly anchor) được coi là vùng bất thường.
- **Input:** (1) Visual patch features V_patch từ visual encoder, (2) Text anchors T_N, T_A từ text encoder đã adapt.
- **Output:** Anomaly heatmap kích thước N_patches (e.g., 37×37 cho input 518px), upscale lên H×W → pseudo mask nhị phân sau khi threshold.
- **Cách thực hiện:**
  ```
  Với mỗi patch i:
    sim_A(i) = CosSim(V_patch_i, T_A)     # Độ giống anomaly
    sim_N(i) = CosSim(V_patch_i, T_N)     # Độ giống normal
    margin(i) = sim_A(i) - sim_N(i)       # Dương = nghi anomaly
  → Reshape margin thành grid 37×37
  → Bilinear upscale lên H×W
  → Threshold → pseudo mask {0, 1}
  ```
- **Ưu điểm:** Đơn giản, dùng components có sẵn trong AA-CLIP (T_N, T_A, V_patch), không cần thêm model hay training.
- **Nhược điểm:** Độ phân giải thô (mỗi patch = 14×14 pixel), boundary giữa normal/anomaly không sắc nét.

**Phương pháp B — GradCAM / Attention Map**

- **Định nghĩa:** Sử dụng gradient hoặc attention weights từ visual encoder (ViT) để xác định vùng ảnh mà model "chú ý" nhất khi đưa ra dự đoán anomaly. GradCAM tính gradient của anomaly score theo feature maps, Attention Map lấy trực tiếp từ self-attention layers.
- **Input:** (1) Ảnh đầu vào, (2) Visual encoder (ViT) đã forward, (3) Anomaly score (image-level).
- **Output:** Heatmap H×W chỉ ra vùng model focus → threshold → pseudo mask.
- **Cách thực hiện:**
  ```
  GradCAM:
    1. Forward ảnh → anomaly score s
    2. Backward: gradient = ∂s/∂F_last_layer
    3. GradCAM = ReLU(Σ_c (gradient_c × F_c))     # weighted sum theo channel
    4. Upscale → heatmap → threshold → pseudo mask

  Attention Map:
    1. Forward ảnh qua ViT
    2. Lấy attention weights từ self-attention layer cuối: A ∈ R^(N×N)
    3. Lấy attention từ [CLS] token tới các patch tokens: A_cls ∈ R^N
    4. Reshape A_cls thành grid → upscale → heatmap → threshold → pseudo mask
  ```
- **Ưu điểm:** Không cần thêm model, giải thích được model đang nhìn vào đâu, có thể kết hợp với Phương pháp A để tăng tin cậy.
- **Nhược điểm:** GradCAM trên ViT cho kết quả kém hơn trên CNN (do architecture khác). Attention map có thể diffuse (lan rộng), không localize chính xác.

**Phương pháp C — Dùng Foundation Model (SAM)**

- **Định nghĩa:** Sử dụng Segment Anything Model (SAM) — foundation model cho segmentation — để refine boundary của vùng anomaly. Similarity map thô (từ Phương pháp A) được dùng làm prompt points cho SAM, SAM trả về mask có boundary chính xác.
- **Input:** (1) Ảnh gốc, (2) Rough anomaly region từ similarity map (dùng làm prompt cho SAM).
- **Output:** Pseudo mask có boundary sắc nét, pixel-level accurate.
- **Cách thực hiện:**
  ```
  1. Similarity map (Phương pháp A) → xác định vùng nghi anomaly
  2. Chọn top-k points có anomaly score cao nhất → dùng làm point prompts
  3. Feed ảnh + point prompts vào SAM
  4. SAM output → mask với boundary chính xác
  ```
- **Ưu điểm:** Chất lượng mask cao nhất, boundary sắc nét, SAM là pretrained model miễn phí.
- **Nhược điểm:** Thêm dependency ngoài (SAM model, ~600MB+), inference chậm hơn, khó phân biệt contribution của method mình vs SAM. Chỉ cần SAM lúc train, nhưng thêm complexity cho pipeline.

**Phương pháp D — Iterative Self-Refinement**

- **Định nghĩa:** Model tự sinh pseudo mask, train trên pseudo mask đó, rồi dùng model đã train để sinh pseudo mask tốt hơn, lặp lại nhiều rounds cho đến khi converge. Tương tự self-training trong semi-supervised learning.
- **Input:** (1) Ảnh + image-level labels, (2) Model từ round trước (hoặc pretrained CLIP cho round 0).
- **Output:** Pseudo mask cải thiện dần qua mỗi round.
- **Cách thực hiện:**
  ```
  Round 0: Train model chỉ với image-level labels (BCE) → model thô
  Round 1: Model thô sinh similarity map → pseudo mask v1
           → Train lại model với pseudo mask v1 → model v1
  Round 2: Model v1 sinh pseudo mask v2 (tốt hơn v1)
           → Train lại → model v2
  ...
  Dừng khi: |performance(round_n) - performance(round_{n-1})| < ε
  ```
- **Ưu điểm:** Pseudo mask tự cải thiện, không cần model ngoài, tiềm năng đạt chất lượng cao.
- **Nhược điểm:** Chậm (train nhiều lần), risk confirmation bias (model ngày càng confident vào lỗi sai của chính mình), không guarantee converge, implementation phức tạp.

#### 4.3.2. So sánh và lựa chọn

| Tiêu chí | A. Similarity Map | B. GradCAM/Attention | C. SAM | D. Iterative |
|----------|-------------------|---------------------|--------|-------------|
| Chất lượng mask | Trung bình | Trung bình | Cao | Cao dần |
| Độ phức tạp | Thấp | Thấp | Cao | Rất cao |
| Cần thêm model | Không | Không | Có (SAM) | Không |
| Thời gian train | Bình thường | Bình thường | Bình thường | ×3-5 lần |
| Dễ reproduce | Cao | Cao | Trung bình | Thấp |
| Phù hợp project |  **Chọn chính** | Kết hợp bổ trợ | Không chọn | Không chọn |

**Lựa chọn: Phương pháp A (Similarity Map)** làm phương pháp chính, với lý do:
- Dùng components có sẵn (T_N, T_A, V_patch) → không thêm dependency
- Đơn giản, dễ implement và reproduce
- Kết hợp với adaptive threshold + confidence weighting đã giảm noise đáng kể
- Phương pháp B (Attention Map) có thể được thử nghiệm bổ trợ trong ablation

#### 4.3.3. Chi tiết phương pháp được chọn — Similarity Map + Adaptive Threshold

**Bước 1 — Tính similarity margin:** Sau Stage 1 (text anchors đã disentangled), tính cosine similarity giữa mỗi visual patch và T_N, T_A:
```
sim_N(i) = CosSim(V_patch_i, T_N)    # Similarity với normal
sim_A(i) = CosSim(V_patch_i, T_A)    # Similarity với anomaly
margin(i) = sim_A(i) - sim_N(i)      # Margin: dương = nghi anomaly
```

**Bước 2 — Adaptive threshold:** Phân loại patches theo margin, threshold tự điều chỉnh theo phân phối margin từng ảnh:
```
τ_high = mean(margin) + α · std(margin)    # Ngưỡng trên: rõ ràng anomaly
τ_low  = mean(margin) - β · std(margin)    # Ngưỡng dưới: rõ ràng normal

Nếu margin(i) > τ_high   → pseudo label = anomaly (tin cậy cao)
Nếu margin(i) < τ_low    → pseudo label = normal (tin cậy cao)
Nếu τ_low ≤ margin(i) ≤ τ_high → KHÔNG gán label (ambiguous, bỏ qua trong loss)
```
α, β là hyperparameters (mặc định α=1.0, β=0.5). Threshold thay đổi theo từng ảnh nên phù hợp cả anomaly lớn, nhỏ, và ảnh normal.

**Bước 3 — Confidence weighting:** Dùng |margin| làm trọng số cho từng patch:
```
weight(i) = |margin(i)| / max(|margin|)    # Normalize về [0,1]
```
Patches có margin lớn (xa decision boundary) → weight cao → đóng góp nhiều vào loss.
Patches có margin nhỏ (gần boundary) → weight thấp → ít ảnh hưởng.

**Ưu điểm của phương pháp được chọn:**
- Adaptive: threshold thay đổi theo ảnh, không cần tune 1 global threshold cho tất cả
- Robust: bỏ qua vùng ambiguous → tránh model học noise
- Principled: dựa trên distance tới T_N/T_A có sẵn, không cần thêm network
- Confidence weighting tự nhiên: margin càng rõ → càng tin cậy → weight càng cao

### 4.4. Thay đổi 3  Consistency Loss

Bổ sung consistency loss để ổn định training khi dùng pseudo labels. Có hai hình thức:

**Multi-level consistency:** Anomaly maps từ các levels khác nhau (layer 6, 12, 18, 24) nên nhất quán — nếu vùng (x,y) là anomaly ở level sâu, nó cũng nên là anomaly ở level nông:
```
L_multi_level = MSE(anomaly_map_level_i, anomaly_map_level_j)
```

**Augmentation consistency:** Prediction map của ảnh gốc và ảnh sau augmentation nhẹ (flip, small crop, color jitter) nên tương đồng:
```
L_augment = MSE(anomaly_map(x), anomaly_map(aug(x)))
```

Tổng consistency loss:
```
L_consistency = L_multi_level + L_augment
```

Khi pseudo labels noisy, consistency loss giúp model không phụ thuộc vào từng pseudo label cụ thể mà học patterns ổn định hơn — các levels phải "đồng thuận" và augmentation không được thay đổi kết quả.

### 4.5. Tổng hợp Loss Function

**AA-CLIP gốc:**
```
Stage 1: L = L_cls(BCE) + L_seg(Dice + Focal, dùng GT mask) + γ·L_dis
Stage 2: L = L_cls(BCE) + L_seg(Dice + Focal, dùng GT mask)
```

**Đề xuất:**
```
Stage 1: L = L_cls(BCE) + γ·L_dis                    ← Chỉ image-level + disentangle
Stage 2: L = L_cls(BCE) + L_pseudo_seg + L_consistency
```

Trong đó:
```
L_pseudo_seg = Σ weight(i) · [Dice(pseudo_pred_i, pseudo_label_i)
                              + Focal(pseudo_pred_i, pseudo_label_i)]
```
Chỉ tính trên patches có pseudo label (bỏ ambiguous), weighted theo confidence.

**Lưu ý quan trọng:** Stage 1 **không dùng segmentation loss** vì chưa có visual adapter → pseudo masks chưa tốt. Chỉ dùng image-level BCE + Disentangle Loss để tạo text anchors. Stage 2 mới dùng pseudo masks (sinh từ text anchors đã disentangled ở Stage 1).

### 4.6. Pipeline tóm tắt

```
Stage 1: Text Adaptation (giữ nguyên AA-CLIP, bỏ L_seg)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Text prompts + Images + Image-level labels
Trainable: LoRA adapters ở text encoder + projector
Loss: L_cls(BCE) + γ·L_dis
Output: Disentangled T_N, T_A

Stage 2: Visual Adaptation (thay GT mask bằng pseudo mask)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Images + Fixed T_N, T_A từ Stage 1
Trainable: LoRA adapters ở visual encoder + projectors

Mỗi epoch:
  1. Forward pass → tính margin(i) = CosSim(V_i, T_A) - CosSim(V_i, T_N)
  2. Sinh pseudo mask: adaptive threshold theo mean ± α·std
  3. Tính confidence weight: |margin| normalized
  4. Loss: L_cls + L_pseudo_seg (weighted) + L_consistency
  5. Backward → update LoRA params + projectors

Inference (giống AA-CLIP):
━━━━━━━━━━━━━━━━━━━━━━━━━
Image → Adapted Visual Encoder → V_image, V_patch
Class name → Adapted Text Encoder → T_N, T_A
→ CosSim → Anomaly score + Anomaly map
```

---

## 5. Kết quả dự kiến

### 5.1. Về mô hình

- Xây dựng được Mask-Free AA-CLIP hoạt động **không cần pixel-level masks** khi huấn luyện.
- LoRA adapter giảm **~64 lần** số parameters so với full-rank adapter gốc.

### 5.2. Về hiệu quả (dự kiến)

| Metric | AA-CLIP gốc (GT mask) | Mask-Free AA-CLIP (dự kiến) |
|--------|------------------------|------------------------------|
| Pixel AUROC | 92.8 (64-shot) | ~89-91 (chấp nhận giảm 2-4%) |
| Image AUROC | 83.1 (64-shot) | ~82-83 (gần bằng hoặc bằng) |
| Annotation cần | Image labels + Pixel masks | **Chỉ image labels** |
| Overfitting full-shot | Có (82.5 < 83.1) | Giảm nhờ LoRA ít params |

**Kỳ vọng thực tế:**
- Pixel-level có thể giảm nhẹ (vì pseudo mask không chính xác bằng GT)  chấp nhận giảm 2-4%.
- Image-level có thể giữ nguyên hoặc cải thiện  vì LoRA giảm overfitting.
- Trade-off: **giảm annotation cost đáng kể** đổi lấy **giảm nhẹ pixel accuracy** → xứng đáng trong thực tế.

### 5.3. Về ý nghĩa nghiên cứu

- Chứng minh hiệu quả AA-CLIP đến từ **anomaly-aware text anchors** (core contribution), không phải chỉ từ pixel-level supervision.
- Nếu kết quả khả quan → zero-shot AD không nhất thiết cần pixel masks → mở rộng khả năng ứng dụng.
- LoRA adapter có thể là lựa chọn tốt hơn full-rank cho CLIP adaptation trong AD.

### 5.4. Về ứng dụng

Mô hình giảm phụ thuộc vào pixel masks có giá trị thực tiễn cao trong:
- **Kiểm tra lỗi công nghiệp:** Trong dây chuyền sản xuất, việc gán nhãn vùng lỗi chi tiết trên ảnh sản phẩm (vết xước, lỗi hàn, nứt vỡ) rất tốn thời gian và cần kỹ thuật viên giàu kinh nghiệm. Mask-free cho phép triển khai nhanh chỉ với nhãn "lỗi/không lỗi".
- **Y tế:** Annotation pixel-level trên ảnh CT/MRI cần bác sĩ chuyên khoa, chi phí rất cao và thời gian dài. Giảm dependency vào pixel masks giúp mở rộng ứng dụng AD sang các bệnh viện/phòng khám có ít nguồn lực annotation.

---

## 6. Kế hoạch nghiên cứu

### Giai đoạn 1 (Tuần 1-2): Khảo sát và chuẩn bị

- Đọc kỹ paper AA-CLIP, hiểu pipeline, code, và kết quả.
- Nghiên cứu LoRA implementation trong vision transformers.
- Chuẩn bị datasets: VisA (training), MVTec-AD (evaluation).
- Setup môi trường: OpenCLIP, ViT-L/14, GPU.

### Giai đoạn 2 (Tuần 3-4): Reproduce Baseline

- Chạy AA-CLIP gốc, xác nhận kết quả baseline.
- Ghi nhận: image AUROC, pixel AUROC trên MVTec-AD (zero-shot).
- Phân tích similarity maps và anomaly maps để hiểu behavior.

### Giai đoạn 3 (Tuần 5-8): Phát triển phương pháp

- **Tuần 5:** Implement LoRA adapter thay full-rank, test với GT masks trước (verify LoRA không làm giảm quá nhiều).
- **Tuần 6:** Implement pseudo mask generation (K-NN distance-based, adaptive threshold).
- **Tuần 7:** Implement confidence weighting + consistency loss.
- **Tuần 8:** Kết hợp tất cả, debug, tune hyperparameters (α, β, rank r, λ).

### Giai đoạn 4 (Tuần 9-10): Thực nghiệm và đánh giá

- So sánh các biến thể:
  - (A) AA-CLIP gốc (GT mask, full-rank adapter)
  - (B) AA-CLIP + LoRA (GT mask, LoRA adapter)  isolate LoRA effect
  - (C) AA-CLIP + pseudo mask (full-rank, pseudo mask)  isolate pseudo mask effect
  - (D) Mask-Free AA-CLIP (LoRA + pseudo mask + consistency)  full method
- Ablation study: đánh giá vai trò từng component.
- Đánh giá zero-shot trên MVTec-AD, BTAD, MPDD.
- Viết báo cáo, phân tích kết quả.
