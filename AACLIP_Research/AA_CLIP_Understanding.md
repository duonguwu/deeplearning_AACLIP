# Tìm hiểu Paper: AA-CLIP — Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP
### CVPR 2025 | Wenxin Ma et al. | USTC & Stanford

---

## 1. Paper này làm gì?

AA-CLIP là phương pháp cải tiến CLIP để phát hiện bất thường (anomaly detection) trong ảnh mà **không cần huấn luyện trên từng loại đối tượng cụ thể** (zero-shot). Paper đề xuất cách "dạy" CLIP phân biệt giữa "bình thường" và "bất thường" thông qua việc tinh chỉnh nhẹ cả text encoder lẫn visual encoder theo chiến lược hai giai đoạn.

**Kết quả:** Đạt SOTA trên 11 bộ dữ liệu (công nghiệp + y tế), chỉ cần 2 mẫu/lớp đã vượt các phương pháp trước dùng toàn bộ dữ liệu.

---

## 2. Động lực — Tại sao họ làm điều này?

### Bối cảnh
- **Anomaly Detection (AD)** là bài toán quan trọng trong công nghiệp (phát hiện lỗi sản phẩm) và y tế (phát hiện tổn thương).
- Các phương pháp truyền thống cần **nhiều dữ liệu huấn luyện** cho từng loại sản phẩm/bệnh → không thực tế khi gặp loại mới.
- **CLIP** (Contrastive Language-Image Pretraining) có khả năng generalization tốt, hứa hẹn cho zero-shot AD.

### Vấn đề phát hiện: Anomaly Unawareness
Tác giả phát hiện CLIP có **hạn chế cốt lõi** cho bài toán AD:

- CLIP được pre-train trên cặp ảnh-text ở mức **category** ("a photo of a carpet"), không bao giờ thấy mô tả anomaly ("a photo of a **broken** carpet").
- Hệ quả: text embeddings cho "normal carpet" và "broken carpet" **gần như giống nhau** trong feature space.
- Minh chứng: Cho ảnh carpet bị rách rõ ràng, CLIP vẫn cho similarity cao hơn với prompt "normal" (0.19) so với "broken" (0.18).

→ **Nếu text anchor không phân biệt được normal/anomaly, thì mọi so sánh visual đều vô nghĩa.**

### Tại sao các phương pháp trước chưa giải quyết được?
- **WinCLIP, VAND, MVFA-AD:** Chỉ cải thiện visual features, vẫn dùng text embeddings gốc của CLIP → text vẫn anomaly-unaware.
- **AnomalyCLIP, AdaCLIP:** Dùng learnable prompts nhưng có thể làm mất class information của CLIP → giảm generalization.
- Chưa ai chỉ ra và giải quyết trực tiếp vấn đề "Anomaly Unawareness" ở text space.

---

## 3. Mục tiêu nghiên cứu

1. **Làm CLIP nhận biết được anomaly** — tách biệt rõ ràng semantics "normal" và "anomaly" trong text space.
2. **Giữ generalization** — không phá hủy kiến thức đã học của CLIP (class knowledge) khi fine-tune.
3. **Hiệu quả dữ liệu** — đạt kết quả tốt chỉ với rất ít mẫu huấn luyện (few-shot).
4. **Tổng quát** — hoạt động tốt trên cả domain công nghiệp lẫn y tế, kể cả với classes chưa thấy (zero-shot).

---

## 4. Phương pháp — Họ làm như thế nào?

### Ý tưởng chính

Sửa text space trước (tạo "neo" tốt), rồi mới sửa visual space (align ảnh theo neo). Dùng **Residual Adapter** để tinh chỉnh nhẹ, không phá CLIP.

### Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────┐
│  AA-CLIP = CLIP (frozen) + Residual Adapters (nhẹ)  │
│                                                     │
│  Stage 1: Tinh chỉnh Text Encoder                  │
│     → Tạo text anchors T_N (normal), T_A (anomaly)  │
│     → T_N ⊥ T_A (orthogonal, tách biệt rõ)         │
│                                                     │
│  Stage 2: Tinh chỉnh Visual Encoder                 │
│     → Align patch features với T_N, T_A đã cố định  │
│     → Anomaly map chính xác hơn                     │
└─────────────────────────────────────────────────────┘
```

### Stage 1: Tạo Text Anchors tách biệt

- **Input:** Text prompts (normal: "a photo of [CLS]"; anomaly: "a photo of broken [CLS]") + Ảnh
- **Trainable:** Residual Adapters ở 3 shallow layers đầu text encoder + final projector
- **Frozen:** Toàn bộ visual encoder
- **Loss:** Classification (BCE) + Segmentation (Dice + Focal) + **Disentangle Loss** (force T_N ⊥ T_A)
- **Kết quả:** T_N và T_A tách biệt rõ ràng → làm "neo" cho Stage 2

### Stage 2: Align Visual Features theo Text Anchors

- **Input:** Ảnh + Text anchors cố định từ Stage 1
- **Trainable:** Residual Adapters ở 6 shallow layers visual encoder + 4 Projectors
- **Frozen:** Text encoder đã adapt + deep layers visual encoder
- **Loss:** Classification (BCE) + Segmentation (Dice + Focal)
- **Multi-level:** Trích features từ 4 levels (layer 6, 12, 18, 24) → project → cộng lại → so sánh với text anchors

### Residual Adapter — Cơ chế bảo vệ CLIP

```
x_enhanced = 0.1 × adapted_feature + 0.9 × original_feature
```

- Chỉ blend 10% feature mới, giữ 90% CLIP gốc → tránh catastrophic forgetting.
- Ablation cho thấy: adapter thường (không residual) **phá hủy** CLIP → pixel AUROC giảm 40 điểm!

### Tại sao phải two-stage?

- Nếu train cả text + visual cùng lúc (one-stage) → class information bị collapse → mất generalization.
- Two-stage: mỗi giai đoạn có 1 "điểm neo" cố định → adaptation ổn định.

### Inference (Dự đoán)

1. Ảnh → Visual Encoder → features
2. Tên class → Text Encoder → T_N, T_A  
3. So sánh cosine similarity → anomaly score (ảnh) + anomaly map (pixel)

---

## 5. Kết quả đạt được

### Setup
- **Train:** VisA dataset (industrial). **Test zero-shot:** 11 datasets (4 industrial + 7 medical).
- **Backbone:** OpenCLIP ViT-L/14, input 518×518
- **Metric:** AUROC ở image-level và pixel-level

### Bảng kết quả chính

| Method | Venue | Data dùng train | Pixel AUROC | Image AUROC |
|--------|-------|-----------------|-------------|-------------|
| CLIP | - | - | 49.9 | 68.3 |
| WinCLIP | CVPR'23 | - | 74.7 | 67.8 |
| VAND | CVPRw'23 | full | 89.3 | - |
| AnomalyCLIP | ICLR'24 | full | 91.3 | 78.4 |
| AdaCLIP | ECCV'24 | full | 90.4 | 80.6 |
| **AA-CLIP** | **CVPR'25** | **2-shot** | **92.0** | 78.1 |
| **AA-CLIP** | **CVPR'25** | **64-shot** | **92.8** | **83.1** |
| **AA-CLIP** | **CVPR'25** | **full** | **93.4** | 82.5 |

**Điểm nổi bật:**
- Chỉ 2 mẫu/lớp đã vượt tất cả methods dùng full data ở pixel level.
- 64-shot đạt SOTA cả 2 levels.
- Hoạt động tốt trên cả industrial và medical domain.

---

## 6. Hạn chế của paper

### Hạn chế tác giả thừa nhận
1. **Overfitting ở full-shot:** Image AUROC full-shot (82.5) < 64-shot (83.1) → model bão hòa khi thêm data.
2. **Hyperparameter nhạy cảm:** Nhiều hyperparameter (λ, K_T, K_I, γ) cần tune cẩn thận.
3. **Bias từ training data:** Nếu data train chủ yếu có anomaly hình tròn → model ưu tiên shape hơn semantic thật sự.

### Hạn chế chưa được nói rõ
4. **Adapter đơn giản:** Dùng full-rank linear (W ∈ R^(d×d)), không parameter-efficient. Chưa so sánh với LoRA, bottleneck adapter.
5. **Prompts cố định:** "broken", "damaged", "with flaw" — hand-crafted, không adapt theo domain. Với y tế, "broken" có phù hợp không?
6. **Chỉ test 1 backbone:** ViT-L/14. Chưa biết hiệu quả trên ViT-B/16 hay ViT-B/32.
7. **Medical image-level yếu:** Brain MRI full-shot (80.2) thấp hơn AnomalyCLIP (83.3). Liver CT ~64-70 cho tất cả methods.
8. **Không có error bars:** Chỉ report 1 lần chạy, không có multiple runs → không rõ statistical significance.
9. **Chưa so sánh reverse order:** Chưa thử adapt visual trước rồi text sau → chưa prove text-first là tối ưu.
10. **Thiếu failure case analysis:** Không phân tích khi nào model fail.

---

## 7. Hướng cải tiến khả thi

### 7.1. Cải tiến Adapter Architecture
- **Ý tưởng:** Thay full-rank linear bằng LoRA (Low-Rank Adaptation) hoặc bottleneck adapter → giảm params, có thể giảm overfitting.
- **Lý do:** Adapter hiện tại có d² params/layer, LoRA chỉ cần 2×d×r params (r << d).

### 7.2. Learnable/Adaptive Prompts thay vì Hand-crafted
- **Ý tưởng:** Thay "broken", "damaged" bằng learnable prompt tokens tối ưu cho từng domain.
- **Lý do:** Prompts hand-crafted không phù hợp cho mọi domain (industrial vs medical rất khác).

### 7.3. Giải quyết Overfitting
- **Ý tưởng:** Thêm regularization (dropout, weight decay mạnh hơn), hoặc data augmentation ở feature level.
- **Lý do:** Full-shot < 64-shot ở image level → overfitting rõ ràng.

### 7.4. Cross-domain Training
- **Ý tưởng:** Train trên mixed data (industrial + medical) thay vì chỉ 1 domain → test generalization.
- **Lý do:** Paper chỉ train trên VisA (industrial), chưa biết train trên medical thì sao.

### 7.5. Cải tiến Disentangle Loss
- **Ý tưởng:** Thêm contrastive loss (e.g., InfoNCE) hoặc triplet loss thay vì chỉ orthogonality → enforce separation mạnh hơn.
- **Lý do:** L_dis cải thiện chỉ 0.6-0.7% → có thể loss design chưa đủ mạnh.

### 7.6. Multi-scale / Attention-based Aggregation
- **Ý tưởng:** Thay sum aggregation bằng learned attention weights cho multi-level features.
- **Lý do:** Mỗi anomaly type có scale khác nhau → learned weighting có thể tốt hơn equal sum.

### 7.7. Mở rộng Backbone
- **Ý tưởng:** Test trên nhiều backbone (ViT-B/16, ViT-B/32, SigLIP, EVA-CLIP) → verify method robustness.
- **Lý do:** Chỉ test ViT-L/14 → chưa biết method có generalize across architectures.

### 7.8. Domain-specific Text Generation
- **Ý tưởng:** Dùng LLM (GPT-4, etc.) tự động sinh anomaly descriptions phù hợp từng domain thay vì dùng template cố định.
- **Lý do:** "Broken" phù hợp cho industrial nhưng không cho medical. LLM có thể sinh "lesion", "tumor", "hemorrhage" cho y tế.

---

## 8. Tóm tắt nhanh

| Câu hỏi | Trả lời |
|----------|---------|
| **Họ làm gì?** | Cải tiến CLIP để detect anomaly zero-shot bằng two-stage adaptation |
| **Tại sao?** | CLIP bị "Anomaly Unawareness" — không phân biệt normal/anomaly text |
| **Làm cách nào?** | Stage 1: disentangle text anchors + Stage 2: align visual features. Dùng Residual Adapter (λ=0.1) |
| **Kết quả?** | SOTA trên 11 datasets, 2-shot đã vượt full-data methods ở pixel level |
| **Hạn chế chính?** | Overfitting full-shot, prompts hand-crafted, adapter chưa tối ưu, medical results yếu |
| **Cải tiến tiềm năng?** | LoRA adapter, learnable prompts, cross-domain training, attention aggregation |
