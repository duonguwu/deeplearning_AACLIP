# AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP
## Paper Summary — CVPR 2025

**Authors:** Wenxin Ma, Xu Zhang, Qingsong Yao, Fenghe Tang, Chenxu Wu, Yingtai Li, Rui Yan, Zihang Jiang, S. Kevin Zhou  
**Affiliations:** USTC, Stanford University  
**Code:** [github.com/Mwxinnn/AA-CLIP](https://github.com/Mwxinnn/AA-CLIP)

---

### 1. Motivation

CLIP đạt kết quả tốt trong zero-shot tasks nhờ generalization mạnh, nhưng có vấn đề **Anomaly Unawareness**: text embeddings của "normal" và "anomaly" prompts gần như trùng nhau trong feature space. Nguyên nhân: CLIP được pre-train ở category level (e.g., "a photo of a cat"), không bao giờ thấy fine-grained anomaly descriptions. Hệ quả: CLIP-based anomaly detection methods dùng text features làm anchor sẽ không phân biệt được normal/abnormal regions.

---

### 2. Phương pháp: AA-CLIP

AA-CLIP sử dụng **two-stage adaptation** với **Residual Adapters** để inject anomaly awareness vào CLIP mà vẫn giữ generalization:

| | Stage 1: Text Adaptation | Stage 2: Visual Adaptation |
|---|---|---|
| **Mục tiêu** | Tạo text anchors T_N, T_A tách biệt | Align patch features với text anchors |
| **Trainable** | Adapters ở 3 shallow layers text encoder + projector | Adapters ở 6 shallow layers visual encoder + 4 projectors |
| **Frozen** | Visual encoder + deep text layers | Adapted text encoder + deep visual layers |
| **Loss** | L_cls + L_seg + γ·L_dis | L_cls + L_seg |
| **Config** | 5 epochs, lr=1e-5 | 20 epochs, lr=5e-4 |

**Residual Adapter:** `x_enhanced = λ·Norm(Act(W·x)) + (1-λ)·x` với λ=0.1 → chỉ blend 10% adapted features, giữ 90% CLIP gốc.

**Disentangle Loss:** `L_dis = |⟨T_N, T_A⟩|²` — enforce normal/anomaly anchors orthogonal.

**Multi-level aggregation:** Features từ layer {6, 12, 18, 24} → project + sum → V_patch.

**Inference:** CosSim(visual features, [T_N, T_A]) → anomaly score (image) + anomaly map (pixel).

---

### 3. Kết quả chính

**Setup:** Train trên VisA, zero-shot test trên 11 datasets (4 industrial + 7 medical). Backbone: OpenCLIP ViT-L/14, input 518×518.

| Method | Venue | Pixel AUROC (avg) | Image AUROC (avg) |
|--------|-------|-------------------|--------------------|
| CLIP | - | 49.9 | 68.3 |
| WinCLIP | CVPR'23 | 74.7 | 67.8 |
| AnomalyCLIP | ICLR'24 | 91.3 | 78.4 |
| AdaCLIP | ECCV'24 | 90.4 | 80.6 |
| **AA-CLIP (2-shot)** | - | **92.0** | 78.1 |
| **AA-CLIP (64-shot)** | - | **92.8** | **83.1** |
| **AA-CLIP (full)** | - | **93.4** | 82.5 |

- **2-shot** đã vượt tất cả prior methods (full data) ở pixel level
- **64-shot** đạt SOTA cả 2 levels
- Full-shot image AUROC (82.5) < 64-shot (83.1) → dấu hiệu overfitting

---

### 4. Ablation highlights

| Component | Pixel AUROC | Image AUROC |
|-----------|-------------|-------------|
| CLIP baseline | 50.3 | 69.3 |
| + Linear Proj (VAND) | 88.9 | 69.3 |
| + Vanilla Adapter (no residual) | 48.9 (**-40.0**) | 53.4 |
| + **Residual Adapter** | 91.3 | 80.7 |
| + **Text Adapter** | 92.1 | 82.6 |
| + **Disentangle Loss** | 92.7 | 83.3 |

- Vanilla adapter **phá hủy** CLIP (giảm 40 điểm pixel AUROC)
- Residual mechanism critical cho preservation of pretrained knowledge
- One-stage training → class information collapse (Fig. 7)

---

### 5. Đánh giá

**Strengths:** Problem formulation rõ ràng, method đơn giản hiệu quả, data-efficient (2-shot competitive), evaluation toàn diện 11 datasets.

**Limitations:** Hyperparameters heuristic (λ, K_T, K_I, γ), anomaly prompts hand-crafted, overfitting ở full-shot, medical image-level chưa strong, chưa thử backbone khác hay adapter architectures khác (LoRA, bottleneck).

**Novelty:** Chủ yếu ở problem identification (Anomaly Unawareness) + two-stage strategy. Technical components (residual adapters, orthogonality loss) đều well-known.

---

### Key Takeaway

> Trước khi align visual features, hãy fix text anchors trước. Nếu "normal" ≈ "anomaly" trong text space, mọi visual alignment đều vô nghĩa. AA-CLIP chứng minh rằng chỉ cần disentangle text space + controlled residual adaptation là đủ để CLIP trở nên anomaly-aware, với rất ít data.
