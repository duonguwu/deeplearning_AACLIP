# Tổng hợp: Hướng dẫn & Cấu hình Huấn luyện AA-CLIP

Tài liệu này tổng hợp các cấu hình huấn luyện chuẩn được trích xuất từ Paper (AA-CLIP CVPR 2025) và các lưu ý kỹ thuật để chạy mô hình thực tế trên môi trường Kaggle, giúp bạn dễ dàng theo dõi và setup baseline cho nghiên cứu của mình.

---

## 1. Cấu hình Huấn luyện chuẩn từ Paper
Dựa vào phần Implementation Details trong Main Paper và Supplementary, dưới đây là các tham số mặc định được tác giả sử dụng (các tham số này đã được thiết lập sẵn làm giá trị default trong code `train.py`):

### 1.1. Tham số Kiến trúc (Architecture Params)
- **Model:** OpenCLIP `ViT-L-14-336px`
- **Kích thước ảnh đầu vào:** $518 \times 518$
- **Visual Features Extracted:** Lấy đầu ra từ các layers `6`, `12`, `18`, `24`.
- **Số lượng layer gắn Adapter:**
  - $K_T$ (Text encoder): `3` (Tham số code: `--text_adapt_until 3`)
  - $K_I$ (Image encoder): `6` (Tham số code: `--image_adapt_until 6`)
- **Tỉ lệ Residual ($\lambda$) & Trọng số Loss ($\gamma$):** Đều thiết lập ở mức `0.1`.

### 1.2. Cấu hình 2 Giai đoạn Huấn luyện (2-Stage Training)
Thuật toán AA-CLIP đóng băng toàn bộ backbone và chỉ train các Adapter qua 2 bước:

**Giai đoạn 1: Huấn luyện Text Adapter**
- **Mục tiêu:** Tạo các "text anchors" phân biệt rõ ràng giữa Anomaly và Normal.
- **Số Epochs:** `5` (`--text_epoch 5`)
- **Learning Rate:** $1 \times 10^{-5}$ (`--text_lr 0.00001`)

**Giai đoạn 2: Huấn luyện Image Adapter**
- **Mục tiêu:** Căn chỉnh patch-level features với các text anchors đã train ở Giai đoạn 1 để localization chính xác.
- **Số Epochs:** `20` (`--image_epoch 20`)
- **Learning Rate:** $5 \times 10^{-4}$ (`--image_lr 0.0005`)

- **Optimizer chung:** Adam (Được tác giả chạy trên 1 con GPU RTX 3090 - 24GB VRAM).

### 1.3. Giao thức Đánh giá Chéo Dataset (Cross-Dataset Zero-Shot Protocol)
Theo sát phần *4.1 Experiment Setups* trong Paper, mục tiêu của dự án là **Zero-Shot Anomaly Detection**. Điều này có nghĩa là mô hình không được phép nhìn thấy dữ liệu của tập Test trong lúc Train. Do đó, tác giả sử dụng chiến lược **Train chéo**:

1. **Khi muốn test trên 10 bộ dữ liệu (MVTec-AD, MPDD, Brain MRI, Liver CT, Retina OCT, CVC-ClinicDB...):**
   - Dữ liệu dùng để Train: **VisA**
   - Dữ liệu dùng để Test: Từng tập trong 10 tập còn lại.
2. **Khi muốn test trên bộ dữ liệu VisA:**
   - Dữ liệu dùng để Train: **MVTec-AD**
   - Dữ liệu dùng để Test: **VisA**

> ⚠️ **Quan trọng:** Nếu bạn muốn tái tạo chính xác bảng kết quả Zero-Shot của họ trên MVTec-AD, bạn **bắt buộc phải train bằng VisA** sau đó lấy trọng số đó để test trên MVTec-AD, chứ không phải train MVTec-AD rồi test luôn trên MVTec-AD.

---

## 2. Hướng dẫn chạy thực tế trên Kaggle (Đã khắc phục lỗi)

Mã nguồn gốc được thiết kế chạy trên server có GPU 24GB. Khi chạy trên Kaggle (GPU 15-16GB như T4/P100), sẽ xảy ra lỗi tràn RAM (OOM - Out Of Memory). Ngoài ra, tác giả fix cứng đường dẫn `BASE_PATH = "/data/wenxinma"`.

Vì vậy, mình đã hướng dẫn bạn và chuẩn bị sẵn file `kaggle_training.ipynb` với luồng xử lý như sau:

### 2.1. Đơn giản hóa Đường dẫn bằng `config.yaml`
Thay vì tìm và sửa từng đoạn code bị hard-code, mình đã cập nhật file `dataset/constants.py` để nó đọc tự động từ `config.yaml`:
```yaml
paths:
  base_path: "/kaggle/input" # Chỉ cần đổi dòng này trên Kaggle
  datasets:
    MVTec: "tên-thư-mục-mvtec-bạn-add-trên-kaggle"
    VisA: "tên-thư-mục-visa"
```

### 2.2. Tránh lỗi Out Of Memory (OOM)
Trong file notebook, lệnh train được giảm batch size xuống để vừa vặn với Kaggle:
```bash
!python train.py \
    --dataset MVTec \
    --training_mode full_shot \
    --shot 0 \
    --save_path /kaggle/working/AA-CLIP/ckpt/mvtec_baseline \
    --image_batch_size 2 \
    --text_batch_size 4
```
*Lưu ý: Nếu `image_batch_size=2` vẫn bị OOM trong một số session Kaggle, bạn có thể ép nó xuống `--image_batch_size 1`.*

### 2.3. Quy trình Đánh giá (Evaluation)
Sau khi script train kết thúc, các file `text_adapter.pth` và `image_adapter.pth` sẽ được lưu. Mình đã chèn sẵn cell chạy file `test.py` vào cuối Notebook:
```bash
!python test.py \
    --dataset MVTec \
    --save_path /kaggle/working/AA-CLIP/ckpt/mvtec_baseline
```
Script này sẽ tự động load test set, generate các dự đoán mask và tính ra `Image-level AUROC` cũng như `Pixel-level AUROC` để bạn trực tiếp so sánh kết quả baseline với Paper.

---
**Tóm lại:** Với `config.yaml` và `kaggle_training.ipynb`, bạn có thể Add các Dataset vào Kaggle và ấn **"Run All"** để tái tạo kết quả một cách mượt mà nhất.
