# ============================================================
# PATCH: Thêm AMP (Mixed Precision) vào train.py
# Chạy cell này NGAY SAU Cell 1 (Setup) và TRƯỚC khi train
#
# Tại sao AMP chứ không phải DataParallel?
#   - DataParallel replicate model lên CẢ 2 GPU → mỗi GPU vẫn cần full weights
#   - AMP (float16 activations) giảm ~50% memory → model fit thoải mái trên 1 GPU
#   - Với AMP: có thể tăng image_batch_size lên 4-8 thay vì 2
# ============================================================

import re

with open('train.py', 'r') as f:
    code = f.read()

# ── Patch 1: Import GradScaler ──────────────────────────────
AMP_IMPORT = "from torch.cuda.amp import autocast, GradScaler\n"
if "GradScaler" not in code:
    code = code.replace(
        "import warnings\n",
        "import warnings\n" + AMP_IMPORT
    )
    print("✅ Patch 1: Added AMP imports")

# ── Patch 2: Khởi tạo scaler trong train_image_adapter ──────
# Thêm scaler = GradScaler() ngay sau dòng "for epoch in range..."
OLD_IMG_LOOP = "    for epoch in range(start_epoch, image_epoch):\n        logger.info(f\"training image epoch {epoch}:\")\n        loss_list = []"
NEW_IMG_LOOP = "    scaler = GradScaler()  # AMP patch\n    for epoch in range(start_epoch, image_epoch):\n        logger.info(f\"training image epoch {epoch}:\")\n        loss_list = []"
if "scaler = GradScaler()" not in code:
    code = code.replace(OLD_IMG_LOOP, NEW_IMG_LOOP)
    print("✅ Patch 2: Added GradScaler init")

# ── Patch 3: Wrap forward pass với autocast ─────────────────
OLD_FORWARD = "            # forward image\n            patch_features, det_feature = model(image)\n            # calculate similarity and get prediction\n            loss = 0.0"
NEW_FORWARD = "            # forward image (AMP autocast)\n            with autocast():\n                patch_features, det_feature = model(image)\n            # calculate similarity and get prediction\n            loss = 0.0"
if "with autocast():" not in code:
    code = code.replace(OLD_FORWARD, NEW_FORWARD)
    print("✅ Patch 3: Wrapped forward with autocast")

# ── Patch 4: Thay backward + step bằng scaler version ───────
OLD_BACKWARD = "            optimizer.zero_grad()\n            loss.backward()\n            optimizer.step()\n            loss_list.append(loss.item())\n            scheduler.step()"
NEW_BACKWARD = "            optimizer.zero_grad()\n            scaler.scale(loss).backward()  # AMP patch\n            scaler.step(optimizer)           # AMP patch\n            scaler.update()                  # AMP patch\n            loss_list.append(loss.item())\n            scheduler.step()"
if "scaler.scale(loss)" not in code:
    code = code.replace(OLD_BACKWARD, NEW_BACKWARD)
    print("✅ Patch 4: Replaced backward with scaler version")

with open('train.py', 'w') as f:
    f.write(code)

print("\n✅ AMP patch hoàn thành!")
print("   → image_batch_size có thể tăng lên 4-8 mà không bị OOM trên T4 16GB")
print("   → Stage 1 (text) không dùng AMP vì forward image dùng no_grad (không cần)")
