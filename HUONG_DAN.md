# HƯỚNG DẪN TỔNG QUAN (đọc cái này là hiểu hết)

> File này giải thích: **(1) bài toán là gì → (2) cách mình giải → (3) từng script làm gì → (4) cách chạy từ A-Z → (5) các lỗi đã gặp → (6) cách tăng điểm.** Viết cho người không chuyên, đọc từ trên xuống.

---

## 1. Bài toán mình đang thi

**Cuộc thi:** Viettel AI Race 2026 – Bài 1: *BTS Digital Twin (Novel View Synthesis)*.

**Nói đơn giản:** cho một đống ảnh chụp một trạm BTS (hoặc vật thể) từ nhiều góc bằng drone. Máy tính phải "hiểu" được cảnh đó trong không gian 3D, rồi **vẽ ra ảnh ở những góc nhìn MỚI mà chưa ai chụp**.

```
Ảnh thật từ nhiều góc  ──►  Dựng lại cảnh 3D  ──►  Vẽ ảnh ở góc nhìn mới (theo yêu cầu)
   (input)                     (model)                  (output nộp bài)
```

**Người ta cho mình gì (mỗi scene):**
- `train/images/` : ~240 ảnh thật (đã biết chụp từ đâu).
- `train/sparse/0/` : kết quả COLMAP — tức là **vị trí + hướng của từng camera** và một đám điểm 3D thưa của cảnh (điểm xuất phát để dựng).
- `test/test_poses.csv` : danh sách **các góc nhìn mục tiêu** cần vẽ (mỗi dòng = 1 camera: xoay ở đâu, đứng ở đâu, tiêu cự bao nhiêu, ảnh to bao nhiêu).

**Mình phải nộp gì:** file `submission_round1.zip` chứa ảnh `.png` vẽ ra tại **đúng từng góc** trong file CSV, đúng tên, đúng kích thước.

**Chấm điểm** (so ảnh mình vẽ với ảnh thật mà BTC giữ bí mật):
```
Score = 0.4·(1 − LPIPS) + 0.3·SSIM + 0.3·PSNR_norm
```
- **LPIPS**: giống nhau về "cảm quan" (càng THẤP càng tốt).
- **SSIM**: giống nhau về cấu trúc (càng CAO càng tốt).
- **PSNR**: sai số pixel (càng CAO càng tốt).
- Điểm cuối = trung bình tất cả scene. **Thiếu scene = mất điểm.**

---

## 2. Cách mình giải: 3D Gaussian Splatting (3DGS)

Mình dùng đúng phương pháp BTC gợi ý làm baseline: **3D Gaussian Splatting** (repo chính thức của INRIA).

**3DGS là gì (hình dung nôm na):** thay vì dựng cảnh bằng lưới tam giác như game, nó biểu diễn cảnh bằng **hàng triệu hạt "đốm" mờ 3D** (gọi là Gaussian) — mỗi hạt có vị trí, màu, kích thước, độ trong. Xếp đủ nhiều hạt đúng chỗ thì nhìn từ góc nào cũng ra ảnh giống thật. Vẽ ảnh cực nhanh.

**Quy trình cho MỖI scene:**
```
1. TRAIN:   học đám Gaussian từ 240 ảnh train  → ra file model (point_cloud.ply)
2. RENDER:  đặt camera đúng từng dòng trong test_poses.csv, chụp lại từ model → ảnh PNG
3. ĐÓNG GÓI: gom ảnh 7 scene thành submission_round1.zip
```

**Điểm mấu chốt mình phải tự viết:** repo 3DGS gốc chỉ vẽ được những góc đã có ảnh. Còn đề bài yêu cầu vẽ ở **góc mới trong CSV** → mình viết thêm script đọc CSV, dựng camera, gọi bộ vẽ của 3DGS. (Chi tiết ở mục 3.)

---

## 3. Các file trong repo làm gì

```
AI-Race-2026/
├── setup_kaggle.sh          # cài môi trường + tải & build 3DGS gốc (chạy 1 lần/session)
├── scripts/
│   ├── run_all.py           # "nhạc trưởng": lặp qua 7 scene, gọi train rồi render
│   ├── prepare_pinhole.py   # xử lý dữ liệu cho hợp với 3DGS (xem mục 5)
│   ├── render_test_poses.py # ĐỌC test_poses.csv → vẽ ảnh ở đúng góc yêu cầu
│   ├── make_submission.py   # gom ảnh thành file .zip đúng format nộp bài
│   └── evaluate.py          # (chỉ khi có ảnh gốc) tự chấm điểm thử
└── third_party/
    └── gaussian-splatting/  # repo 3DGS gốc — KHÔNG sửa, chỉ gọi vào
```

**Cái nào là "não" của 3DGS?** → nằm hết trong `third_party/gaussian-splatting/` (train + bộ vẽ). Mấy script trong `scripts/` của mình chỉ là **"keo dán"**: điều phối, đọc CSV, đóng gói. Không viết lại thuật toán.

---

## 4. Cách chạy từ A-Z trên Kaggle

> Chuẩn bị 1 lần: đã upload dữ liệu thành Kaggle Dataset (Private), đã push repo lên GitHub.

**Mở Kaggle Notebook, cài đặt bên phải:** Accelerator = **GPU T4 x2**, Internet = **On**, Add Input = dataset của bạn.

### Cell 1 — cài môi trường (chạy 1 lần mỗi khi mở session mới, ~5 phút)
```bash
!git clone https://github.com/nhlich03/AI-Race-2026.git
%cd AI-Race-2026
!bash setup_kaggle.sh
```
Cuối phải in `torch ... cuda ... gpus 2`. Không thấy → lỗi, đọc log.

### Cell 2 — chạy thử 1 scene ít iter (kiểm pipeline, ~3 phút)
```bash
!python scripts/run_all.py --gs_repo third_party/gaussian-splatting \
    --data_root /kaggle/input/datasets/nhlich2003/ai-race-problem1 \
    --only HCM0421 --iterations 2000
```
Xong mở thử 1 ảnh trong `submission/HCM0421/` xem có ra cảnh không.

### Cell 3 — chạy FULL 7 scene (bản để nộp, ~1.5h với 7000 iter)
```bash
!python scripts/run_all.py --gs_repo third_party/gaussian-splatting \
    --data_root /kaggle/input/datasets/nhlich2003/ai-race-problem1 \
    --iterations 7000
```

### Cell 4 — đóng gói nộp bài
```bash
!python scripts/make_submission.py --submission submission --out /kaggle/working/submission_round1.zip
```
Tải `submission_round1.zip` ở tab **Output** → nộp lên trang thi.

**Giải thích tham số:**
| Tham số | Nghĩa |
|---|---|
| `--gs_repo` | đường dẫn tới 3DGS gốc (luôn là `third_party/gaussian-splatting`) |
| `--data_root` | thư mục chứa các scene (`HCM0421/`, `bonsai/`, …) |
| `--iterations` | số vòng train: **2000**=nháp, **7000**=nộp được, **30000**=đẹp nhất |
| `--only` | chỉ chạy vài scene (bỏ đi = chạy hết 7) |
| `--skip_trained` | bỏ qua scene đã train xong (dùng khi chạy bù nhiều phiên) |

---

## 5. Các lỗi đã gặp & cách đã fix (để hiểu tại sao có `prepare_pinhole.py`)

Dữ liệu thi có 2 chỗ khiến 3DGS gốc **báo lỗi ngay**, mình đã tự xử lý (chạy tự động trong `run_all.py`):

1. **Camera "SIMPLE_RADIAL"**: COLMAP ghi camera có 1 thông số méo ống kính nhỏ xíu (k≈0.008). 3DGS gốc chỉ nhận camera "PINHOLE" (không méo) → mình **bỏ hệ số méo tí hon đó, đổi thành PINHOLE**. Không ảnh hưởng chất lượng vì nó gần như bằng 0.
2. **File `images.bin` thừa ảnh**: nó liệt kê cả ảnh test (mà BTC không đưa file) → 3DGS mở file không có sẽ crash → mình **lọc bỏ, chỉ giữ ảnh thật sự có trên đĩa** (vd 350 → 240).

→ `prepare_pinhole.py` làm 2 việc này, tạo ra bản dữ liệu "sạch" trong thư mục `prepared/` rồi mới đưa cho 3DGS train. **Bạn không cần gọi tay — `run_all.py` tự làm.**

---

## 6. Cách tăng điểm (sau khi có baseline)

Theo thứ tự dễ→khó:
1. **Train lâu hơn: 7000 → 30000 iter.** Dễ nhất, nét hơn rõ. (~4-5h cho 7 scene, vẫn trong 1 session.)
2. **Xử lý cột anten BTS bị "bóng ma"** (cấu trúc kim loại mảnh GS dựng kém) và **nền xa mờ** — cần kỹ thuật nâng cao hơn.
3. **Thử biến thể GS mới** (Mip-Splatting chống răng cưa, 2DGS, Scaffold-GS…) nếu còn thời gian.

**Lưu ý luật thi:** chỉ dùng dữ liệu BTC cấp; ảnh phải sinh tự động 100% (không photoshop); giữ lại code + model để tái lập nếu BTC yêu cầu.

---

## Tóm tắt 1 dòng
Dùng 3D Gaussian Splatting: **train model 3D từ ảnh drone → vẽ lại ở các góc trong test_poses.csv → nén thành zip nộp**. Chạy trên Kaggle GPU, 7 scene, ~1.5h ở mức 7000 iter.
