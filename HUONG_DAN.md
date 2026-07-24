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

## 1b. Phân tích dữ liệu thực tế (data có gì)

Bộ dữ liệu (`VAI_NVS_DATA_ROUND2.zip`, ~1.2GB) có **7 scene**, chia 2 loại:
- **5 scene trạm BTS thật** (`HCM0421, HCM0539, HCM0540, HCM0644, HCM0674`) — ảnh drone DJI chụp khu dân cư dày đặc ở TP.HCM, tên ảnh kiểu `DJI_2024...._V.JPG`.
- **2 scene benchmark** (`bonsai, chair`) — vật thể chụp cận, tên ảnh kiểu `frame_000xxx.jpg`.

| Scene | Loại | Ảnh train | Góc test (phải vẽ) | Camera model | Kích thước ảnh | Tiêu cự fx | Méo (k) |
|---|---|---:|---:|---|---|---:|---:|
| HCM0421 | BTS drone | 240 | 60 | SIMPLE_RADIAL | 1320×989 | 926.4 | 0.0089 |
| HCM0539 | BTS drone | 240 | 60 | SIMPLE_RADIAL | 1320×989 | 925.4 | 0.0081 |
| HCM0540 | BTS drone | 240 | 60 | SIMPLE_RADIAL | 1320×989 | 926.7 | 0.0089 |
| HCM0644 | BTS drone | 240 | 60 | SIMPLE_RADIAL | 1320×989 | 925.5 | 0.0090 |
| HCM0674 | BTS drone | 240 | 60 | SIMPLE_RADIAL | 1320×989 | 925.3 | 0.0088 |
| bonsai | benchmark | 248 | 28 | SIMPLE_PINHOLE | 1920×1080 | 1650.0 | – |
| chair | benchmark | 205 | 58 | SIMPLE_PINHOLE | 720×1280 | 1114.0 | – |
| **Tổng** | | **1653** | **386** | | | | |

**Cấu trúc 1 scene:**
```
HCM0421/
├── README.txt                    # mô tả ngắn (train/test 80/20, scale 1/4)
├── train/
│   ├── images/                   # 240 ảnh .JPG
│   └── sparse/0/                 # kết quả COLMAP:
│       ├── cameras.bin           #   thông số camera (tiêu cự, méo…)
│       ├── images.bin            #   vị trí + hướng từng camera
│       ├── points3D.bin/.ply     #   đám điểm 3D thưa (init cho 3DGS)
│       └── rigs.bin, frames.bin  #   metadata COLMAP bản mới (không dùng tới)
└── test/
    └── test_poses.csv            # danh sách góc nhìn cần vẽ (KHÔNG có ảnh gốc)
```

**Những điểm quan trọng rút ra từ data:**
1. **Phải vẽ tổng cộng 386 ảnh** (5 scene BTS × 60 + bonsai 28 + chair 58). Thiếu bất kỳ ảnh nào của scene nào = scene đó mất điểm.
2. **`test/` KHÔNG có ảnh gốc** → không tự chấm điểm trực tiếp được. Chỉ soi ảnh bằng mắt, hoặc giữ lại một phần train làm tập kiểm để ước lượng.
3. **Điểm chính (cx, cy) của mọi scene đều nằm đúng tâm ảnh** (vd 660 = 1320/2, 494.5 = 989/2) → thuận lợi, bộ vẽ 3DGS gốc render đúng, không bị lệch.
4. **5 scene BTS có méo ống kính rất nhỏ (k ≈ 0.008–0.009)** → mình bỏ đi, coi như PINHOLE (xem mục 5). 2 scene benchmark vốn đã không méo.
5. **`images.bin` liệt kê nhiều ảnh hơn số file thật** (vd HCM0421: 350 đăng ký nhưng chỉ 240 file) vì nó chứa cả ảnh test + ảnh COLMAP đăng ký nhưng BTC không phát → mình lọc bỏ (xem mục 5).
6. **Scale 1/4**: ảnh đã được BTC thu nhỏ còn 1/4 gốc → nhẹ, train nhanh.

---

## 1c. Đám điểm 3D (point cloud) trong `sparse/0/` từ đâu ra?

Trong `train/sparse/0/` có một **point cloud** (đám mây điểm 3D). Nó **không phải mình tạo** — BTC đã chạy sẵn công cụ **COLMAP** trên bộ ảnh train bằng kỹ thuật **SfM (Structure from Motion)** — "dựng cấu trúc 3D từ nhiều ảnh".

**COLMAP làm 4 bước:**
```
1. Tìm đặc trưng : trên mỗi ảnh dò các điểm nổi bật (góc mái, mép cửa, cạnh anten…)
2. Ghép cặp       : điểm ở ảnh A trùng điểm nào ở ảnh B,C… (cùng 1 vật thể thật)
3. Tam giác hóa   : 1 điểm thấy từ ≥2 camera đã biết góc → giao 2 tia → ra tọa độ 3D (x,y,z)
4. Tối ưu tổng thể: chỉnh lại đồng thời vị trí camera + điểm 3D cho khớp nhất
```

**COLMAP cho ra 2 thứ (đều trong `sparse/0/`):**
- Vị trí + hướng từng camera (`cameras.bin`, `images.bin`).
- **Đám điểm 3D** (`points3D.bin`/`.ply`) — point cloud cần hỏi.

**Nó như nào:**
- Thưa (**sparse**): chỉ vài trăm nghìn điểm (vd HCM0421 = **171.304 điểm** — chính là dòng `Number of points at initialisation: 171304` khi train). Đủ phác hình, chưa phải bề mặt kín — nhìn như đám bụi 3D lờ mờ ra dáng cảnh.
- Mỗi điểm có tọa độ **(x, y, z)** + **màu RGB** (lấy từ pixel ảnh gốc).

**Liên quan gì tới 3DGS → nó là ĐIỂM XUẤT PHÁT:**
```
Point cloud COLMAP (171k điểm thưa)   →  TRAIN 3DGS  →  vài triệu Gaussian dày đặc
= "hạt giống" ban đầu                     (nhiều iter)     = model vẽ ảnh đẹp
```
- 3DGS không train từ số 0: mỗi điểm 3D thành **1 hạt Gaussian ban đầu** (đúng chỗ, sẵn màu).
- Khi train, nó **nhân bản & tách nhỏ** hạt ở chỗ thiếu chi tiết (*densification*) → từ 171k điểm nở ra hàng triệu Gaussian.
- Chỗ COLMAP ít điểm (cột anten mảnh, bầu trời) chính là chỗ 3DGS hay lỗi "bóng ma" — đúng như ảnh render thấy lúc test.

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
