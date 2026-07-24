# Khanh_readme — Ghi chú kỹ thuật repo VAR-2026 Bài 1 (3DGS)

> Dành cho người đã nắm 3DGS/COLMAP/NeRF. Mục tiêu: hiểu repo này đang làm gì, kiến trúc,
> chỗ nào là code gốc / chỗ nào tự viết, cách chạy, và các quyết định kỹ thuật (kèm cái bẫy 350MB).

## 0. TL;DR

Repo là **lớp wrapper mỏng** quanh 3DGS gốc ([graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)), **không fork/sửa** repo gốc. Với mỗi scene: tiền xử lý COLMAP → `train.py` gốc (per-scene, no eval holdout) → render tại các pose trong `test_poses.csv` bằng `MiniCam` → đóng gói JPEG zip ≤350MB.

Phương pháp = baseline BTC gợi ý (vanilla 3DGS). Chưa có cải tiến thuật toán — đây là điểm để mình bàn sau.

## 1. Bài toán & chấm điểm

Novel View Synthesis cho scene trạm BTS (drone) + vài scene benchmark. Input mỗi scene: `train/images/` + `train/sparse/0/` (COLMAP), `test/test_poses.csv` (pose mục tiêu, **không có GT ảnh**). Output: render RGB tại các pose đó.

```
Score = 0.4·(1 − LPIPS) + 0.3·SSIM + 0.3·PSNR_norm,   PSNR_norm = clamp(PSNR/PSNR_max, 0, 1)
```
`PSNR_max` BTC không công bố. Điểm = trung bình các scene; **thiếu/thừa scene ⇒ scene đó = 0**. Deadline vòng 1: **30/07/2026**.

## 2. Kiến trúc repo

```
scripts/
  run_all.py            # orchestrator: loop scene → prepare → train → render
  prepare_pinhole.py    # tiền xử lý COLMAP cho hợp reader 3DGS (mục 4)
  render_test_poses.py  # CORE: render tại pose từ CSV (mục 5)
  make_submission.py    # zip submission (giữ nguyên cấu trúc)
  compress_submission.py# nén JPEG về ≤350MB (mục 7 — cái bẫy dung lượng)
  evaluate.py           # tự chấm LPIPS/SSIM/PSNR nếu có GT (public set)
third_party/
  gaussian-splatting/   # repo gốc, clone nguyên bản (setup_kaggle.sh), KHÔNG commit
```

Code gốc lo train + rasterization CUDA. `scripts/` chỉ là glue: điều phối, đọc CSV, đóng gói. Không reimplement thuật toán.

## 3. Dữ liệu thực tế (đã kiểm bằng cách parse .bin)

7 scene: `HCM0421/0539/0540/0644/0674` (drone BTS, ảnh DJI) + `bonsai, chair` (benchmark).

| Scene | Train | Test poses | Camera model | Resolution | Ghi chú |
|---|--:|--:|---|---|---|
| HCM04xx/06xx (×5) | 240 | 60 | SIMPLE_RADIAL (k≈0.008) | 1320×989 | cx,cy = tâm ảnh |
| bonsai | 248 | 28 | SIMPLE_PINHOLE | 1920×1080 | |
| chair | 205 | 58 | SIMPLE_PINHOLE | 720×1280 | |

Tổng phải render: **386 ảnh**. `sparse/0/` là COLMAP format mới (có thêm `rigs.bin`, `frames.bin`, `points3D.ply` — reader 3DGS chỉ đọc `cameras/images/points3D.bin`, phần còn lại bỏ qua). Init point cloud vd HCM0421 = 171k điểm.

## 4. Tiền xử lý (`prepare_pinhole.py`) — vì sao cần

Reader COLMAP của 3DGS gốc `assert` chết ở 2 chỗ với data này:

1. **Camera model SIMPLE_RADIAL** không nằm trong {PINHOLE, SIMPLE_PINHOLE}. k≈0.008 (bỏ qua được) và `test_poses.csv` cho intrinsics pinhole thuần → mình rewrite `cameras.bin` thành PINHOLE (`fx=fy=f`, bỏ k). Nếu về sau gặp scene méo nặng thì phải undistort thật (COLMAP `image_undistorter`), không dùng cách này.
2. **`images.bin` đăng ký nhiều ảnh hơn số file trên đĩa** (gồm cả test views BTC không phát: HCM0421 350 vs 240 file). Reader mở file thiếu → crash. Mình lọc `images.bin`, chỉ giữ record có file tương ứng.

Output: source dir mới (`prepared/<scene>`) — `images/` symlink, `cameras.bin`/`images.bin` viết lại, `points3D.bin` reuse. `run_all.py` gọi tự động, train.py trỏ vào dir này.

## 5. Render tại pose CSV (`render_test_poses.py`) — phần lõi tự viết

Vấn đề: 3DGS gốc chỉ render camera do COLMAP loader nạp; pose mục tiêu lại nằm ở CSV. Giải:

- Đọc mỗi dòng CSV: `qw,qx,qy,qz` → rotation (COLMAP world-to-cam), `tx,ty,tz` → translation, `fx,fy` → FoV qua `focal2fov`, `w,h` → kích thước render.
- Quy ước khớp 3DGS: `R = qvec2rotmat(q).T`, `T = tvec`, dựng `world_view` bằng `getWorld2View2`, `proj` bằng `getProjectionMatrix`, ghép `full_proj`.
- Bọc vào **`MiniCam`** (class có sẵn của repo gốc, vốn dùng cho interactive viewer) rồi gọi `gaussian_renderer.render()` gốc. Load Gaussian bằng `GaussianModel.load_ply(point_cloud/iteration_N/point_cloud.ply)`.

**Giới hạn đã biết:** rasterizer gốc giả định principal point ở tâm ảnh → mình bỏ `cx,cy`. May là data này cx,cy = tâm nên vô hại. Nếu về sau lệch tâm: cần rasterizer fork hỗ trợ principal point, hoặc render lớn + crop.

**Filename:** `--name_mode exact` (mặc định) ghi đúng `image_name` từ CSV (đuôi `.JPG`). Nội dung mã hoá theo `--img_format` (mặc định **jpeg q95**, xem mục 7).

## 6. Chạy

**Kaggle (T4 x2, Internet On):**
```bash
git clone https://github.com/nhlich03/AI-Race-2026.git && cd AI-Race-2026
bash setup_kaggle.sh          # clone 3DGS gốc + build diff-gaussian-rasterization/simple-knn (arch 7.5)
python scripts/run_all.py --gs_repo third_party/gaussian-splatting \
    --data_root /kaggle/input/<slug>/  --iterations 7000     # 7000 baseline, 30000 full
python scripts/make_submission.py --submission submission --out submission_round1.zip
```
3DGS single-GPU; T4 thứ 2 idle → có thể chia scene thủ công qua `--gpu 0/1 --only ...`. Thời gian ~ 10-13 phút/scene @7000, ~35-50 phút/scene @30000 (bonsai chậm nhất do 1080p). Env đã verify: torch 2.10.0+cu128, CUDA 12.8.

Tham số đáng chú ý `run_all.py`: `--iterations`, `--only`, `--skip_trained` (bỏ scene đã train xong — chỉ dựa trên sự tồn tại của `iteration_N/point_cloud.ply`, **không resume train dở**), `--img_format {png,jpeg}`, `--jpeg_quality`.

## 7. ⚠️ Bẫy dung lượng 350MB — ĐỌC KỸ

**Portal cap 350MB/lần nộp, và KHÔNG chia scene qua nhiều lần được** (thiếu scene = 0 điểm) → toàn bộ 386 ảnh phải nằm trong 1 zip ≤350MB.

- Render **PNG lossless** = **~720MB** → vượt cap.
- Fix: xuất **JPEG q95** ≈ **200MB**. Sai số thêm ~48dB PSNR, chìm dưới sai số render → gần như không đổi điểm. Đuôi file vốn là `.JPG` nên JPEG còn đúng format hơn.

Đã set `run_all.py` **mặc định `--img_format jpeg --jpeg_quality 95`** → các lần chạy sau tự ra ~200MB, khỏi xử lý tay.

**Nếu lỡ có zip/folder PNG cũ (720MB), nén lại không cần chạy Kaggle:**
```bash
python scripts/compress_submission.py --src submission_round1.zip --out fixed.zip --max_mb 350
# --src nhận cả folder submission/ lẫn file .zip; tự hạ quality 95→92→90… tới khi ≤ max_mb
```

## 8. Trạng thái hiện tại & hướng cải tiến

**Đã có:** baseline full 7 scene @7000 iter, render tốt cả BTS lẫn benchmark, nộp JPEG q95 (~200MB). Đây mới là **vanilla 3DGS**, chưa tối ưu gì.

**Hướng đẩy điểm (từ dễ → cần đọc paper):**
1. `--iterations 30000` (mặc định 3DGS) — win dễ nhất.
2. Cấu trúc mảnh (cột anten) bị floaters + nền xa mờ → thử: densification/opacity reset tuning, depth regularization (bản 3DGS mới có `--depth_l1_weight`), hoặc anti-aliasing.
3. Chênh sáng giữa ảnh drone → per-image exposure / appearance embedding (`--use_trained_exp` ở bản mới, hoặc kiểu GLO/Wild-GS).
4. Biến thể: **Mip-Splatting** (chống aliasing đa tỉ lệ — hợp với đổi resolution/khoảng cách camera), **2DGS** (bề mặt sắc hơn), **Scaffold-GS** (ít floater vùng thưa). Cân nhắc theo thời gian còn lại.

**Ràng buộc luật:** chỉ data BTC; sinh ảnh tự động 100%; giữ code + checkpoint + log để tái lập.

## 9. Lưu ý vận hành

- `.gitignore` chặn `data/`, `output/`, `third_party/`, `*.zip` — không commit data/model.
- File `.sh` ép LF qua `.gitattributes` (tránh CRLF lỗi trên Kaggle).
- Không có GT cho test → muốn ước lượng điểm phải tự holdout (vd `train.py --eval` giữ 1/8 ảnh) rồi dùng `evaluate.py` / `metrics.py` của repo gốc.
