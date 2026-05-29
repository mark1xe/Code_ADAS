#!/usr/bin/env python3
"""
train_yolo_traffic.py — Train YOLOv8 nhận biết biển giao thông + người
=======================================================================
Nhãn:
  0: stop_sign      — biển dừng (bát giác đỏ STOP)
  1: no_entry       — biển cấm vào (tròn đỏ, vạch trắng ngang)
  2: red_light      — đèn đỏ
  3: yellow_light   — đèn vàng
  4: green_light    — đèn xanh
  5: person         — người đi bộ

Cấu trúc dataset yêu cầu (YOLO format):
  dataset/
    images/
      train/   *.jpg / *.png
      val/     *.jpg / *.png
    labels/
      train/   *.txt   (mỗi dòng: class cx cy w h — chuẩn hoá 0-1)
      val/     *.txt

Cài đặt:
  pip install ultralytics opencv-python pyyaml

Chạy:
  python3 train_yolo_traffic.py                    # train mặc định
  python3 train_yolo_traffic.py --epochs 100       # train 100 epoch
  python3 train_yolo_traffic.py --model yolov8s    # model lớn hơn
  python3 train_yolo_traffic.py --data /path/to/dataset
  python3 train_yolo_traffic.py --export-only      # chỉ export ONNX (đã có weights)
  python3 train_yolo_traffic.py --verify-dataset   # kiểm tra dataset trước khi train

Export ONNX sau khi train:
  python3 train_yolo_traffic.py --export-only --weights runs/train/exp/weights/best.pt
"""

import os, sys, shutil, argparse, time, textwrap
from pathlib import Path

# ─── Hằng số mặc định ────────────────────────────────────────────────────────
CLASSES = [
    "stop_sign",    # 0
    "no_entry",     # 1
    "red_light",    # 2
    "yellow_light", # 3
    "green_light",  # 4
    "person",       # 5
]
NUM_CLASSES   = len(CLASSES)

MODEL_NAME    = "yolov8n"   # nano — nhẹ nhất, chạy được trên RPi/Jetson
                             # yolov8s = small, yolov8m = medium
EPOCHS        = 50
IMG_SIZE      = 640
BATCH_SIZE    = 16           # giảm xuống 8 nếu thiếu RAM GPU
DATASET_DIR   = Path("dataset")
RUNS_DIR      = Path("runs/train")
PROJECT_NAME  = "traffic_detector"

# Augmentation mạnh hơn cho biển nhỏ và đèn giao thông
AUGMENT_CFG = dict(
    hsv_h       = 0.015,   # jitter màu sắc nhẹ
    hsv_s       = 0.5,
    hsv_v       = 0.4,
    degrees     = 5.0,     # xoay nhẹ — biển thường không nghiêng nhiều
    translate   = 0.1,
    scale       = 0.5,     # scale up/down 50% — quan trọng cho object nhỏ
    shear       = 2.0,
    perspective = 0.0005,
    flipud      = 0.0,     # KHÔNG lật dọc — biển giao thông không bao giờ ngược
    fliplr      = 0.5,     # lật ngang OK
    mosaic      = 1.0,     # mosaic augmentation — rất hiệu quả cho small objects
    mixup       = 0.1,
    copy_paste  = 0.1,     # copy-paste augmentation
)


# ─── Tạo file dataset.yaml ────────────────────────────────────────────────────
def create_dataset_yaml(dataset_dir: Path) -> Path:
    """Tạo file YAML mô tả dataset cho Ultralytics."""
    import yaml
    yaml_path = dataset_dir / "dataset.yaml"
    cfg = {
        "path"  : str(dataset_dir.resolve()),
        "train" : "images/train",
        "val"   : "images/val",
        "nc"    : NUM_CLASSES,
        "names" : CLASSES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"[✓] Đã tạo: {yaml_path}")
    return yaml_path


# ─── Kiểm tra dataset ─────────────────────────────────────────────────────────
def verify_dataset(dataset_dir: Path) -> bool:
    """Kiểm tra cấu trúc dataset và thống kê nhãn."""
    print("\n═══ KIỂM TRA DATASET ═══")
    ok = True

    for split in ["train", "val"]:
        img_dir = dataset_dir / "images"  / split
        lbl_dir = dataset_dir / "labels"  / split

        if not img_dir.exists():
            print(f"[✗] Thiếu thư mục: {img_dir}")
            ok = False; continue
        if not lbl_dir.exists():
            print(f"[✗] Thiếu thư mục: {lbl_dir}")
            ok = False; continue

        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + \
               list(img_dir.glob("*.jpeg"))
        lbls = list(lbl_dir.glob("*.txt"))
        print(f"\n[{split}]  ảnh={len(imgs):5d}  nhãn={len(lbls):5d}")

        if len(imgs) == 0:
            print(f"  [!] Không có ảnh trong {img_dir}")
            ok = False

        # Đếm số lượng từng class
        counts = [0] * NUM_CLASSES
        errors = 0
        for lbl_file in lbls:
            try:
                with open(lbl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) != 5:
                            errors += 1; continue
                        cls_id = int(parts[0])
                        if 0 <= cls_id < NUM_CLASSES:
                            counts[cls_id] += 1
                        else:
                            print(f"  [!] class ID {cls_id} không hợp lệ: {lbl_file.name}")
                            errors += 1
            except Exception as e:
                print(f"  [!] Lỗi đọc {lbl_file.name}: {e}")
                errors += 1

        print("  Phân bổ nhãn:")
        for i, (cls, cnt) in enumerate(zip(CLASSES, counts)):
            bar = "█" * min(cnt // 10, 40)
            print(f"    [{i}] {cls:<15} {cnt:6d}  {bar}")

        if errors > 0:
            print(f"  [!] {errors} dòng nhãn bị lỗi format")
        if sum(counts) == 0:
            print(f"  [!] Không tìm thấy nhãn hợp lệ")
            ok = False

        # Kiểm tra ảnh không có nhãn tương ứng
        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        unmatched = img_stems - lbl_stems
        if unmatched:
            print(f"  [!] {len(unmatched)} ảnh không có file nhãn tương ứng")

    if ok:
        print("\n[✓] Dataset hợp lệ")
    else:
        print("\n[✗] Dataset có vấn đề — hãy sửa trước khi train")
    return ok


# ─── Tạo dataset mẫu (demo) ──────────────────────────────────────────────────
def create_sample_dataset(dataset_dir: Path):
    """
    Tạo dataset mẫu với ảnh giả để test pipeline.
    Trong thực tế, thay bằng ảnh thật của bạn.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[!] Cần opencv-python: pip install opencv-python")
        return

    print("\n[i] Tạo dataset mẫu (ảnh synthetic)...")
    print("    → Trong thực tế, thay bằng ảnh thật đã annotate!")

    COLORS_BG = [(50,50,50), (100,100,100), (200,200,200)]
    OBJ_COLORS = {
        0: (0,0,200),    # stop_sign — đỏ
        1: (0,0,180),    # no_entry  — đỏ
        2: (0,0,255),    # red_light
        3: (0,165,255),  # yellow_light
        4: (0,255,0),    # green_light
        5: (255,200,100),# person    — da
    }

    np.random.seed(42)
    for split, n_imgs in [("train", 80), ("val", 20)]:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(n_imgs):
            img = np.ones((480, 640, 3), dtype=np.uint8)
            bg = COLORS_BG[idx % len(COLORS_BG)]
            img[:] = bg

            n_objs = np.random.randint(1, 4)
            labels = []
            for _ in range(n_objs):
                cls = np.random.randint(0, NUM_CLASSES)
                w = np.random.randint(40, 120)
                h = np.random.randint(40, 120)
                x = np.random.randint(0, 640 - w)
                y = np.random.randint(0, 480 - h)
                color = OBJ_COLORS[cls]
                cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
                cv2.putText(img, CLASSES[cls][:4], (x+2, y+h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                # YOLO format: cx cy w h (chuẩn hoá)
                cx = (x + w/2) / 640
                cy = (y + h/2) / 480
                nw = w / 640
                nh = h / 480
                labels.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            cv2.imwrite(str(img_dir / f"sample_{idx:04d}.jpg"), img)
            with open(lbl_dir / f"sample_{idx:04d}.txt", "w") as f:
                f.write("\n".join(labels))

    print(f"[✓] Đã tạo dataset mẫu: {dataset_dir}")


# ─── Export ONNX ─────────────────────────────────────────────────────────────
def export_onnx(weights_path: Path, img_size: int = IMG_SIZE) -> Path:
    """Export model PyTorch → ONNX để dùng với OpenCV / cv_lane_driver."""
    from ultralytics import YOLO

    print(f"\n═══ EXPORT ONNX ═══")
    print(f"  Weights : {weights_path}")
    print(f"  ImgSize : {img_size}x{img_size}")

    model = YOLO(str(weights_path))
    onnx_path = model.export(
        format   = "onnx",
        imgsz    = img_size,
        opset    = 12,          # opset 12 tương thích tốt nhất với OpenCV 4.x
        simplify = True,        # onnx-simplifier — giảm node dư thừa
        dynamic  = False,       # fixed batch=1 cho inference real-time
        half     = False,       # FP32 để tương thích RPi (không có GPU)
    )
    print(f"[✓] ONNX đã lưu: {onnx_path}")
    print(f"\n  Dùng trong cv_lane_driver:")
    print(f"    --yolo-model {onnx_path}")
    return Path(onnx_path)


# ─── Train chính ─────────────────────────────────────────────────────────────
def train(args):
    from ultralytics import YOLO

    dataset_dir = Path(args.data)
    yaml_path   = dataset_dir / "dataset.yaml"

    # Tạo YAML nếu chưa có
    if not yaml_path.exists():
        if not dataset_dir.exists():
            print(f"[!] Không tìm thấy dataset: {dataset_dir}")
            print("    → Tạo dataset mẫu để test? (y/n): ", end="")
            if input().strip().lower() == "y":
                create_sample_dataset(dataset_dir)
            else:
                print("    → Hãy chuẩn bị dataset theo cấu trúc:")
                print(textwrap.dedent("""
                    dataset/
                      images/train/*.jpg
                      images/val/*.jpg
                      labels/train/*.txt   (YOLO format: class cx cy w h)
                      labels/val/*.txt
                """))
                sys.exit(1)
        create_dataset_yaml(dataset_dir)

    # Kiểm tra dataset
    if args.verify_dataset or not args.skip_verify:
        ok = verify_dataset(dataset_dir)
        if not ok and not args.force:
            print("\n[!] Dataset có vấn đề. Thêm --force để train dù sao.")
            sys.exit(1)

    print(f"\n═══ BẮT ĐẦU TRAIN ═══")
    print(f"  Model   : {args.model}.pt (pretrained COCO)")
    print(f"  Epochs  : {args.epochs}")
    print(f"  ImgSize : {args.imgsz}x{args.imgsz}")
    print(f"  Batch   : {args.batch}")
    print(f"  Dataset : {dataset_dir.resolve()}")
    print(f"  Classes : {CLASSES}")
    print()

    # Load model pretrained (tự download lần đầu)
    model = YOLO(f"{args.model}.pt")

    # Train
    results = model.train(
        data        = str(yaml_path),
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        project     = str(RUNS_DIR),
        name        = PROJECT_NAME,
        exist_ok    = True,
        # ── Augmentation ────────────────────────────────────────
        **AUGMENT_CFG,
        # ── Optimizer ───────────────────────────────────────────
        optimizer   = "AdamW",      # AdamW tốt hơn SGD cho fine-tune
        lr0         = 0.001,
        lrf         = 0.01,
        momentum    = 0.937,
        weight_decay= 0.0005,
        warmup_epochs    = 3,
        warmup_momentum  = 0.8,
        # ── Loss weights (tăng box và cls cho object nhỏ) ───────
        box         = 7.5,
        cls         = 0.5,
        dfl         = 1.5,
        # ── Misc ────────────────────────────────────────────────
        patience    = 30,           # early stopping sau 30 epoch không cải thiện
        save_period = 10,           # lưu checkpoint mỗi 10 epoch
        plots       = True,         # vẽ confusion matrix, PR curve
        verbose     = True,
        device      = args.device,
    )

    # Tìm best weights
    best_weights = RUNS_DIR / PROJECT_NAME / "weights" / "best.pt"
    if not best_weights.exists():
        # fallback
        best_weights = list((RUNS_DIR / PROJECT_NAME / "weights").glob("best*.pt"))
        best_weights = best_weights[0] if best_weights else None

    print(f"\n═══ KẾT QUẢ ═══")
    if best_weights and best_weights.exists():
        print(f"[✓] Best weights: {best_weights}")

        # Validate
        print("\n[i] Chạy validation trên best.pt...")
        val_model = YOLO(str(best_weights))
        metrics   = val_model.val(data=str(yaml_path), imgsz=args.imgsz)
        print(f"  mAP50      : {metrics.box.map50:.4f}")
        print(f"  mAP50-95   : {metrics.box.map:.4f}")
        print(f"  Precision  : {metrics.box.mp:.4f}")
        print(f"  Recall     : {metrics.box.mr:.4f}")

        # Export ONNX
        if not args.no_export:
            onnx_path = export_onnx(best_weights, args.imgsz)
            print(f"\n[✓] File ONNX sẵn sàng: {onnx_path}")
            print(f"    Dùng ngay:")
            print(f"    python3 cv_lane_driver_v2.py --yolo-model {onnx_path}")
    else:
        print("[!] Không tìm được best.pt")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Train YOLOv8 nhận biết biển giao thông và người",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--data",           default=str(DATASET_DIR),
                    help=f"Thư mục dataset (mặc định: {DATASET_DIR})")
    ap.add_argument("--model",          default=MODEL_NAME,
                    choices=["yolov8n","yolov8s","yolov8m","yolov8l","yolov8x"],
                    help="Model size (n=nano, s=small, m=medium — mặc định: yolov8n)")
    ap.add_argument("--epochs",         type=int,   default=EPOCHS,
                    help=f"Số epoch train (mặc định: {EPOCHS})")
    ap.add_argument("--imgsz",          type=int,   default=IMG_SIZE,
                    help=f"Kích thước ảnh (mặc định: {IMG_SIZE})")
    ap.add_argument("--batch",          type=int,   default=BATCH_SIZE,
                    help=f"Batch size (mặc định: {BATCH_SIZE})")
    ap.add_argument("--device",         default="",
                    help="Device: '' (auto), 'cpu', '0' (GPU 0), 'mps' (Apple)")
    ap.add_argument("--verify-dataset", action="store_true",
                    help="Chỉ kiểm tra dataset, không train")
    ap.add_argument("--skip-verify",    action="store_true",
                    help="Bỏ qua bước verify dataset")
    ap.add_argument("--force",          action="store_true",
                    help="Train dù dataset có lỗi")
    ap.add_argument("--no-export",      action="store_true",
                    help="Không export ONNX sau khi train")
    ap.add_argument("--export-only",    action="store_true",
                    help="Chỉ export ONNX (cần --weights)")
    ap.add_argument("--weights",        default=None,
                    help="Path đến .pt weights khi dùng --export-only")
    ap.add_argument("--sample-dataset", action="store_true",
                    help="Tạo dataset mẫu synthetic để test pipeline")
    args = ap.parse_args()

    # ── Mode: chỉ tạo dataset mẫu ──────────────────────────────────────────
    if args.sample_dataset:
        create_sample_dataset(Path(args.data))
        create_dataset_yaml(Path(args.data))
        verify_dataset(Path(args.data))
        return

    # ── Mode: chỉ verify ───────────────────────────────────────────────────
    if args.verify_dataset and not args.export_only:
        if not Path(args.data).exists():
            print(f"[!] Không tìm thấy: {args.data}")
            sys.exit(1)
        verify_dataset(Path(args.data))
        return

    # ── Mode: chỉ export ONNX ─────────────────────────────────────────────
    if args.export_only:
        if args.weights is None:
            # Tự tìm best.pt gần nhất
            candidates = list(RUNS_DIR.rglob("best.pt"))
            if candidates:
                args.weights = str(sorted(candidates, key=os.path.getmtime)[-1])
                print(f"[i] Tự tìm weights: {args.weights}")
            else:
                print("[!] Cần chỉ định --weights path/to/best.pt")
                sys.exit(1)
        export_onnx(Path(args.weights), args.imgsz)
        return

    # ── Mode: train ────────────────────────────────────────────────────────
    try:
        import ultralytics
        print(f"[✓] ultralytics {ultralytics.__version__}")
    except ImportError:
        print("[✗] Chưa cài ultralytics!")
        print("    pip install ultralytics")
        sys.exit(1)

    train(args)


if __name__ == "__main__":
    main()
