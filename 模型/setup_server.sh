#!/usr/bin/env bash
# =============================================================================
# 比色管 YOLOv8 检测 —— 服务器一键环境搭建 + 数据自检
# =============================================================================
# 用法（在 iwork 仓库根目录）:
#     bash 模型/setup_server.sh              # 建 venv + 装依赖 + 校验数据
#     bash 模型/setup_server.sh --cpu-torch  # 强制装 torch CPU 轮子（无 GPU 机器）
#     bash 模型/setup_server.sh --skip-deps  # 只校验数据，不装依赖
#
# 幂等：重复执行安全（已存在的 venv 会复用）。
# 兼容：Ubuntu/Debian/macOS；中文目录安全（全程引号）。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # 模型/
ROOT="$(dirname "$HERE")"                                 # 仓库根
VENV="$HERE/.venv"
PY="${PYTHON:-python3}"

CPU_TORCH=0
SKIP_DEPS=0
for arg in "$@"; do
  case "$arg" in
    --cpu-torch) CPU_TORCH=1 ;;
    --skip-deps) SKIP_DEPS=1 ;;
    *) echo "未知参数: $arg"; exit 2 ;;
  esac
done

echo "=========================================================="
echo " 比色管 YOLO 环境搭建"
echo " 仓库根: $ROOT"
echo " 虚拟环境: $VENV"
echo "=========================================================="

# ---------- 1. Python 检查 ----------
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[错误] 找不到 $PY，请先安装 Python 3.9+"
  exit 1
fi
echo "[1/5] Python: $("$PY" --version)"

# ---------- 2. venv ----------
if [ ! -x "$VENV/bin/python" ]; then
  echo "[2/5] 创建虚拟环境（避开 PEP 668 系统 Python 限制）..."
  "$PY" -m venv "$VENV"
else
  echo "[2/5] 复用已存在的虚拟环境"
fi
PYBIN="$VENV/bin/python"
PIPBIN="$VENV/bin/pip"

# ---------- 3. 依赖 ----------
if [ "$SKIP_DEPS" -eq 0 ]; then
  echo "[3/5] 安装依赖（ultralytics / torch / Pillow / numpy / matplotlib）..."
  "$PIPBIN" install --quiet --upgrade pip
  if [ "$CPU_TORCH" -eq 1 ]; then
    echo "      → 使用 torch CPU 轮子源"
    "$PIPBIN" install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  fi
  "$PIPBIN" install --quiet -r "$HERE/requirements.txt"
else
  echo "[3/5] 跳过依赖安装（--skip-deps）"
fi

# ---------- 4. 环境自检 ----------
echo "[4/5] 环境自检..."
"$PYBIN" - <<'PYEOF'
import sys
try:
    import ultralytics, torch
except Exception as e:
    print(f"[错误] 依赖导入失败: {e}")
    sys.exit(1)
cuda = torch.cuda.is_available()
print(f"      ultralytics {ultralytics.__version__} | torch {torch.__version__} | CUDA {cuda}")
print(f"      训练设备建议: {'--device 0 (GPU)' if cuda else '--device cpu (无 GPU)'}")
PYEOF

# 预训练权重检查（离线可用）
WT="$HERE/weights/yolov8n.pt"
if [ -f "$WT" ]; then
  echo "      预训练权重就绪: weights/yolov8n.pt ($(du -h "$WT" | cut -f1))"
else
  echo "      [警告] 缺 weights/yolov8n.pt，训练时需联网下载或手动放入"
fi

# ---------- 5. 数据集自检 ----------
echo "[5/5] 数据集自检（模型/dataset）..."
"$PYBIN" - <<PYEOF
import os, glob
ds = r"$HERE/dataset"
ok = True
for split in ("train", "val"):
    imgs = sorted(glob.glob(os.path.join(ds, "images", split, "*.jpg")))
    labs = sorted(glob.glob(os.path.join(ds, "labels", split, "*.txt")))
    names_i = {os.path.splitext(os.path.basename(p))[0] for p in imgs}
    names_l = {os.path.splitext(os.path.basename(p))[0] for p in labs}
    miss_lab = names_i - names_l
    miss_img = names_l - names_i
    nbox = 0
    bad = []
    for p in labs:
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 5:
                bad.append(f"{os.path.basename(p)}: 字段数={len(parts)}")
                continue
            try:
                c, x, y, w, h = parts
                c = int(c); vals = list(map(float, (x, y, w, h)))
            except Exception:
                bad.append(f"{os.path.basename(p)}: 非数字")
                continue
            if c != 0:
                bad.append(f"{os.path.basename(p)}: 类别={c}(应为0)")
            if not all(0.0 <= v <= 1.0 for v in vals):
                bad.append(f"{os.path.basename(p)}: 坐标越界")
            nbox += 1
    print(f"  [{split}] 图片 {len(imgs)} 张 | 标签 {len(labs)} 个 | 框 {nbox} 个")
    if miss_lab:
        print(f"    !! 缺标签: {sorted(miss_lab)[:5]}"); ok = False
    if miss_img:
        print(f"    !! 缺图片: {sorted(miss_img)[:5]}"); ok = False
    if bad:
        print(f"    !! 标签格式问题({len(bad)}): {bad[:5]}"); ok = False
if ok:
    print("  数据自检通过 ✅")
else:
    print("  数据自检发现问题 ⚠️（新补的图请按 模型/data/标注规范.md 修正后重跑）")
PYEOF

echo ""
echo "=========================================================="
echo " 完成。下一步："
echo "   source 模型/.venv/bin/activate"
echo "   python 模型/train.py --epochs 120 --batch 8 --device cpu"
echo "   cp 模型/runs/train/tubes_v1/weights/best.pt 模型/weights/tubes_v1.pt"
echo "   python 模型/infer.py 照片.jpg --weights 模型/weights/tubes_v1.pt --show /tmp/p.jpg"
echo " 详见 模型/服务器部署训练手册.md"
echo "=========================================================="
