# -*- coding: utf-8 -*-
"""
一键训练：比色管检测模型（YOLOv8）
==================================

用法（仓库根目录，用装好 ultralytics 的 Python 环境）:

    python 模型/train.py [--data 模型/data/data.yaml] [--model yolov8n.pt]
                         [--epochs 120] [--imgsz 640] [--batch 16] [--name tubes_v1]

产出: 模型/runs/train/<name>/weights/best.pt → 复制到 模型/weights/ 供 infer.py 使用

标注格式（YOLO 检测，单类 tube）:
    images/ 与 labels/ 同名同目录树，labels 每行:
        <类别=0> <x_center> <y_center> <w> <h>   （归一化 0-1）
    框住整根比色管（管口到管底可见部分），标注口径详见 data/标注规范.md。

⚠️ 沙箱网络坑（2026-10-13 实录）:
    - github.com 直连超时 → ultralytics 自动下载 yolov8n.pt 会失败；
      解决: 先从 https://hf-mirror.com/Ultralytics/YOLOv8/resolve/main/yolov8n.pt
      下载到本地，再把 --model 指向本地文件。
    - 系统 Python 受 PEP 668 管控不能直接 pip install；
      解决: python3 -m venv 建独立环境后安装。
    - torch 装 CPU 源即可（无 GPU 也能训/推）:
      pip install torch --index-url https://download.pytorch.org/whl/cpu
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser(description="训练比色管检测模型")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "data.yaml"))
    ap.add_argument("--model", default="yolov8n.pt",
                    help="预训练权重（github 不通时先手动从 hf-mirror 下载）")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="tubes_v1")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"数据配置不存在: {args.data}（先准备 dataset/ 图片与标注）")
    if not os.path.exists(args.model):
        sys.exit(
            f"预训练权重不存在: {args.model}\n"
            "github.com 在沙箱内直连超时，请先从 hf-mirror 下载:\n"
            "  curl -L -o yolov8n.pt https://hf-mirror.com/Ultralytics/YOLOv8/resolve/main/yolov8n.pt\n"
            "再重跑本脚本。"
        )

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=os.path.join(HERE, "runs", "train"),
        exist_ok=True,
        plots=True,
    )
    best = os.path.join(HERE, "runs", "train", args.name, "weights", "best.pt")
    print(f"\n训练完成，最优权重: {best}")
    print(f"复制到 weights/ 供推理使用: cp {best} {os.path.join(HERE, 'weights', args.name + '.pt')}")


if __name__ == "__main__":
    main()
