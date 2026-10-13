# -*- coding: utf-8 -*-
"""
模型推理：比色管照片 → 水质读数初判（YOLOv8 检测 + water_toolkit 业务规则复用）
==========================================================================

用法（在 iwork 仓库根目录执行）:

    python 模型/infer.py 照片路径 [--weights 模型/weights/tubes_v1.pt]
                           [--view side|top] [--show 输出预览图路径]

流程（两段式管线的推理侧）:
    1. YOLO 检测比色管（单类 tube）→ 得到每根管的紧致边框，按 x 排序；
    2. 每管：白平衡校正 → 框内定位液柱段（饱和度显著高于背景的最长连续行段）
       → 色调族过滤取液柱代表色（框紧致 + 过滤双保险，避开蓝天/水面背景污染）；
    3. 指标判定：液柱代表色与 pH/氨氮/亚盐 三族锚点算加权距离，取最近族定指标
       —— 不依赖"左pH中氨氮右亚盐"的管序约定，管序不同也不会张冠李戴；
    4. 读数：复用 `工具/water_toolkit.py` 的 hsv_to_value（锚点最近邻 + 亚盐超量程外推）
       与 need_warn（⚠️ 规则），保持与现有业务口径完全一致。

⚠️ 红线（与 工作流/01 第八节一致）:
    本脚本输出为"辅助初判"，外拍照上不得直接作为读数汇报。
    与人眼/地面真值矛盾时以人眼为准，并在记录中备注"人眼锚定"。

依赖: ultralytics + torch(CPU 可跑) + PIL/numpy/matplotlib
      环境安装与权重下载的坑（沙箱网络限制）见 模型/README.md。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.colors as mcolors
from PIL import Image

# ---------- 复用仓库既有业务资产（单一事实来源，不重造轮子） ----------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # 仓库根
TOOLKIT = os.path.join(ROOT, "工具", "water_toolkit.py")

if not os.path.exists(TOOLKIT):
    sys.exit(f"找不到 {TOOLKIT}，请在仓库根目录下运行")

_tk = {}
exec(open(TOOLKIT, encoding="utf-8").read(), _tk)
wb_correct = _tk["wb_correct"]
bg_color = _tk["bg_color"]
hsv_to_value = _tk["hsv_to_value"]
need_warn = _tk["need_warn"]
ANCHORS = _tk["ANCHORS"]
W_HV = _tk["W_HV"]
HF = _tk["HF"]
BASE_PARAMS = ["pH", "氨氮", "亚盐"]

DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "tubes_v1.pt")
# 微调权重不存在时退回预训练权重（COCO 的 cup/bottle 类比色管有基础响应，
# 2026-10-13 实测 rot_1 三根全中，可作无权重时的降级方案）
FALLBACK_WEIGHTS = os.path.join(HERE, "weights", "yolov8n.pt")


def _weighted_dist(hsv, param):
    """代表色与某指标全部锚点的最小加权距离（复用 water_toolkit 的 W_HV）。"""
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])
    wh, ws, wv = W_HV[param]
    best = 1e9
    for ah, a_s, av, _ in ANCHORS[param]:
        dh = min(abs(h - ah), 1 - abs(h - ah))
        d = (wh * dh) ** 2 + (ws * (s - a_s)) ** 2 + (wv * (v - av)) ** 2
        best = min(best, float(np.sqrt(d)))
    return best


def decide_param(hsv):
    """液柱代表色 → 最接近的指标（pH/氨氮/亚盐）。不依赖管序。"""
    if hsv is None:
        return None
    dists = {p: _weighted_dist(hsv, p) for p in BASE_PARAMS}
    return min(dists, key=dists.get)


def liquid_segment_and_color(img, box, param):
    """在检测框内定位液柱段并取代表色。

    img: WB 校正后的 RGB 数组; box: (x1,y1,x2,y2)
    返回 (hsv 或 None, 备注)。近无色返回 (None, "近无色")。
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = img[y1:y2, x1:x2].reshape(-1, 3)
    if crop.size == 0:
        return None, "空框"
    hsv = mcolors.rgb_to_hsv(crop / 255.)
    H, W = hsv.shape[0], 1  # crop 已拉平
    hsv2d = hsv.reshape(y2 - y1, x2 - x1, 3)
    # 液柱段：逐行饱和度中位数 > 0.20 的最长连续行段（避开管口空气段与管底）
    row_s = np.median(hsv2d[:, :, 1], axis=1)
    hot = row_s > 0.20
    best, cur, start, bstart = 0, 0, 0, 0
    for y, v in enumerate(hot):
        if v:
            if cur == 0:
                start = y
            cur += 1
            if cur > best:
                best, bstart = cur, start
        else:
            cur = 0
    if best < 10:
        return None, "框内无液柱段(全框近无色)"
    seg = hsv2d[bstart:bstart + best].reshape(-1, 3)
    # 色调族过滤后取主色档（与 water_toolkit.sample_liquid 同口径）
    mask = seg[:, 1] > 0.10
    if mask.sum() < 30:
        return None, "近无色"
    sel = seg[mask]
    q = (sel // 12).astype(int)
    keys, counts = np.unique(q, axis=0, return_counts=True)
    dom = sel[(q == keys[counts.argmax()]).all(axis=1)]
    med = np.median(dom, axis=0)
    return mcolors.rgb_to_hsv(med / 255.), ""


def infer_photo(fname, weights, view="side"):
    """单张照片 → {指标: {"value","warn","hsv","note","box"}} + meta 列表。"""
    from ultralytics import YOLO

    w = weights
    if not os.path.exists(w):
        w = FALLBACK_WEIGHTS
        print(f"[提示] 权重 {weights} 不存在，退回预训练权重 {w}")
    model = YOLO(w)

    img = np.array(Image.open(fname).convert("RGB")).astype(float)
    img_wb, wbok = wb_correct(img)

    r = model.predict(fname, conf=0.25, verbose=False)[0]
    out, meta = {}, []
    if r.boxes is None or len(r.boxes) == 0:
        return out, meta

    boxes = sorted([tuple(int(v) for v in b.xyxy[0]) for b in r.boxes], key=lambda b: b[0])
    for box in boxes:
        hsv, note = liquid_segment_and_color(img_wb, box, None)
        if hsv is None:
            meta.append({"box": box, "param": None, "note": note})
            continue
        param = decide_param(hsv)
        if param is None:
            meta.append({"box": box, "param": None, "note": "指标判定失败"})
            continue
        val, vnote = hsv_to_value(param, hsv)
        warn = need_warn(param, val)
        if view == "top" and param == "pH":
            vnote = (vnote + ";" if vnote else "") + "俯拍偏深约0.1-0.2"
        out[param] = {"value": val, "warn": warn,
                      "hsv": tuple(np.round(hsv, 3)), "note": vnote, "box": box}
        meta.append({"box": box, "param": param, "note": vnote or note})
    return out, meta


def draw_preview(fname, out, meta, save_to):
    """画检测框+读数预览图，供人工复核。"""
    img = Image.open(fname).convert("RGB")
    from PIL import ImageDraw
    dr = ImageDraw.Draw(img)
    for m in meta:
        x1, y1, x2, y2 = m["box"]
        dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
        label = m.get("param") or m.get("note", "")
        dr.text((x1 + 4, max(0, y1 - 18)), str(label), fill=(255, 0, 0))
    for p, d in out.items():
        print(f"  {p}: {d['value']} warn={d['warn']} hsv={d['hsv']} note={d['note']} box={d['box']}")
    img.save(save_to)
    print(f"预览图已存: {save_to}")


def main():
    ap = argparse.ArgumentParser(description="比色管照片 → 读数初判（YOLO+water_toolkit）")
    ap.add_argument("photo")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--view", default="side", choices=["side", "top"])
    ap.add_argument("--show", default=None, help="保存标注预览图路径")
    args = ap.parse_args()

    out, meta = infer_photo(args.photo, args.weights, args.view)
    print(f"\n=== {args.photo} ===")
    if not out:
        print("未检出比色管或无有效液柱")
        for m in meta:
            print("  meta:", m)
        return
    for p in ["pH", "氨氮", "亚盐"]:
        if p in out:
            d = out[p]
            flag = " ⚠️" if d["warn"] else ""
            print(f"  {p}: {d['value']}{flag}  hsv={d['hsv']}  note={d['note']}  box={d['box']}")
    if args.show:
        draw_preview(args.photo, out, meta, args.show)


if __name__ == "__main__":
    main()
