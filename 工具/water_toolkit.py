# -*- coding: utf-8 -*-
"""
水质检测照片处理工具包 water_toolkit.py
========================================

用法（在本资料包内，路径相对于 iwork 仓库根目录）:

    # 全自动（首选，无需人工框选，任何会 exec 的会话都能跑）:
    exec(open("工具/water_toolkit.py", encoding="utf-8").read())
    r, meta = auto_read_photo("照片存档/水样照片_20261013_00X.jpg")
    # meta 含每根管的检测方式(边缘检出/周期推算)/是否判透明/采样框，供人工复核

    # 手动框选（自动检测可疑时复核用）:
    r = read_photo("照片存档/水样照片_20261007_001.jpg",
                   {"pH": (x1,y1,x2,y2), "氨氮": (...), "亚盐": (...)})

流程: 【自动] 检测液柱(竖直边缘+等宽等距推算) → 白平衡校正 → 透明判定 →
      色调族采样 → 锚点换算 → ⚠️规则
依赖: PIL(Pillow) / numpy / matplotlib

----------------------------------------------------------------
⚠️ 透明判定（用户规则，2026-10-13 追加）
----------------------------------------------------------------
比色管液体与背景很相似（近无色、与背景融为一体）时，**看成透明**：
直接读低档——氨氮 0.20、亚盐 0.005（即"水质很好，检测不出来"），
不强行匹配锚点。判定：饱和度 < 0.22，或与背景色几乎一致(ΔS<0.05 且 ΔH<30°)。

----------------------------------------------------------------
⚠️ 比色卡（标准锚点来源）
----------------------------------------------------------------
测水读数以仓库根目录 `比色卡/` 下的 5 张单盘标准卡为权威锚点来源，
避免依赖任何外部文档（先前 数据/水质检测项目完整档案.txt 被误删，
本工具现把色阶/锚点/避坑/业务机理全部内嵌，单一事实来源，不再丢失）。

    COLOR_CARDS = {
        "pH低量程":   "比色卡/pH低量程.jpg",     # pH 6.50–7.60（黄→绿）
        "pH高量程":   "比色卡/pH高量程.jpg",     # pH 7.80–9.50（绿→蓝）
        "氨氮低量程": "比色卡/氨氮低量程.jpg",   # 氨氮 0.20–1.50（浅黄→橙）
        "氨氮高量程": "比色卡/氨氮高量程.jpg",   # 氨氮 2–15（黄→暗红）
        "亚硝酸盐":   "比色卡/亚硝酸盐.jpg",     # 亚盐 0.005–0.30（粉→品红）
    }

SCALES（印刷色阶，2026-10-13 从 5 张单盘特写逐扇区核读）:
    pH    : 低量程 6.50 / 6.80 / 7.00 / 7.20 / 7.40 / 7.60
            高量程 7.80 / 8.00 / 8.20 / 8.60 / 9.00 / 9.50
    氨氮  : 低量程 0.20 / 0.40 / 0.60 / 0.90 / 1.20 / 1.50
            高量程 2 / 3 / 5 / 8 / 15
    亚盐  : 0.005 / 0.05 / 0.10 / 0.15 / 0.20 / 0.25 / 0.30

ANCHORS 锚点由上述标准卡实测：每盘按扇区切分 → 与工具相同的白平衡校正
→ 取扇区中位色 (H,S,V)。亚盐各档色相几乎相同(≈0.90–0.93)，主要靠
饱和度+明度分档，故换算用全 HSV 加权最近邻，不再单靠色相。

----------------------------------------------------------------
⚠️ 避坑（历史训练沉淀，读数前必看）
----------------------------------------------------------------
1. 红管/粉管在蓝天水面背景前会被拍偏紫 → 先白平衡校正；亚盐像素色相落在
   0.80–0.88 属白平衡过冲偏紫，按明度定档，不按色相。
2. 俯拍（管立卡上）读数偏深：pH 偏高约 0.1–0.2，氨氮/亚盐饱和度偏高 →
   优先手持侧视读；俯拍传 view="top"，工具包自动备注。
3. 像素读数与人眼矛盾时，以人眼颜色为地面真值，并在备注写"人眼锚定"。
4. 近无色一律按低档读，不上探（氨氮→0.20，亚盐→0.005）。
5. 相邻档位色差细微时（如亚盐 0.25/0.30、氨氮 1.20/1.50），以人眼复核为准。

----------------------------------------------------------------
⚠️ 报警线（自动执行，不询问）
----------------------------------------------------------------
- 亚硝酸盐 ≥ 0.3，或接近（0.28）→ 标 ⚠️，备注"亚盐接近/超量程"
- 氨氮 ≥ 5（高量程）→ 标 ⚠️，备注"氨氮≥5⚠️"
- 亚盐超量程时给估值并注明"超量程估值，鱼易浮头"；其余一律不标
"""
from __future__ import annotations

import os

from PIL import Image
import re
import numpy as np
import matplotlib.colors as mcolors
# ============== 0. 比色卡注册表（标准锚点来源） ==============
COLOR_CARDS = {
    "pH低量程":   "比色卡/pH低量程.jpg",
    "pH高量程":   "比色卡/pH高量程.jpg",
    "氨氮低量程": "比色卡/氨氮低量程.jpg",
    "氨氮高量程": "比色卡/氨氮高量程.jpg",
    "亚硝酸盐":   "比色卡/亚硝酸盐.jpg",
}

# 印刷色阶（读数字面量）；与 COLOR_CARDS 标准卡一一对应
SCALES = {
    "pH":   {"低量程": [6.50, 6.80, 7.00, 7.20, 7.40, 7.60],
             "高量程": [7.80, 8.00, 8.20, 8.60, 9.00, 9.50]},
    "氨氮": {"低量程": [0.20, 0.40, 0.60, 0.90, 1.20, 1.50],
             "高量程": [2, 3, 5, 8, 15]},
    "亚盐": {"低量程": [0.005, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
             "高量程": [0.005, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]},
}


def list_color_cards(root="."):
    """返回当前可访问到的标准比色卡路径（相对 root）。"""
    out = {}
    for k, p in COLOR_CARDS.items():
        fp = os.path.join(root, p)
        out[k] = fp if os.path.exists(fp) else f"(缺失) {p}"
    return out


# ============== 1. 图片旋转 ==============
def rotate_tubes_down(fname, out=None, direction="cw"):
    """手持横拍图转正：默认顺时针90°使管口朝下；已是竖图则原样返回。
    返回 PIL.Image（竖图）。out 不为 None 时保存。"""
    img = Image.open(fname).convert("RGB")
    w, h = img.size
    if w > h:
        img = img.transpose(Image.ROTATE_270 if direction == "cw" else Image.ROTATE_90)
    if out:
        img.save(out, quality=95)
    return img


# ============== 2. 白平衡校正 ==============
def wb_correct(img, gray_world_fallback=True):
    """用画面内白色像素（水印白字/白卡纸/白桶）做白平衡校正；
    白色像素不足时退化为灰度世界法（整图平均灰做缩放）。
    返回 (校正后数组, 方法)。"""
    hsv = mcolors.rgb_to_hsv(img / 255.)
    white = (hsv[:, :, 1] < 0.15) & (hsv[:, :, 2] > 0.85)
    if white.sum() >= 500:
        mean_rgb = img[white].mean(axis=0)
        gray = mean_rgb.mean()
        return np.clip(img * (gray / mean_rgb), 0, 255), "white"
    if gray_world_fallback:
        mean_rgb = img.reshape(-1, 3).mean(axis=0)
        gray = mean_rgb.mean()
        return np.clip(img * (gray / mean_rgb), 0, 255), "grayworld"
    return img, "none"


# ============== 3. 液体采样 ==============
# 色调族（H 0-1）窗口：滤除背景/地板等异色像素
HF = {"pH": [(0.08, 0.62)], "氨氮": [(0.02, 0.22)], "亚盐": [(0.78, 0.99), (0.00, 0.05)]}

# 透明判定阈值：饱和度低于此值判"近无色(透明)"，按低档读（避坑#4）
S_TRANSPARENT = 0.22


def _base_param(p):
    return re.sub(r"[（(].*?[)）]", "", p).strip()


def bg_color(src):
    """估计背景色(HSV 0-1)：取画面四周边框5%区域的中位色。
    近无色液体与背景融为一体时用于'透明'判定的参照。"""
    if isinstance(src, str):
        img = np.array(Image.open(src).convert("RGB")).astype(float)
        hsv = mcolors.rgb_to_hsv(img / 255.)
    elif src.ndim == 3 and src.shape[2] == 3 and src.max() > 1.5:
        hsv = mcolors.rgb_to_hsv(src / 255.)          # RGB 数组
    else:
        hsv = src                                     # 已是 HSV
    hh, ww = hsv.shape[:2]
    border = np.concatenate([
        hsv[:max(1, int(hh * 0.05))].reshape(-1, 3), hsv[-max(1, int(hh * 0.05)):].reshape(-1, 3),
        hsv[:, :max(1, int(ww * 0.05))].reshape(-1, 3), hsv[:, -max(1, int(ww * 0.05)):].reshape(-1, 3)])
    return np.median(border, axis=0)


def sample_liquid(src, box, param, s_thr=0.20, bg=None):
    """WB校正后按色调族过滤，取液体中位色。
    src: 路径或图像数组; box: (x1,y1,x2,y2); param: pH/氨氮/亚盐
    bg: 背景色(HSV)，None 时自动估计；用于"与背景相近判透明"。
    返回 dict: rgb/hsv/n/wb/transparent；近无色或与背景相近返回 hsv=None"""
    if isinstance(src, str):
        img = np.array(Image.open(src).convert("RGB")).astype(float)
    else:
        img = src.astype(float)
    img, ok = wb_correct(img)
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = img[y1:y2, x1:x2].reshape(-1, 3)
    hsv = mcolors.rgb_to_hsv(crop / 255.)
    # ---- 透明判定（在色调族过滤之前）：近无色或与背景融为一体的液体直接判透明 ----
    if bg is None:
        bg = bg_color(img)
    med_hsv = np.median(hsv, axis=0)
    ds_bg = abs(float(med_hsv[1]) - float(bg[1]))
    dh_bg = min(abs(float(med_hsv[0]) - float(bg[0])), 1 - abs(float(med_hsv[0]) - float(bg[0]))) * 360
    if float(med_hsv[1]) < S_TRANSPARENT or (ds_bg < 0.05 and dh_bg < 30):
        return {"rgb": None, "hsv": None, "n": int(len(crop)), "wb": ok,
                "transparent": True, "note": "与背景相近/近无色,按透明处理"}
    m = (hsv[:, 1] > s_thr) & (hsv[:, 2] > 0.30)
    hue_ok = np.zeros(len(hsv), bool)
    for lo, hi in HF[param]:
        hue_ok |= (hsv[:, 0] >= lo) & (hsv[:, 0] <= hi)
    mask = m & hue_ok
    if mask.sum() < 60:
        mask = (hsv[:, 1] > 0.10) & hue_ok
    if mask.sum() < 30:
        return {"rgb": None, "hsv": None, "n": int(mask.sum()), "wb": ok,
                "transparent": True, "note": "近无色"}
    sel = crop[mask]
    q = (sel // 12).astype(int)          # RGB量化分箱(每通道~21档)
    keys, counts = np.unique(q, axis=0, return_counts=True)
    dom = sel[(q == keys[counts.argmax()]).all(axis=1)]
    med = np.median(dom, axis=0)          # 取面积最大档内中位色作代表色
    return {"rgb": med, "hsv": mcolors.rgb_to_hsv(med / 255.), "n": int(mask.sum()),
            "wb": ok, "transparent": False, "note": ""}


# ============== 4. 锚点换算表（侧视、WB校正后，2026-10-13 实测标定） ==============
# 格式: (hue, sat, val, 读数)  —— 锚点源自 比色卡/ 5张单盘标准卡实测
ANCHORS = {
    "pH": [
        (0.147, 0.995, 0.683, "6.50"), (0.174, 0.993, 0.511, "6.80"),
        (0.211, 1.000, 0.507, "7.00"), (0.288, 0.615, 0.534, "7.20"),
        (0.363, 0.666, 0.487, "7.40"), (0.406, 0.746, 0.475, "7.60"),
        (0.376, 0.714, 0.480, "7.80"), (0.402, 0.713, 0.421, "8.00"),
        (0.392, 0.575, 0.342, "8.20"), (0.456, 0.987, 0.330, "8.60"),
        (0.492, 0.471, 0.315, "9.00"), (0.570, 0.990, 0.429, "9.50"),
    ],
    "氨氮": [
        (0.156, 0.149, 0.822, "0.20"), (0.154, 0.462, 0.847, "0.40"),
        (0.141, 0.707, 0.884, "0.60"), (0.131, 0.781, 0.966, "0.90"),
        (0.125, 0.996, 0.888, "1.20"), (0.115, 0.996, 0.843, "1.50"),
        (0.103, 0.996, 0.925, "2"), (0.097, 0.995, 0.807, "3"),
        (0.074, 0.994, 0.673, "5"), (0.056, 0.983, 0.665, "8"),
        (0.028, 0.991, 0.426, "15"),
    ],
    "亚盐": [
        (0.919, 0.242, 0.874, "0.005"), (0.926, 0.628, 0.854, "0.05"),
        (0.918, 0.767, 0.817, "0.10"), (0.910, 0.894, 0.788, "0.15"),
        (0.900, 1.000, 0.710, "0.20"), (0.916, 1.000, 0.747, "0.25"),
        (0.927, 1.000, 0.735, "0.30"),
    ],
}
# 各指标最近邻距离权重 (dh, ds, dv)：按主判别通道加权
W_HV = {
    "pH":   (3.0, 1.2, 1.0),   # 色相为主（黄→绿→蓝）
    "氨氮": (1.0, 2.0, 1.2),   # 饱和度为主（浅黄→橙→暗红）
    "亚盐": (2.0, 1.8, 1.0),   # 饱和度为主，色相微漂
}
NO_COLOR = {"pH": None, "氨氮": "0.20", "亚盐": "0.005"}   # 近无色按低档


def hsv_to_value(param, hsv):
    """锚点最近邻换算（全 HSV 加权）。返回 (读数字符串, 备注)。"""
    base = _base_param(param)
    if hsv is None:
        v0 = NO_COLOR[base]
        if base == "氨氮" and "高量程" in param:
            v0 = "2"
        return v0, "近无色"
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])
    anchors = ANCHORS[base]
    note = ""
    if base == "亚盐" and 0.80 <= h <= 0.88:
        note = "偏紫(背景干扰),按明度定档"
    wh, ws, wv = W_HV[base]
    best, bd = None, 1e9
    for ah, a_s, av, val in anchors:
        dh = min(abs(h - ah), 1 - abs(h - ah))
        d = (wh * dh) ** 2 + (ws * (s - a_s)) ** 2 + (wv * (v - av)) ** 2
        if d < bd:
            bd, best = d, val
    # 亚盐超量程外推: ①比整卡都暗(v<0.670; 卡面最深档0.20的明度0.710,留0.040噪声带,
    #   防真·0.20样本因测量噪声误报); ②色相漂向纯红(h>0.945 或回绕至 h<0.80)。
    # 注意亚盐明度非单调(0.20档V0.710反而暗于0.25/0.30档),故暗样本最近邻会落到
    # 0.20/0.25,外推门必须包含这三档,否则真爆表样本被漏读为0.20且不标⚠️。
    # 偏紫带(0.80–0.88)色相贡献为0,只按明度外推——即避坑#1"按明度定档"。
    # 饱和度在深档已封顶(1.0)不携带区分信息,不参与幅度计算。
    if base == "亚盐" and best in ("0.20", "0.25", "0.30") and (v < 0.670 or h > 0.945 or h < 0.80):
        h_eff = h if h >= 0.5 else h + 1.0               # 色相绕回处理
        hue_x = min(1.0, max(0.0, (h_eff - 0.927) / 0.073))
        dark_x = min(1.0, max(0.0, (0.710 - v) / 0.20))  # 越过卡面最深档(0.20)后越暗越深
        x = 0.5 * hue_x + 0.5 * dark_x
        val = min(1.0, 0.30 + 0.70 * x)
        if val > 0.33:   # 幅度不足阈值的(如色相在带边、测量噪声)仍按原档读,不虚报超量程
            return f"{val:.2f}", (note + ";" if note else "") + f"超量程估值({val:.2f}),鱼易浮头"
    return best, note


# ============== 5. ⚠️ 规则 ==============
def need_warn(param, value_str):
    """亚盐>=0.3(或接近,如0.28)；氨氮>=5。"""
    try:
        v = float(value_str)
    except (TypeError, ValueError):
        return False
    p = _base_param(param)
    if p == "亚盐" and v >= 0.28:
        return True
    if p == "氨氮" and v >= 5:
        return True
    return False


# ============== 5.5 自动液柱检测（无需人工框选，低级会话也可直接跑） ==============
def _smooth(a, k):
    """一维移动平均（避免 scipy 依赖）。"""
    if k < 2:
        return a
    ker = np.ones(k) / k
    return np.convolve(a, ker, mode="same")


def detect_tubes(src, expect=3):
    """自动检测手持比色管：返回按 x 排序的管位列表。
    [{"cx": 中心x, "W": 管宽, "detected": 是否边缘检出(False=周期推算), "y1","y2": 液柱段}]

    原理：有色管(液体与背景差异大)的玻璃边缘在竖直边缘投影上呈强峰 → 峰对即管带；
    近无色管与背景融为一体、边缘弱，检不出——但同批管等宽、并排手持近似等距，
    用检出管的宽度 W 与间距 P（多管检出取中位；单管取 1.5W 先验）推算邻管位置。
    液柱段由检出管（S 显著高于背景的最长连续行段）确定，并排管液柱对齐故共用。
    """
    if isinstance(src, str):
        img = np.array(Image.open(src).convert("RGB")).astype(float)
    else:
        img = src.astype(float)
    img, _ = wb_correct(img)
    hsv = mcolors.rgb_to_hsv(img / 255.)
    hh, ww = hsv.shape[:2]
    bg = bg_color(hsv)
    bgS = float(bg[1])
    V = hsv[:, :, 2]

    # 竖直边缘 → 列投影（只算上部 75%，避开底部水印/桌面）
    Ex = np.abs(V[:, 2:] - V[:, :-2])
    Ex = np.pad(Ex, ((0, 0), (1, 1)))
    ytop = max(50, int(hh * 0.75))
    proj = _smooth(Ex[:ytop].sum(axis=0), 15)
    thr = proj.mean() + 2.0 * proj.std()
    peaks = [x for x in range(10, ww - 10)
             if proj[x] > thr and proj[x] >= proj[max(0, x - 8):x + 9].max()]
    merged = []
    for p in peaks:
        if merged and p - merged[-1] < 40:
            if proj[p] > proj[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)

    # 峰对 → 管带；验证（满足任一）：
    #  a. 带中心区域行饱和度高分位数显著高于背景（有色管的液体充满管心）
    #  b. 带管区内有横跨带的强水平线（液面/管底——玻璃管的结构特征）
    # 管旁折射暗带/背景间隙：中心是背景色、无横跨水平线，两条都不满足，借此排除。
    # 采样窗两端各缩 12px 避开贴邻管的折射暗边；高分位数而非全列中位——
    # 液柱只占画面高度一小部分，中位会被背景稀释；水平线检测限定在管区(y<ytop)，
    # 避开底部水印/桌面。
    gy_all = np.abs(V[2:ytop] - V[:ytop - 2]).mean(axis=1)   # 全图水平梯度投影(限管区)
    thr_line = gy_all.mean() + 3.0 * gy_all.std()
    colored_list = []
    bands = []
    for i in range(len(merged) - 1):
        a, b = merged[i], merged[i + 1]
        if not (60 < b - a < 800):
            continue
        cw = int(0.15 * (b - a))
        c = (a + b) // 2
        core = hsv[:, max(0, c - cw + 12):min(ww, c + cw - 12), 1]
        row_s = np.median(core, axis=1)
        colored = np.quantile(row_s, 0.75) > bgS + 0.15
        band_v = hsv[:ytop, a + 15:b - 15, 2]
        gy = np.abs(band_v[2:] - band_v[:-2]).mean(axis=1)
        has_line = False
        if float(gy.max()) > thr_line:
            yp = int(gy.argmax())
            line_row = np.abs(band_v[yp + 1] - band_v[yp - 1])
            has_line = float((line_row > 0.05).mean()) > 0.4   # 线须基本横跨带宽
        if colored or has_line:
            bands.append((a, b))
            colored_list.append(bool(colored))
    # 合并共享峰且两半都"有色"的相邻带：一根有色管被内部竖直结构峰
    # (液体纹路/管壁高光)劈成两半时还原为整管。靠水平线检出的近无色管
    # 不参与合并——那是独立的另一根管。
    consolidated, cons_flag = [], []
    for (a, b), cf in zip(bands, colored_list):
        if consolidated and consolidated[-1][1] == a and cons_flag[-1] and cf:
            consolidated[-1] = (consolidated[-1][0], b)
        else:
            consolidated.append((a, b))
            cons_flag.append(cf)
    bands = consolidated
    if not bands:
        return []
    centers = [float(a + b) / 2 for a, b in bands]
    W = int(np.median([b - a for a, b in bands]))
    if len(centers) >= 2:
        P = int(np.median(np.diff(sorted(centers))))
    else:
        P = int(1.5 * W)          # 同批管等宽并排，间隙≈半管宽

    # 液柱段：检出带内 S>bg+0.08 的最长连续行段，取最宽者共用（并排管液柱对齐）
    yband = None
    for a, b in bands:
        band = hsv[:, a + 20:b - 20, 1]
        hot = np.median(band, axis=1) > bgS + 0.08
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
        if best > 40 and (yband is None or best > yband[1] - yband[0]):
            yband = (bstart, bstart + best)
    if yband is None:
        yband = (int(hh * 0.30), int(hh * 0.60))     # 退化：图像中上部

    # 补齐到 expect 根（沿检出管左右按 P 推算；越界即止）
    tubes = list(centers)
    lo, hi = min(centers), max(centers)
    while len(tubes) < expect and lo - P > 0.06 * ww:
        lo -= P
        tubes.append(lo)
    while len(tubes) < expect and hi + P < 0.94 * ww:
        hi += P
        tubes.append(hi)
    tubes = sorted(tubes)[:expect]
    detected_set = set(int(c) for c in centers)
    return [{"cx": int(round(t)), "W": W, "y1": yband[0], "y2": yband[1],
             "detected": int(round(t)) in detected_set} for t in tubes]


def auto_read_photo(fname, view="side"):
    """全自动读数：检测液柱 → 采样 → 锚点换算。无需人工框选，脚本可直接执行。
    管序约定（按 x 排序）：左 pH、中氨氮、右亚盐。
    近无色/与背景融为一体的管按'透明'处理，直接读低档（氨氮0.20、亚盐0.005）。
    返回 (结果dict, 管位meta)。"""
    img = np.array(Image.open(fname).convert("RGB")).astype(float)
    img, wbok = wb_correct(img)
    bg = bg_color(img)
    tubes = detect_tubes(img)
    params = ["pH", "氨氮", "亚盐"]
    out, meta = {}, []
    for i, t in enumerate(tubes[:3]):
        p = params[i]
        cx, W, y1, y2 = t["cx"], t["W"], t["y1"], t["y2"]
        # 采样区：液柱段中部 60%（避开液面与管底反光），管心 ±0.3W（避开管壁）
        sy1 = int(y1 + (y2 - y1) * 0.2)
        sy2 = int(y1 + (y2 - y1) * 0.8)
        box = (max(0, cx - int(0.3 * W)), sy1, min(img.shape[1], cx + int(0.3 * W)), sy2)
        r = sample_liquid(img, box, p, bg=bg)
        val, note = hsv_to_value(p, r["hsv"])
        if r["transparent"]:
            note = (note + ";" if note else "") + r.get("note", "透明")
        if view == "top" and _base_param(p) == "pH" and val:
            note = (note + ";" if note else "") + "俯拍偏深约0.1-0.2"
        out[p] = {"value": val, "warn": need_warn(p, val),
                  "hsv": None if r["hsv"] is None else tuple(np.round(r["hsv"], 3)),
                  "n": r["n"], "note": note, "box": box}
        meta.append({"param": p, "cx": cx, "W": W, "detected": t["detected"],
                     "transparent": r["transparent"], "box": box})
    return out, meta


# ============== 6. 比色卡接入：从标准卡采样/重标定 ==============
def sample_card_swatch(card_path, boxes):
    """从一张标准比色卡采样各色块的代表 HSV。
    boxes: {读数值: (x1,y1,x2,y2)}（色块在卡图上的坐标）。
    返回 {读数值: (h,s,v, n)}，可直接用于核对/重建锚点。"""
    img = np.array(Image.open(card_path).convert("RGB")).astype(float)
    img, _ = wb_correct(img)
    out = {}
    for val, box in boxes.items():
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = img[y1:y2, x1:x2].reshape(-1, 3)
        hsv = mcolors.rgb_to_hsv(crop / 255.)
        m = (hsv[:, 1] > 0.10) & (hsv[:, 2] > 0.20)
        if m.sum() < 10:
            out[val] = (None, None, None, int(m.sum()))
            continue
        q = (crop[m] // 16).astype(int)
        keys, counts = np.unique(q, axis=0, return_counts=True)
        dom = crop[m][(q == keys[counts.argmax()]).all(axis=1)]
        med = np.median(dom, axis=0)
        h, s, v = mcolors.rgb_to_hsv(med / 255.)
        out[val] = (round(float(h), 3), round(float(s), 3), round(float(v), 3), int(m.sum()))
    return out


def build_anchors_from_card(param, card_path, boxes):
    """用标准卡上标定的色块坐标重建该指标的锚点表。
    boxes: {读数值: (x1,y1,x2,y2)}。返回 [(h,s,v,读数值), ...]，可赋值给 ANCHORS。"""
    sw = sample_card_swatch(card_path, boxes)
    return [(h, s, v, str(val)) for val, (h, s, v, _) in sw.items() if h is not None]


def verify_anchors(param, card_path, boxes):
    """比对当前 ANCHORS 与某张标准卡实测色块的色相差，输出漂移报告。
    返回 [(读数值, 当前锚点(h,s,v), 实测(h,s,v), 加权距离)]。"""
    sw = sample_card_swatch(card_path, boxes)
    base = _base_param(param)
    cur = {v: (h, s, vv) for h, s, vv, v in ANCHORS[base]}
    wh, ws, wv = W_HV[base]
    rep = []
    for val, (h, s, v, _) in sw.items():
        if h is None or str(val) not in cur:
            continue
        ah, a_s, av = cur[str(val)]
        dh = min(abs(h - ah), 1 - abs(h - ah))
        d = (wh * dh) ** 2 + (ws * (s - a_s)) ** 2 + (wv * (v - av)) ** 2
        rep.append((val, (round(ah, 3), round(a_s, 3), round(av, 3)),
                    (h, s, v), round(float(np.sqrt(d)), 3)))
    return rep


# ============== 7. 一键读图 ==============
def read_photo(fname, tubes, view="side"):
    """tubes: {param: box}；view: side=手持侧视 / top=俯拍(色偏深)
    返回 {param: {"value","warn","hsv","n","note"}}"""
    bg = bg_color(fname)
    out = {}
    for param, box in tubes.items():
        r = sample_liquid(fname, box, _base_param(param), bg=bg)
        val, note = hsv_to_value(param, r["hsv"])
        if r.get("transparent"):
            note = (note + ";" if note else "") + r.get("note", "透明")
        if view == "top" and _base_param(param) == "pH" and val:
            note = (note + ";" if note else "") + "俯拍偏深约0.1-0.2"
        out[param] = {"value": val, "warn": need_warn(param, val),
                      "hsv": None if r["hsv"] is None else tuple(np.round(r["hsv"], 3)),
                      "n": r["n"], "note": note}
    return out


# ============== 8. 自检 ==============
def self_test():
    print("比色卡注册表:", list_color_cards())
    print("色阶 pH 低量程:", SCALES["pH"]["低量程"])
    print("色阶 pH 高量程:", SCALES["pH"]["高量程"])
    print("色阶 氨氮低/高:", SCALES["氨氮"]["低量程"], SCALES["氨氮"]["高量程"])
    print("色阶 亚盐:", SCALES["亚盐"]["低量程"])
    print("透明判定阈值 S<%.2f（近无色/与背景相近 → 读低档）" % S_TRANSPARENT)
    print("OK: 锚点表/⚠️规则/自动检测(auto_read_photo)/透明判定已加载；")
    print("    全自动读图: r, meta = auto_read_photo(照片路径)")


if __name__ == "__main__":
    self_test()
