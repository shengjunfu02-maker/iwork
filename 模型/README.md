# 模型（比色管视觉识别 · YOLOv8 两段式）

> 2026-10-13 建。起因：自研 `工具/water_toolkit.py` 的 `detect_tubes`（手工边缘投影检管）
> 在塘头外拍照片上**系统性失灵**（把水面/水印当管、漏检近无色管），采样框还会混入
> 过半天空像素导致误读（2号塘 pH 误读 9.00 实为 7.4）。用户定性"纯程序判断根本读不准"，
> 决定引入成熟视觉模型 YOLOv8 解决"**管在哪、哪些像素是液柱**"。

## 一、体系定位（重要边界）

YOLO **不直接吐读数**。本目录是两段式管线的第一段：

```
照片 ──► ① YOLOv8 检测 tube ──► 每根管的紧致边框（外拍鲁棒，替代 detect_tubes）
                │
                ▼
        ② infer.py 框内定位液柱段 + 取代表色（复用 water_toolkit 的白平衡/色调族口径）
                │
                ▼
        ③ 按代表色与三族锚点距离定指标（不依赖管序）→ hsv_to_value 换算 + need_warn ⚠️
                │
                ▼
        读数初判（辅助）──► 人眼/地面真值仲裁（红线，见 工作流/01 避坑#6）
```

- 第二段"色档定档"的升级路线（可选）：用 `比色卡/` 5 张标准卡程序化增广生成海量
  训练对，训一个 30 档色档分类器替代锚点最近邻——标准卡是权威源，数据零成本。
- `工具/water_toolkit.py` 的色阶表/⚠️规则/亚盐外推/避坑等业务资产**全部保留复用**，
  本体系只替换"检管+采样"两个失灵的环节。

## 二、目录

```
模型/
├── README.md          ← 本文件
├── infer.py           ← 推理：照片 → 读数初判（YOLO 检测 + water_toolkit 规则复用）
├── train.py           ← 一键训练
├── requirements.txt   ← 依赖
├── data/
│   ├── data.yaml      ← 数据集配置（单类 tube）
│   └── 标注规范.md     ← 标注口径 + 覆盖矩阵 + 质检清单
├── dataset/           ← images/ + labels/（YOLO 格式），照片与标注入库处
└── weights/           ← 训练产出权重（best.pt 复制到此，6MB 级可直接 git 管理）
```

## 三、环境安装（沙箱实录，2026-10-13）

沙箱网络：`api.github.com` 通，**`github.com` 直连超时**（ultralytics 自动下载权重必失败）。

```bash
# 1. venv（系统 Python 受 PEP 668 管控，不能直接 pip install）
python3 -m venv /tmp/yoloenv
/tmp/yoloenv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/yoloenv/bin/pip install ultralytics

# 2. 预训练权重手动下载（hf-mirror 源可达；jsdelivr 对该仓库 404；ghproxy 不通）
curl -L -o yolov8n.pt https://hf-mirror.com/Ultralytics/YOLOv8/resolve/main/yolov8n.pt
# 训练: python 模型/train.py --model yolov8n.pt
# 推理: python 模型/infer.py 照片 --weights 模型/weights/yolov8n.pt
```

## 四、试跑证据（未微调预训练权重，2026-10-13）

| 照片 | 检出 | 说明 |
|---|---|---|
| rot_1（丁力强4号塘） | **3/3** | conf 0.90/0.77/0.74，管框与人工真值高度吻合 |
| rot_0（5号塘） | 2/3 | 漏绿草地前的绿色 pH 管 |
| rot_2 / rot_3 / 8月士宅村 | 1/3 | conf 0.29–0.41，近无色管检不出 |

→ 学习门槛不高，**50–100 张覆盖到位的标注照即可产出可用 v1**；
漏检集中在"绿背景前的绿管"与"近无色管"，标注时按 `data/标注规范.md` 的覆盖矩阵加权。

## 五、当前状态与下一步

- [x] 环境验证 + 预训练试跑（可行性成立）
- [x] 目录/脚本/标注规范落库
- [ ] **待供片**：用户批量导历史塘头照（今天 4 塘、8月冯坡 3 张、更早照片）入 `dataset/images/`
- [ ] 标注（模型目视生成 YOLO txt）+ train/val 划分
- [ ] `python 模型/train.py` 训练 → 权重入 `weights/`
- [ ] infer.py 回归今日 4 塘照片，与人眼读数逐项比对；通过后更新 `工作流/01`

## 六、红线（不变）

1. 外拍照上模型输出为**辅助初判**，矛盾时以人眼/地面真值为准，备注"人眼锚定"；
2. 不是比色管的图不硬读（工作流01 第 0 步门禁优先）；
3. 改推理行为须同步 `工作流/01` 第八节与 `工具/water_toolkit.py` docstring（互为指针）。
