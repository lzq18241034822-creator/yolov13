# µSHM‑YOLO 统一标注 GUI（PyQt5）

这是按 first.md 要求实现的桌面版标注软件（PyQt5），支持：
- 多边形/矩形绘制；自动生成 bbox（显示用）；保存统一 TXT：5分类 + polygon点序列
- 从 YAML 自动加载类目与中文名；cell_organization 支持新 44“多(>4)”并兼容显示旧 10/11
- 自动注入文件头（放大倍数、像素大小、图像尺寸等元数据）与前缀命名模板
- 加载已有标注并叠加显示；撤销一点/完成多边形/清空/删除选中

## 环境

- 建议 Python 3.10/3.11（Windows）
- 环境名：`µSHM-YOLO`（若无法输入 `µ`，用 `uSHM-YOLO`）

### Conda

```powershell
conda create -n µSHM-YOLO python=3.11 -y
conda activate µSHM-YOLO
cd g:\yoloV13\µSHM-YOLO
pip install -r requirements.txt
```

### venv（可选）

```powershell
cd g:\yoloV13\µSHM-YOLO
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 启动（桌面 GUI）

简洁启动（推荐）：

```powershell
cd g:\yoloV13\µSHM-YOLO\tools
python unified_annotator_gui.py
```

打开软件后，在左侧点击“导入图片目录”和“选择输出目录”完成路径设置，再开始标注。配置文件默认读取 `g:\yoloV13\µSHM-YOLO\yolov13_transformer_unified_v2_1.yaml`。

可选：命令行传参（一次性指定目录）

```powershell
cd g:\yoloV13\µSHM-YOLO
python tools\unified_annotator_gui.py --image_dir E:\datasets\microalgae\images\train --output_dir E:\datasets\microalgae\labels\train --cfg g:\yoloV13\µSHM-YOLO\yolov13_transformer_unified_v2_1.yaml
```

## 操作流程

- 模式：按钮切换“多边形模式”或“矩形模式”
- 添加点：左键点击（多边形）；矩形框：拖拽起止点
- 默认分类：右侧下拉选择 5 个分类作为“新标注默认值”
- 完成多边形：点击“完成多边形”将当前点列入列表
- 撤销一点/清空全部/删除选中：对应按钮操作
- 导入图片：点击“导入图片目录”，选择新的 `images/train` 目录，自动加载图像列表
  - 支持常见格式：`jpg/jpeg/jpe/jfif/png/bmp/dib/tif/tiff/webp/jp2/ppm/pgm/pbm/pnm/gif/ico/ras`
  - 若 OpenCV 无法读取，则使用 Pillow 回退；部分特殊格式（如 HEIC/AVIF/RAW）需安装额外插件
- 选择输出：点击“选择输出目录”，设置 `labels/train` 保存路径
- 自动导出：点击“自动导出TXT”，即时将当前图片的标注写入统一 TXT（与“保存到 labels”等效）
- 上/下一张：切换图像；加载已有标注时自动叠加显示
- 保存：写入 `--output_dir` 中，文件名与原图同名（仅扩展名为 `.txt`，例：`img001.txt`），并根据实际图像尺寸注入头部元数据

## 标注 TXT 格式

- 头部（受 `dataset.naming.auto_inject_header` 控制）：
  - `# magnification: 400x`
  - `# pixel_size_um: 0.0863`
  - `# pixel_dimensions_px: WxH`（按实际图像）
  - `# magnification_camera: 40x`
  - `# pixel_size_um_source: ...`
- 每行：
  - `species cell_org shape flagella chloroplast x1 y1 ... xn yn`
  - 坐标均归一化到 `[0,1]`；矩形以四角点输出 polygon；多边形按绘制顺序输出

## 与 YOLOv13 源仓的关系

- 本工具完全独立运行；你后续删除 `g:\yoloV13\yolov13-main` 不受影响
- 训练期兼容映射（旧 10/11 → 多 44）由 YAML 的 `classes.label_spaces.cell_org` 负责，标注端允许显示旧类不强制改写

### 主程序与第三方源的关系（答复你的疑问）
- 主程序位置是 `g:\yoloV13\µSHM-YOLO`，这里包含“可直接运行”的所有脚本、模型与数据管道；这是我们主要维护和使用的入口。
- 顶层还有一个 `g:\yoloV13\yolov13-main`（官方/第三方 YOLOv13 源）。它是“参考/研发用”的独立代码库，只有在该目录下运行它自己的脚本时才会被使用；目前主程序不直接依赖它。
- 若需要调用 Ultralytics 官方训练引擎进行对比验证，请在 `yolov13-main` 目录内或通过专门脚本（例如 `tools/train_ultra_stage1.py`）运行；主程序的最小闭环不要求引入或改动 `yolov13-main`。

### 模型目录详解（`µSHM-YOLO/models/`）
- `uscale_film.py`：像素尺度注入模块（µScale）。包含 `PixelScaleNormalizer` 与 `FiLM2d`，用于在网络特征上执行条件调制（支持方法如 log 比值）。
- `simple_seg_film.py`：Stage1 简易分割网络，已接入 FiLM 注入并允许 DCNv3 的占位回退；前向 `forward(images, pixel_scale=...)` 会透传像素尺度。
- `multitask_heads.py`：Stage2 多头分类器（教师/学生）结构与头部组合。
- 提示：如果你后续需要把 FiLM/DCNv3 精确移植到 YOLOv13 的 `RepGFPN` 颈部与分割头（proto+coeff），可以补入相应模块；当前主程序已用最小网络完成训练闭环与管线联通。

## 常见问题

- `µ` 编码问题：改用 `uSHM-YOLO` 环境名即可
- GUI 未显示：请确认本地显示环境正常（远程/无显示会失败）
- 大图显示缩放：绘制坐标按原图像素保存，缩放只影响显示，不影响数据正确性

## 项目总览与训练/推理流程

本项目包含从统一标注到三阶段训练与推理的完整闭环：

- Stage1（分割）：SimpleSegNetFiLM 基于统一多边形数据训练，导出 ROI 及可视化报告。
- Stage2（多头分类）：
  - 教师模型训练（MultiHeadClassifier）；
  - 导出教师 `logits.npz`；
  - 蒸馏学生模型（Mask2Morph Distill，沿用 MultiHeadClassifier 结构）。
- Stage3（推理管线）：分割→ROI分类→软规则检查→物理参数估算→叠加可视化与 JSON 输出。

## 目录结构与职责说明（已对齐与清理）

为提高可读性与一致性，顶层与子目录已对齐；所有入口统一在 `µSHM-YOLO` 下。以下为主要目录及用途：

### 顶层目录（g:\yoloV13）
- `µSHM-YOLO/`：主程序与脚本入口。包含标注 GUI、训练/推理/评测脚本、模型与数据集实现、工具与可视化等。
- `yolov13-main/`：第三方 YOLOv13 源代码（参考用）。本项目最小闭环不依赖此目录；需要研究或比较时可保留。
- `runs/`：训练与推理输出目录（权重、日志、叠加图、报告、NPZ 等）。
- `datasets/`：已废弃的旧数据集脚本位置，现已重命名为 `.deprecated.py`（权威版本在 `µSHM-YOLO/datasets/`）。
- `tools/`：已废弃的旧预览脚本位置，现已重命名为 `.deprecated.py`（统一使用 `µSHM-YOLO/tools/`）。
- `utils/`：已废弃的旧工具函数位置，现已重命名为 `.deprecated.py`（统一使用 `µSHM-YOLO/utils/`）。
- `ultralytics/`：旧的占位副本（不完整、未使用）。真实 Ultralytics 库位于 `yolov13-main/ultralytics/`。如无需要，建议移除或忽略。
- `stage3/`：顶层空目录（未使用）。所有 Stage3 代码在 `µSHM-YOLO/stage3/`。

重复文件已标记为“不可用的历史版本”（避免误用）：
- `tools/preview_unified_batch.deprecated.py`
- `utils/aug_poly.deprecated.py`
- `datasets/unified_seg_dataset.deprecated.py`

### 子目录结构（µSHM-YOLO）
- `AIplan/`：开发迭代计划与设计文档（`0–11.md`）。
- `models/`：核心模型实现。
  - `simple_seg_film.py`：Stage1 简易分割网络（FiLM 注入 `pixel_scale`，支持 DCNv3 回退）。
  - `uscale_film.py`：`PixelScaleNormalizer` 与 `FiLM2d` 注入模块。
  - `multitask_heads.py`：Stage2 多头分类器实现。
- `datasets/`：统一数据集与 ROI 数据集。
  - `unified_seg_dataset.py`：统一分割数据集，读取标注 TXT（含头部 `pixel_size_um`），提供 `collate_fn_unified`。
  - `roi_multitask_dataset.py / roi_with_teacher_logits.py`：ROI 训练与蒸馏用数据集。
- `tools/`：训练/推理/评测/导出与可视化脚本。
  - `unified_annotator_gui.py`：统一标注 GUI。
  - `train_unified_stage1.py`：Stage1 训练（导出 ROI 与预览）。
  - `train_stage2_teacher.py`：教师模型训练；`export_teacher_logits.py`：导出教师 logits。
  - `train_mask2morph_distill.py`：蒸馏学生训练；`export_student_logits.py`：导出学生 logits。
  - `infer_stage3_pipeline.py`：Stage3 一键推理管线（分割→分类→规则→参数→叠加图与 JSON）。
  - `eval_stage3_report.py`：Stage3 评测与报告（混淆矩阵与统计）。
  - `refine_hypergraph_stage3.py`：超图（邻近聚团）离线细化。
  - `preview_unified_batch.py`：数据加载与几何一致增强的快速预览（输出到 `tools/reports/`）。
  - `train_yolov13_stage1.py`：YOLOv13 风格的最小分割训练脚本（使用统一多边形数据与 FiLM 注入，含 DCNv3 开关占位与自动回退）。
- `utils/`：工具函数与可视化。
  - `aug_poly.py`：分割多边形安全增广（letterbox/随机仿射/翻转等）。
  - `losses_seg.py / mask_ops.py / matching.py`：训练与掩膜相关操作。
  - `metrics_cls.py`：分类指标；`vis_overlay.py`：叠加绘制（中文字体回退）。
  - `early_stop.py / distill.py`：早停与蒸馏过程辅助。
- `stage3/`：规则引擎与参数估算。
  - `rule_engine.py`：软规则检查与降级提示；`param_calc.py`：10 项物理参数估算。
  - `hypergraph_refine.py`：邻近聚团细化算法。
- `samples/`：样例数据根（images/labels）；`splits/`：固定划分列表。
- `yolov13_transformer_unified_v2_1.yaml`：统一配置（类目、显微参数、训练开关、FiLM/DCNv3 等）。

### 对齐与使用建议
- 统一入口：请在 `µSHM-YOLO/tools` 下运行所有训练/推理/评测脚本，减少 `PYTHONPATH` 混淆。
- 第三方源：`yolov13-main` 仅作参考/研发，不影响本项目最小闭环；如需 Ultralytics 引擎测试，请在该目录内独立运行。
- 顶层冗余：`ultralytics/` 与 `stage3/` 顶层目录不再使用；如需精简可手动删除。
- 历史脚本：顶层 `tools/utils/datasets` 的旧脚本已重命名为 `.deprecated.py`，如需对照请仅阅读，不要导入运行。

## 脚本一览与职责（详细版）
- `tools/unified_annotator_gui.py`：统一标注 GUI（PyQt5）。绘制多边形/矩形，保存统一 TXT（含头部元数据）。
- `tools/validate_unified_dataset.py`：统一格式数据质检与统计，输出报告到 `tools/reports/`。
- `tools/make_fixed_split.py`：按 `images/train` 生成固定划分 `splits/train.txt|val.txt`。
- `tools/preview_unified_batch.py`：预览统一分割数据（polygon 安全增强、掩膜栅格化），生成示意图到 `tools/reports/`。
- `tools/train_unified_stage1.py`：Stage1 训练（简易分割网络 + FiLM），可选导出 ROI 与预览图。
- `tools/train_yolov13_stage1.py`：YOLOv13 风格的最小分割训练（统一 polygon→掩膜 + FiLM 注入）。支持 `--use_dcnv3` 回退占位；新增 `--use_ultra_head`（可选 proto+coeff 组装头），`--save_last`，`--eval_best`。
- `tools/train_ultra_stage1.py`：Ultralytics SegmentTrainer 快速对接脚本（猴补路径、标签转换为单类分割、PNG 回退、清理缓存），用于对比验证。
- `tools/train_stage2_teacher.py`：Stage2 教师模型训练（ROI 多头分类）。
- `tools/export_teacher_logits.py`：导出教师 `logits.npz` 供蒸馏使用。
- `tools/train_mask2morph_distill.py`：学生蒸馏训练（联合真实标签与教师 logits）。
- `tools/export_student_logits.py`：导出学生 logits（分析/主动学习）。
- `tools/infer_stage3_pipeline.py`：Stage3 一键推理（分割→ROI 分类→规则→几何参数→叠加图/JSON）。
- `tools/eval_stage3_report.py`：Stage3 评测与报告（混淆矩阵与统计）。
- `tools/refine_hypergraph_stage3.py`：超图（邻近聚团）离线细化工具，生成 refined JSON 与新版叠加图。
- `tools/select_uncertain_rois.py`：不确定 ROI 选择（主动学习辅助）。

- `models/simple_seg_film.py`：简易分割网络（FiLM 注入；可选 DCNv3 回退）。新增 `forward_feats` 以便外部分割头复用解码特征。
- `models/ultra_seg_head.py`：最小 YOLOv8 风格分割头（原型 `Proto` + 系数 `UltraSegHead`）；组合类 `FiLMUNetUltraSeg(backbone)`。
- `models/uscale_film.py`：像素尺度注入（µScale）：`PixelScaleNormalizer` 与 `FiLM2d`。
- `models/multitask_heads.py`：Stage2 多头分类器组合。

- `datasets/unified_seg_dataset.py`：统一分割数据集（解析 TXT 头部 `pixel_size_um`，返回 `collate_fn_unified`）。
- `datasets/roi_multitask_dataset.py` / `datasets/roi_with_teacher_logits.py`：ROI 分类与蒸馏数据集。

- `utils/aug_poly.py`：polygon-safe 增强（letterbox/仿射/翻转等）。
- `utils/mask_ops.py`：segments→掩膜栅格化与批量打包。
- `utils/losses_seg.py`：`BCEDiceLoss` 与二值 IoU 计算。
- `utils/vis_overlay.py`：叠加可视化（中文字体回退）。
- `utils/matching.py` / `utils/metrics_cls.py` / `utils/distill.py` / `utils/early_stop.py`：训练与评测辅助。

## 当前工作进展与产物（重要）
- 分割数据管道：统一 TXT（含头部元数据）解析、polygon 安全增强（letterbox/仿射/翻转）、掩膜栅格化与批量打包已全部跑通。
- 最小训练快验：已完成 `tools/train_yolov13_stage1.py` 的 1 个 epoch 快验并保存权重。
  - 运行命令（CPU 示例）：
    - `python g:\yoloV13\µSHM-YOLO\tools\train_yolov13_stage1.py --epochs 1 --batch_size 4 --imgsz 256 --device cpu --workers 0`
  - 输出：`g:\yoloV13\µSHM-YOLO\tools\runs\stage1_yolov13\best.pt`
  - 新增开关：`--use_ultra_head`（proto+coeff 可选占位头）、`--save_last`、`--eval_best`
  - 示例（CPU 快验）：`python g:\yoloV13\µSHM-YOLO\tools\train_yolov13_stage1.py --epochs 1 --batch_size 2 --imgsz 640 --use_ultra_head --save_last --eval_best`
  - 额外输出：`tools\runs\stage1_yolov13\last.pt`；并在训练结束后载入 `best.pt` 运行一次 `val`
  - 说明：数据集会自动过滤 `splits/train.txt|val.txt` 中不存在的 ID，避免因列表与 `images/*` 不一致而报错；建议将 `splits/val.txt` 对齐为 `images/val` 的 ID（如 39–49）。
- Ultralytics 引擎快速尝试：`tools/train_ultra_stage1.py` 可跑基线；若本机 `yolov13-main/ultralytics` 不齐全或包冲突，可能无法生成权重，当前建议以主程序脚本为准。
- Stage2（分类）与 Stage3（推理/评测）已具备：
  - 训练输出：`g:\yoloV13\runs\stage2_teacher\best_teacher.pt`、`g:\yoloV13\runs\stage2_student\best_student.pt`
  - 推理输出：`g:\yoloV13\runs\infer_stage3\predictions.json` 与若干 `*_overlay.jpg`
  - 评测输出：`g:\yoloV13\runs\eval_stage3\summary.json`、`cm_*.png`；细化版在 `runs/eval_stage3_hg/`

## 快速运行：YOLOv13 风格最小分割（FiLM 注入）
- 训练（CPU 示例）：
  - `python g:\yoloV13\µSHM-YOLO\tools\train_yolov13_stage1.py --epochs 10 --batch_size 4 --imgsz 640 --device cpu --split_dir g:\yoloV13\µSHM-YOLO\splits`
- 推理与评测（沿用现有 Stage3 脚本）：
  - 推理：`python g:\yoloV13\µSHM-YOLO\tools\infer_stage3_pipeline.py --cfg g:\yoloV13\µSHM-YOLO\yolov13_transformer_unified_v2_1.yaml --data_root g:\yoloV13\µSHM-YOLO\samples --stage1_weights g:\yoloV13\µSHM-YOLO\tools\runs\stage1_yolov13\best.pt --student_weights g:\yoloV13\runs\stage2_student\best_student.pt --out_dir g:\yoloV13\runs\infer_stage3 --imgsz 640 --device cpu --split train`
  - 评测：`python g:\yoloV13\µSHM-YOLO\tools\eval_stage3_report.py --data_root g:\yoloV13\µSHM-YOLO\samples --pred_json g:\yoloV13\runs\infer_stage3\predictions.json --imgsz 640 --iou_thr 0.5 --save_dir g:\yoloV13\runs\eval_stage3 --plot 1`



### Stage3 超图（邻近聚团）离线细化

为解决同一图像内的近邻细胞组件被分散赋值的问题，提供“超图（邻近聚团）离线细化”工具：对 `predictions.json` 中的实例按空间邻近进行聚团，并将 `cell_org` 重赋值为“聚团内的并集”。

- 运行细化（生成新版叠加图与 JSON）：

  - `python µSHM-YOLO\tools\refine_hypergraph_stage3.py --pred_json runs\infer_stage3\predictions.json --cfg µSHM-YOLO\yolov13_transformer_unified_v2_1.yaml --imgsz 640 --eps_factor 0.08 --min_size_px 16 --out_dir runs\infer_stage3_hg --plot 1`

  - 输出：
    - `runs/infer_stage3_hg/refined_predictions.json`（细化后的 JSON，`cell_org` 为列表并集）
    - `runs/infer_stage3_hg/*_overlay_hg.jpg`（新版叠加图，标签中 `cell_org` 显示并集名）

- 评测细化结果（生成新版报告与混淆矩阵）：

  - `python µSHM-YOLO\tools\eval_stage3_report.py --data_root µSHM-YOLO\samples --pred_json runs\infer_stage3_hg\refined_predictions.json --imgsz 640 --iou_thr 0.5 --save_dir runs\eval_stage3_hg --plot 1`

  - 输出：
    - `runs/eval_stage3_hg/summary.json`、`params_stats.csv`
    - `runs/eval_stage3_hg/cm_*.npy/png`（五头分类混淆矩阵）

- 预览：

  - 叠加图目录：`runs/infer_stage3_hg/`（可通过 `python -m http.server 8514` 启动本地预览）
  - 评测图表：`runs/eval_stage3_hg/`（可通过 `python -m http.server 8516` 启动本地预览）

- 说明与提示：
  - 细化后 `cell_org` 是“列表并集”，评测脚本已支持：更新混淆矩阵时若并集中包含 GT 类，计入该 GT；否则取并集首项作为预测类。
  - `--eps_factor` 为邻近半径系数（相对 `min(H,W)`），可按密度与像素尺寸调参；`--min_size_px` 用于忽略过小 ROI 噪声。
  - 若初次评测未生成 PNG，可将 `--iou_thr` 暂降为 `0.0` 以确认图表管线与目录权限，再恢复常规阈值查看真实指标。


推荐流程：数据质检 → 固定划分 → 训练 Stage1 → 训练 Stage2 教师 → 导出教师 logits → 蒸馏学生 → 导出学生 logits → 运行 Stage3。

---

## 数据质检与固定划分

- 统一数据集质检：

```powershell
python g:\yoloV13\µSHM-YOLO\tools\validate_unified_dataset.py \
  --data_root g:\yoloV13\µSHM-YOLO\samples \
  --cfg g:\yoloV13\µSHM-YOLO\yolov13_transformer_unified_v2_1.yaml \
  --report_dir g:\yoloV13\µSHM-YOLO\tools\reports
```

- 固定 train/val 划分（按 `images/train` 生成列表）：

```powershell
python g:\yoloV13\µSHM-YOLO\tools\make_fixed_split.py \
  --data_root g:\yoloV13\µSHM-YOLO\samples \
  --val_ratio 0.2 \
  --out_dir g:\yoloV13\µSHM-YOLO\splits
```

---

## Stage1 训练（分割与 ROI 导出）

最小循环脚本：`tools/train_unified_stage1.py`

```powershell
python g:\yoloV13\µSHM-YOLO\tools\train_unified_stage1.py \
  --cfg g:\yoloV13\µSHM-YOLO\yolov13_transformer_unified_v2_1.yaml \
  --data_root g:\yoloV13\µSHM-YOLO\samples \
  --epochs 10 --batch_size 2 --imgsz 640 \
  --max_instances 16 --film 1 --export_rois 1 \
  --split_dir g:\yoloV13\µSHM-YOLO\splits --weighted_sample 1
```

输出：
- `tools/reports/best_stage1.pt`、`tools/reports/last_stage1.pt`
- 训练/验证叠加预览图若干（用于快速核对）
- 若 `--export_rois 1`：导出 `tools/reports/rois.json` 与对应 ROI 图片

### 使用 Ultralytics SegmentTrainer（快速验证）

- 目的：用 Ultralytics 官方训练引擎在样例数据上快速跑通 Stage1 分割基线。
- 数据：保留统一标签不动，脚本会生成 `samples/labels-ultra/*` 与 `samples/data-ultra-seg.yaml`。
- 路径映射：运行时猴补 `images/* → labels-ultra/*`，确保标签路径正确。
- 运行命令：

```powershell
python g:\yoloV13\µSHM-YOLO\tools\train_ultra_stage1.py ^
  --epochs 1 --imgsz 256 --batch 2 --device cuda
```

- 输出：`µSHM-YOLO\tools\reports_ultra\stage1_ultra\`（日志与权重），以及一次 `val` 指标概要。
- 依赖：需存在 `g:\yoloV13\yolov13-main\ultralytics\`（本地 Ultralytics 源），否则无法导入。
- 标签说明：统一 TXT 的 `#` 元数据行会被跳过；每条实例转换为 `cls=0 + polygon` 的 YOLO 分割格式。
- 注意：若你后续改为多类分割，可将 `nc/names` 与转换逻辑调整为多类别。

---

## Stage2 教师训练与蒸馏学生

1) 教师模型训练（输入为 ROI 数据根目录，包含 `rois.json` 与 ROI 图像）：

```powershell
python g:\yoloV13\µSHM-YOLO\tools\train_stage2_teacher.py \
  --cfg g:\yoloV13\µSHM-YOLO\yolov13_transformer_unified_v2_1.yaml \
  --roi_root g:\yoloV13\µSHM-YOLO\tools\reports \
  --out_dir g:\yoloV13\runs\stage2_teacher \
  --epochs 40 --batch_size 64 --safe_loop 1
```

输出：`runs/stage2_teacher/best_teacher.pt`、`log.csv`、（可选）混淆矩阵 `cm_*.npy/png`。

2) 导出教师 logits（供蒸馏使用）：

```powershell
python g:\yoloV13\µSHM-YOLO\tools\export_teacher_logits.py \
  --cfg g:\yoloV13\µSHM-YOLO\yolov13_transformer_unified_v2_1.yaml \
  --roi_root g:\yoloV13\µSHM-YOLO\tools\reports \
  --weights g:\yoloV13\runs\stage2_teacher\best_teacher.pt \
  --out_npz g:\yoloV13\runs\stage2_teacher\teacher_logits.npz \
  --safe_loop 1
```

3) 蒸馏学生（Mask2Morph Distill；同时使用真实标签与教师 logits）：

```powershell
python g:\yoloV13\µSHM-YOLO\tools\train_mask2morph_distill.py \
  --cfg g:\yoloV13\µSHM-YOLO\yolov13_transformer_unified_v2_1.yaml \
  --roi_root g:\yoloV13\µSHM-YOLO\tools\reports \
  --teacher_npz g:\yoloV13\runs\stage2_teacher\teacher_logits.npz \
  --out_dir g:\yoloV13\runs\stage2_student \
  --epochs 40 --batch_size 64 --kd_weight 0.5 --kd_T 2.0 --safe_loop 1
```

输出：`runs/stage2_student/best_student.pt`、`log.csv`。

4) 学生 logits 导出（用于后续分析或主动学习）：

```powershell
python g:\yoloV13\µSHM-YOLO\tools\export_student_logits.py \
  --cfg g:\yoloV13\µSHM-YOLO\yolov13_transformer_unified_v2_1.yaml \
  --roi_root g:\yoloV13\µSHM-YOLO\tools\reports \
  --weights g:\yoloV13\runs\stage2_student\best_student.pt \
  --out_npz g:\yoloV13\runs\stage2_student\student_logits.npz \
  --safe_loop 1
```

---

## Stage3 推理与叠加可视化（当前已跑通）

本仓已提供一键脚本在样例数据上跑 Stage3（分割 → ROI 多头分类 → 规则检查 → 参数估算 → 叠加图与 JSON）。输出目录为 `g:\yoloV13\runs\infer_stage3`。

### 运行命令（CPU）

```powershell
python g:\yoloV13\µSHM-YOLO\tools\infer_stage3_pipeline.py \
  --cfg g:\yoloV13\µSHM-YOLO\yolov13_transformer_unified_v2_1.yaml \
  --data_root g:\yoloV13\µSHM-YOLO\samples \
  --stage1_weights g:\yoloV13\µSHM-YOLO\tools\reports\best_stage1.pt \
  --student_weights g:\yoloV13\runs\stage2_student\best_student.pt \
  --out_dir g:\yoloV13\runs\infer_stage3 \
  --imgsz 640 --device cpu --split train
```

完成后会生成：

- `g:\yoloV13\runs\infer_stage3\predictions.json`（每张图的 5 属性结果、规则检查与物理参数）
- 若干 `*_overlay.jpg`（每图的叠加可视化）

### 预览叠加图（浏览器）

```powershell
cd g:\yoloV13\runs\infer_stage3
python -m http.server 8512
```

浏览器打开 `http://localhost:8512/1_overlay.jpg` 即可查看（文件名按原图序号）。

### 叠加显示说明（中文字体与实例轮廓）

- 文本绘制已切换为 PIL 字体，支持中文显示；优先使用系统字体 `msyh.ttc` / `simhei.ttf` / `simsun.ttc`。
- 若系统缺少上述字体，文本会使用英文字体并将非 ASCII 字符替换为 `·`（请安装任意中文字体到系统字体目录即可恢复）。
- 实例轮廓：
  - 对每个掩膜绘制清晰轮廓与边框；同时进行 25% 透明度的颜色填充，避免“看不到内容”。
  - 降级提示（规则不通过）在标签尾部显示 `[demoted]`。

### 常见问题（可视化相关）

- 只看到整图边框或整屏泛色：
  - 分割阈值偏低导致整图被判为前景，可提高 `--thr` 到 `0.6–0.8` 再试；
  - 或检查 `--stage1_weights` 是否匹配当前数据与像素尺度配置（`microscope.pixel_size_um`）。
- 文本显示为问号：
  - 安装并启用中文字体（Windows 默认 `C:\Windows\Fonts\msyh.ttc` 可用）；
  - 或在 YAML 的 `classes.names_zh` 使用英文名临时替代。
- 色彩过重遮挡细节：
  - 叠加透明度默认为 0.25，可根据需要在 `utils\vis_overlay.py` 中调整 `alpha`。

### 复现说明（数据与参数）

- 归一化均值/方差取自 `dataset_stats.mean/std`（均以 `float32` 处理，避免 `float64` 升阶导致 PyTorch Double/Float 冲突）。
- 分类器加载逻辑：保留 ResNet50 预训练 `backbone.*` 为 `float32`；仅加载学生权重中非 backbone 的部分并强制为 `float32`。
- 规则引擎与 ID 映射与 YAML `classes.label_spaces` 保持一致（cell_organization 训练 4 类，推理映射回全局 ID）。

---

## Stage3 评测与报告（根据 9.md）

在完成 Stage3 推理后，运行评测脚本生成检测与多头分类报告：

```powershell
python g:\yoloV13\µSHM-YOLO\tools\eval_stage3_report.py \
  --data_root g:\yoloV13\µSHM-YOLO\samples \
  --pred_json g:\yoloV13\runs\infer_stage3\predictions.json \
  --imgsz 640 --iou_thr 0.5 \
  --save_dir g:\yoloV13\runs\eval_stage3 --plot 1
```

输出（默认到 `runs/eval_stage3/`）：
- `summary.json`：检测 `precision/recall/f1`、`bbox mIoU`、规则冲突率、五头 `acc/macro_f1`、TP/FP/FN 统计。
- `cm_*.npy/png`：五个头的混淆矩阵（NPY 与热力图 PNG）。
- `params_stats.csv`：10 个物理参数的均值与标准差（来自 `predictions.json.parameters`）。

可视化预览（浏览器）：

```powershell
cd g:\yoloV13\runs\eval_stage3
python -m http.server 8513
```

打开 `http://localhost:8513/cm_species.png`、`cm_cell_org.png` 等查看热力图。

提示：若早期模型导致整图被判为前景，匹配可能失败（TP≈0）。可将 `--iou_thr` 降至 `0.0–0.2` 以进行对齐核查，再回升阈值做正式评测。

---

## 主动学习挑样（Top‑K 不确定 ROI）

根据 Stage1+Stage3 的早期模型，筛选最不确定样本并导出 ROI 以供复标：

```powershell
python g:\yoloV13\µSHM-YOLO\tools\select_uncertain_rois.py \
  --cfg g:\yoloV13\µSHM-YOLO\yolov13_transformer_unified_v2_1.yaml \
  --weights g:\yoloV13\µSHM-YOLO\tools\reports\best_stage1.pt \
  --split train --k 32 \
  --out_dir g:\yoloV13\µSHM-YOLO\tools\reports\active_rois_topk
```

输出目录：`tools/reports/active_rois_topk`（含导出 ROI 与 `rois.json`）。

---

## 批量预览与快速核对

数据加载与几何一致增强的快速预览（生成到 `tools/reports`）：

```powershell
python g:\yoloV13\µSHM-YOLO\tools\preview_unified_batch.py \
  --cfg g:\yoloV13\µSHM-YOLO\yolov13_transformer_unified_v2_1.yaml \
  --data_root g:\yoloV13\µSHM-YOLO\samples
```

---

## 目录结构与输出约定（建议）

- `µSHM-YOLO/samples/images/train|val` 与 `µSHM-YOLO/samples/labels/train|val`
- `µSHM-YOLO/splits/train.txt|val.txt` 固定划分列表
- `µSHM-YOLO/tools/reports`：Stage1 可视化与权重、ROI 与质检报告
- `runs/stage2_teacher` 与 `runs/stage2_student`：分类器训练输出与日志
- `runs/infer_stage3`：推理叠加图与 `predictions.json`

---

## 运行提示（Windows/CPU 环境）

- 若出现多进程/内存相关报错，请将 `--safe_loop` 设为 `1`（教师训练/蒸馏/导出 logits 已内置此参数）。
- 为避免线程库冲突，脚本已在内部设置 `KMP_DUPLICATE_LIB_OK/OMP_NUM_THREADS/MKL_NUM_THREADS` 并限制 PyTorch 线程数为 1。
- 中文叠加文本需系统字体（如 `C:\Windows\Fonts\msyh.ttc`）；缺失时会回退为英文并以 `·` 替代非 ASCII。

---

## 下一步建议与缺口排查（面向 YOLOv13 主线）
- 统一 dataloader 在 YOLOv13 检测+分割主干上严格对齐：批次返回 `pixel_size_um` 并确保几何增强同步作用到 `segments`。
- 将 FiLM 注入扩展到 backbone/neck（如 S3/S4 与 P3/P2），记录 `pixel_size_um` 日志，保留默认像素尺度回退。
- 小目标结构开关核验：P2 输出、RepGFPN 训练态多分支/推理态重参数、DCNv3 接入与可用性回退。
- 正式分割头接线（proto+coeff 或 YOLOv13 自研头）：在保持 `forward(images, pixel_scale)` 接口的一致性下替换占位头。
- Stage2/Stage3 完整链路回归：教师/学生训练日志、混淆矩阵、规则降级与几何参数评估的稳定性检查。
- 超图在线融合（后续）：cell_organization 由 Mask2Morph head 与超图分支融合，训练期一致性损失、推理期置信融合。

## 快速路线图

- 跑通数据质检与固定划分；
- 训练 Stage1 → 导出 ROI；
- 训练 Stage2 教师 → 导出教师 logits；
- 蒸馏学生 → 导出学生 logits；
- 运行 Stage3 推理并浏览叠加图；
- 结合主动学习挑样迭代提升数据质量与模型表现。
