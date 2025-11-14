"""
Qt 可视化界面：串联 AIplan/14 与 0 的现有功能，展示：
- 检测/分割可视化（runs/infer_stage3 的 *_overlay.jpg）
- 超图可视化（runs/infer_stage3_hg 的 *_overlay_hg.jpg）
- 评估指标与混淆矩阵（runs/eval_stage3 与 runs/eval_stage3_hg）
- （可选）一键运行推理流程：调用 tools/infer_stage3_pipeline.py

运行示例：
python tools/qt_pipeline_viewer.py \
  --runs_dir g:/yoloV13/runs/infer_stage3 \
  --runs_hg_dir g:/yoloV13/runs/infer_stage3_hg \
  --eval_dir g:/yoloV13/runs/eval_stage3 \
  --eval_hg_dir g:/yoloV13/runs/eval_stage3_hg

依赖：优先使用 PyQt5；若未安装则尝试 PySide6。
"""
import os
import sys
import json
import argparse
from pathlib import Path

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    QT_LIB = 'PyQt5'
except Exception:
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        QT_LIB = 'PySide6'
    except Exception as e:
        raise ImportError(f"请先安装 PyQt5 或 PySide6：{e}")


class ImagePairViewer(QtWidgets.QWidget):
    """左右并排显示：baseline overlay 与 hypergraph overlay"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.lbl_left = QtWidgets.QLabel('Baseline Overlay')
        self.lbl_right = QtWidgets.QLabel('Hypergraph Overlay')
        self.lbl_left.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_right.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_left.setBackgroundRole(QtGui.QPalette.Base)
        self.lbl_right.setBackgroundRole(QtGui.QPalette.Base)
        self.lbl_left.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.lbl_right.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.lbl_left.setScaledContents(True)
        self.lbl_right.setScaledContents(True)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.lbl_left)
        layout.addWidget(self.lbl_right)

    def set_images(self, left_path: str, right_path: str | None):
        if left_path and os.path.exists(left_path):
            pix = QtGui.QPixmap(left_path)
            self.lbl_left.setPixmap(pix)
        else:
            self.lbl_left.setText('Baseline overlay 不存在')

        if right_path and os.path.exists(right_path):
            pix = QtGui.QPixmap(right_path)
            self.lbl_right.setPixmap(pix)
        else:
            self.lbl_right.setText('Hypergraph overlay 不存在')


class MetricsPanel(QtWidgets.QWidget):
    """展示 summary.json 文本与混淆矩阵 PNG"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.txt = QtWidgets.QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.img_species = QtWidgets.QLabel('cm_species')
        self.img_cell_org = QtWidgets.QLabel('cm_cell_org')
        self.img_shape = QtWidgets.QLabel('cm_shape')
        self.img_flagella = QtWidgets.QLabel('cm_flagella')
        self.img_chloroplast = QtWidgets.QLabel('cm_chloroplast')
        for lbl in [self.img_species, self.img_cell_org, self.img_shape, self.img_flagella, self.img_chloroplast]:
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setScaledContents(True)

        grid = QtWidgets.QGridLayout(self)
        grid.addWidget(self.txt, 0, 0, 1, 2)
        grid.addWidget(self.img_species, 1, 0)
        grid.addWidget(self.img_cell_org, 1, 1)
        grid.addWidget(self.img_shape, 2, 0)
        grid.addWidget(self.img_flagella, 2, 1)
        grid.addWidget(self.img_chloroplast, 3, 0)

    def load_eval_dir(self, eval_dir: str):
        self.txt.setPlainText('')
        for lbl in [self.img_species, self.img_cell_org, self.img_shape, self.img_flagella, self.img_chloroplast]:
            lbl.clear()
        if not eval_dir or not os.path.isdir(eval_dir):
            self.txt.setPlainText('评估目录不存在')
            return
        summary = os.path.join(eval_dir, 'summary.json')
        if os.path.exists(summary):
            try:
                data = json.load(open(summary, 'r', encoding='utf-8'))
                self.txt.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))
            except Exception as e:
                self.txt.setPlainText(f'读取 summary.json 失败：{e}')
        else:
            self.txt.setPlainText('未找到 summary.json')

        def set_img(lbl: QtWidgets.QLabel, name: str):
            p = os.path.join(eval_dir, name)
            if os.path.exists(p):
                lbl.setPixmap(QtGui.QPixmap(p))
            else:
                lbl.setText(f'{name} 不存在')
        set_img(self.img_species, 'cm_species.png')
        set_img(self.img_cell_org, 'cm_cell_org.png')
        set_img(self.img_shape, 'cm_shape.png')
        set_img(self.img_flagella, 'cm_flagella.png')
        set_img(self.img_chloroplast, 'cm_chloroplast.png')


class PipelineWindow(QtWidgets.QMainWindow):
    def __init__(self, runs_dir: str, runs_hg_dir: str, eval_dir: str, eval_hg_dir: str):
        super().__init__()
        self.setWindowTitle(f'µSHM-YOLO Pipeline Viewer ({QT_LIB})')
        self.resize(1200, 800)

        # 状态数据
        self.runs_dir = runs_dir
        self.runs_hg_dir = runs_hg_dir
        self.eval_dir = eval_dir
        self.eval_hg_dir = eval_hg_dir
        self.pred_json = None
        self.pred_map = {}
        # 项目根与默认 YAML 路径
        self.root_dir = os.path.dirname(os.path.dirname(__file__))
        self.default_yaml = os.path.join(self.root_dir, 'yolov13_transformer_unified_v2_1.yaml')

        # 左侧文件列表 + 顶部路径选择
        self.list_images = QtWidgets.QListWidget()
        self.list_images.itemSelectionChanged.connect(self._on_select)

        btn_reload = QtWidgets.QPushButton('重新加载')
        btn_reload.clicked.connect(self._reload_runs)
        self.edit_runs = QtWidgets.QLineEdit(self.runs_dir)
        self.edit_runs_hg = QtWidgets.QLineEdit(self.runs_hg_dir)
        self.btn_browse_runs = QtWidgets.QPushButton('选择 baseline 目录')
        self.btn_browse_runs.clicked.connect(lambda: self._browse_dir(self.edit_runs))
        self.btn_browse_runs_hg = QtWidgets.QPushButton('选择 hypergraph 目录')
        self.btn_browse_runs_hg.clicked.connect(lambda: self._browse_dir(self.edit_runs_hg))

        # 右侧图像对 + 详情 + 评估切换
        self.viewer = ImagePairViewer()
        self.details = QtWidgets.QPlainTextEdit(); self.details.setReadOnly(True)
        self.metrics = MetricsPanel()
        self.tabs = QtWidgets.QTabWidget()
        tab_view = QtWidgets.QWidget(); lv = QtWidgets.QVBoxLayout(tab_view); lv.addWidget(self.viewer); lv.addWidget(self.details)
        tab_metrics_base = QtWidgets.QWidget(); mv1 = QtWidgets.QVBoxLayout(tab_metrics_base); mv1.addWidget(self.metrics)
        tab_metrics_hg = QtWidgets.QWidget(); self.metrics_hg = MetricsPanel(); mv2 = QtWidgets.QVBoxLayout(tab_metrics_hg); mv2.addWidget(self.metrics_hg)
        self.tabs.addTab(tab_view, '可视化')
        self.tabs.addTab(tab_metrics_base, '指标-Baseline')
        self.tabs.addTab(tab_metrics_hg, '指标-Hypergraph')

        # 顶部工具栏：一键运行推理 + 单图分析
        self.stage1_edit = QtWidgets.QLineEdit('g:/yoloV13/µSHM-YOLO/tools/reports/best_stage1.pt')
        self.stage2_edit = QtWidgets.QLineEdit('g:/yoloV13/runs/stage2_student/best_student.pt')
        self.out_dir_edit = QtWidgets.QLineEdit('g:/yoloV13/runs/infer_single')
        self.cfg_edit = QtWidgets.QLineEdit(self.default_yaml)
        self.run_btn = QtWidgets.QPushButton('运行推理管线')
        self.run_btn.clicked.connect(self._run_pipeline)
        # 单图分析控件
        self.image_edit = QtWidgets.QLineEdit('g:/yoloV13/µSHM-YOLO/samples/images/val/39.png')
        self.btn_pick_image = QtWidgets.QPushButton('选择图片')
        self.btn_pick_image.clicked.connect(lambda: self._browse_file(self.image_edit))
        self.btn_analyze_image = QtWidgets.QPushButton('分析该图片')
        self.btn_analyze_image.clicked.connect(self._analyze_single_image)
        self.log_box = QtWidgets.QPlainTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(150)

        # 布局
        left_panel = QtWidgets.QWidget(); left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.addWidget(QtWidgets.QLabel('Baseline 目录'))
        left_layout.addWidget(self.edit_runs); left_layout.addWidget(self.btn_browse_runs)
        left_layout.addWidget(QtWidgets.QLabel('Hypergraph 目录'))
        left_layout.addWidget(self.edit_runs_hg); left_layout.addWidget(self.btn_browse_runs_hg)
        left_layout.addWidget(btn_reload)
        left_layout.addWidget(QtWidgets.QLabel('图像列表'))
        left_layout.addWidget(self.list_images)

        right_panel = QtWidgets.QWidget(); right_layout = QtWidgets.QVBoxLayout(right_panel)
        run_form = QtWidgets.QHBoxLayout()
        run_form.addWidget(QtWidgets.QLabel('Stage1权重'))
        run_form.addWidget(self.stage1_edit)
        run_form.addWidget(QtWidgets.QLabel('Stage2权重'))
        run_form.addWidget(self.stage2_edit)
        run_form.addWidget(QtWidgets.QLabel('YAML配置'))
        run_form.addWidget(self.cfg_edit)
        run_form.addWidget(QtWidgets.QLabel('输出目录'))
        run_form.addWidget(self.out_dir_edit)
        run_form.addWidget(self.run_btn)
        # 单图分析表单（第二行）
        single_form = QtWidgets.QHBoxLayout()
        single_form.addWidget(QtWidgets.QLabel('单图路径'))
        single_form.addWidget(self.image_edit)
        single_form.addWidget(self.btn_pick_image)
        single_form.addWidget(self.btn_analyze_image)
        right_layout.addLayout(run_form)
        right_layout.addLayout(single_form)
        right_layout.addWidget(self.tabs)
        right_layout.addWidget(QtWidgets.QLabel('运行日志'))
        right_layout.addWidget(self.log_box)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 3)
        self.setCentralWidget(splitter)

        # 初始加载
        self._reload_runs()
        self.metrics.load_eval_dir(self.eval_dir)
        self.metrics_hg.load_eval_dir(self.eval_hg_dir)

    def _browse_dir(self, edit: QtWidgets.QLineEdit):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, '选择目录', edit.text() or '')
        if d:
            edit.setText(d)

    def _browse_file(self, edit: QtWidgets.QLineEdit):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择图片文件', edit.text() or '',
                                                      'Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)')
        if fn:
            edit.setText(fn)

    def _reload_runs(self):
        self.runs_dir = self.edit_runs.text().strip()
        self.runs_hg_dir = self.edit_runs_hg.text().strip()
        self.list_images.clear()
        # baseline overlays
        overlays = []
        if os.path.isdir(self.runs_dir):
            for fn in os.listdir(self.runs_dir):
                if fn.endswith('_overlay.jpg'):
                    overlays.append(fn)
        overlays.sort(key=lambda x: int(os.path.basename(x).split('_')[0]) if os.path.basename(x).split('_')[0].isdigit() else x)
        self.list_images.addItems(overlays)
        # 自动选中第一项，避免用户看不到任何可视化
        if overlays:
            self.list_images.setCurrentRow(0)
            # 立即触发一次详情与图片刷新
            QtCore.QTimer.singleShot(0, self._on_select)
        # 读 predictions.json
        pred_path = os.path.join(self.runs_dir, 'predictions.json')
        self.pred_json = None
        self.pred_map = {}
        if os.path.exists(pred_path):
            try:
                data = json.load(open(pred_path, 'r', encoding='utf-8'))
                self.pred_json = data
                # 建索引：按文件 stem 或序号映射
                if isinstance(data, list):
                    for item in data:
                        key = str(item.get('id') or Path(item.get('image_path', '')).stem)
                        self.pred_map[key] = item
                elif isinstance(data, dict):
                    for k, v in data.items():
                        self.pred_map[str(k)] = v
            except Exception as e:
                self.log_box.appendPlainText(f'读取 predictions.json 失败：{e}')

    def _on_select(self):
        items = self.list_images.selectedItems()
        if not items:
            return
        name = items[0].text()
        base_path = os.path.join(self.runs_dir, name)
        # 匹配 hypergraph 对应文件
        stem = name.replace('_overlay.jpg', '')
        hg_name = f'{stem}_overlay_hg.jpg'
        hg_path = os.path.join(self.runs_hg_dir, hg_name)
        self.viewer.set_images(base_path, hg_path if os.path.exists(hg_path) else None)

        # 详情
        info_lines = []
        info_lines.append(f'图像：{name}')
        # 查预测细节（基于 predictions.json 的结构：detections + parameters）
        key_candidates = [stem, stem.split('_')[0]]
        for k in key_candidates:
            if k in self.pred_map:
                item = self.pred_map[k]
                dets = item.get('detections', [])
                num_instances = len(dets)
                info_lines.append(f'实例数：{num_instances}')
                # species 分布（使用 ids5_final/species）
                species_counts = {}
                for d in dets:
                    ids = d.get('ids5_final') or d.get('ids5_raw') or {}
                    sp = ids.get('species')
                    if sp is not None:
                        species_counts[sp] = species_counts.get(sp, 0) + 1
                if species_counts:
                    info_lines.append('species分布：' + ', '.join([f'{kk}:{vv}' for kk, vv in sorted(species_counts.items(), key=lambda x: -x[1])]))
                # 几何参数
                params = item.get('parameters', {})
                if params:
                    info_lines.append('几何参数：')
                    for key, val in params.items():
                        info_lines.append(f'  - {key}: {val:.4f}')
                break
        self.details.setPlainText('\n'.join(info_lines))

    def _run_pipeline(self):
        # 调用 tools/infer_stage3_pipeline.py，将输出目录设置到 out_dir_edit
        stage1 = self.stage1_edit.text().strip()
        stage2 = self.stage2_edit.text().strip()
        out_dir = self.out_dir_edit.text().strip()
        cfg_path = self.cfg_edit.text().strip()
        if not stage1 or not os.path.exists(stage1):
            self.log_box.appendPlainText('请提供有效的 Stage1 权重路径')
            return
        if not stage2 or not os.path.exists(stage2):
            self.log_box.appendPlainText('请提供有效的 Stage2 权重路径')
            return
        if not cfg_path or not os.path.exists(cfg_path):
            self.log_box.appendPlainText('请提供有效的 YAML 配置路径')
            return
        os.makedirs(out_dir, exist_ok=True)

        # 在子进程运行，避免阻塞GUI
        import subprocess
        cmd = [sys.executable, os.path.join('tools', 'infer_stage3_pipeline.py'),
               '--cfg', cfg_path,
               '--stage1_weights', stage1,
               '--student_weights', stage2,
               '--out_dir', out_dir]
        self.log_box.appendPlainText('运行：' + ' '.join(cmd))

        def run_and_update():
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=os.path.dirname(__file__), text=True)
                for line in proc.stdout:
                    self.log_box.appendPlainText(line.rstrip())
                ret = proc.wait()
                if ret == 0:
                    self.log_box.appendPlainText('推理完成，重新加载结果...')
                    self.edit_runs.setText(out_dir)
                    self._reload_runs()
                else:
                    self.log_box.appendPlainText(f'推理失败（退出码{ret}）')
            except Exception as e:
                self.log_box.appendPlainText(f'推理执行错误：{e}')

        QtCore.QTimer.singleShot(0, run_and_update)

    def _analyze_single_image(self):
        # 使用单图路径运行 Stage3 推理，然后运行超图细化
        stage1 = self.stage1_edit.text().strip()
        stage2 = self.stage2_edit.text().strip()
        cfg_path = self.cfg_edit.text().strip()
        img_path = self.image_edit.text().strip()
        base_out = self.out_dir_edit.text().strip() or os.path.join('g:/yoloV13', 'runs', 'infer_single')
        out_dir = os.path.join(base_out, 'single')
        out_dir_hg = out_dir + '_hg'
        if not (img_path and os.path.exists(img_path)):
            self.log_box.appendPlainText('请先选择有效的图片文件')
            return
        if not (stage1 and os.path.exists(stage1)):
            self.log_box.appendPlainText('请提供有效的 Stage1 权重路径')
            return
        if not (stage2 and os.path.exists(stage2)):
            self.log_box.appendPlainText('请提供有效的 Stage2 权重路径')
            return
        if not (cfg_path and os.path.exists(cfg_path)):
            self.log_box.appendPlainText('请提供有效的 YAML 配置路径')
            return
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir_hg, exist_ok=True)

        import subprocess
        tools_dir = os.path.dirname(__file__)
        # 1) 单图推理（修复路径：避免 tools/tools 拼接）
        cmd1 = [sys.executable, os.path.join(tools_dir, 'infer_stage3_pipeline.py'),
                '--cfg', cfg_path,
                '--image', img_path,
                '--stage1_weights', stage1,
                '--student_weights', stage2,
                '--out_dir', out_dir]
        self.log_box.appendPlainText('单图推理：' + ' '.join(cmd1))

        def run_pipeline_and_refine():
            try:
                proc1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=os.path.dirname(__file__), text=True)
                for line in proc1.stdout:
                    self.log_box.appendPlainText(line.rstrip())
                ret1 = proc1.wait()
                if ret1 != 0:
                    self.log_box.appendPlainText(f'单图推理失败（退出码{ret1}）')
                    return
                # 2) 超图细化
                pred_json = os.path.join(out_dir, 'predictions.json')
                cmd2 = [sys.executable, os.path.join(tools_dir, 'refine_hypergraph_stage3.py'),
                        '--pred_json', pred_json,
                        '--cfg', cfg_path,
                        '--out_dir', out_dir_hg]
                self.log_box.appendPlainText('超图细化：' + ' '.join(cmd2))
                proc2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=tools_dir, text=True)
                for line in proc2.stdout:
                    self.log_box.appendPlainText(line.rstrip())
                ret2 = proc2.wait()
                if ret2 == 0:
                    self.log_box.appendPlainText('单图分析完成，载入可视化结果')
                    self.edit_runs.setText(out_dir)
                    self.edit_runs_hg.setText(out_dir_hg)
                    self._reload_runs()
                    # 聚焦到可视化页
                    self.tabs.setCurrentIndex(0)
                else:
                    self.log_box.appendPlainText(f'超图细化失败（退出码{ret2}）')
            except Exception as e:
                self.log_box.appendPlainText(f'执行错误：{e}')

        QtCore.QTimer.singleShot(0, run_pipeline_and_refine)


def main():
    ap = argparse.ArgumentParser(description='µSHM-YOLO Qt 可视化界面（检测/超图/指标）')
    ap.add_argument('--runs_dir', type=str, default='g:/yoloV13/runs/infer_stage3')
    ap.add_argument('--runs_hg_dir', type=str, default='g:/yoloV13/runs/infer_stage3_hg')
    ap.add_argument('--eval_dir', type=str, default='g:/yoloV13/runs/eval_stage3')
    ap.add_argument('--eval_hg_dir', type=str, default='g:/yoloV13/runs/eval_stage3_hg')
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    w = PipelineWindow(args.runs_dir, args.runs_hg_dir, args.eval_dir, args.eval_hg_dir)
    w.show()
    # 兼容 PyQt5 与 PySide6 的 exec/exec_ 差异
    exec_fn = getattr(app, 'exec', None) or getattr(app, 'exec_', None)
    sys.exit(exec_fn())


if __name__ == '__main__':
    main()