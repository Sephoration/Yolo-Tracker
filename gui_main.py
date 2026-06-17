# gui_main.py
"""
期末报告 — PySide6 GUI 整合程式
功能：
  - 影片播放 / 暂停 / 停止
  - 彩色轨迹追踪
  - 分车型计数 (car / bus / truck)
  - 敏感区域监控
  - FPS 显示
  - 计数统计表
  - 参数控制 (置信度/轨迹长度)

执行：
  python gui_main.py
"""

import sys
import os
import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QGridLayout, QFileDialog,
    QSlider, QTextEdit, QCheckBox
)

from yolo_tracker_base import (
    YOLOTracker, VIDEOS_DIR, OUTPUT_DIR, PROJECT_ROOT,
    Point, isInsidePolygon, adaptive_resize, CLS_NAMES
)


# ============================================================
# 影片处理线程（避免 GUI 卡顿）
# ============================================================
class VideoThread(QThread):
    change_pixmap = Signal(np.ndarray)
    update_count = Signal(dict)
    update_fps = Signal(float)
    update_zone = Signal(list)
    frame_processed = Signal(int, int)
    snapshot_taken = Signal(str)

    def __init__(self, video_path=None):
        super().__init__()
        self.video_path = video_path
        self.running = False
        self.paused = False
        self.tracker = None

        # 预设参数
        self.show_zone = True
        self.show_count = True
        self.colorful_trail = True
        self.show_lane_change = False
        self.show_trail = True

        # 影片来源搜寻
        self.search_dirs = [
            VIDEOS_DIR,
            os.path.join(PROJECT_ROOT, '..', 'Exp_Midterm', 'videos'),
            os.path.join(PROJECT_ROOT, '..', '..', 'YOLO26Tracking', 'videos'),
        ]

        # 敏感区域多边形（相对比例）— 梯形：底部宽(近端)、顶部窄(远端)
        self.ZONE_RATIO = [
            (0.12, 0.93),   # 左下 (近端)
            (0.88, 0.93),   # 右下 (近端)
            (0.66, 0.55),   # 右上 (远端)
            (0.34, 0.55),   # 左上 (远端)
        ]

    def find_video(self, filename):
        for d in self.search_dirs:
            abs_d = os.path.abspath(d)
            path = os.path.join(abs_d, filename)
            if os.path.exists(path):
                return path
        return None

    def set_video(self, video_path):
        self.video_path = video_path

    def run(self):
        if not self.video_path:
            return

        # 搜寻影片
        actual_path = self.find_video(self.video_path) if not os.path.exists(self.video_path) else self.video_path
        if not actual_path or not os.path.exists(actual_path):
            print(f"找不到影片: {self.video_path}")
            return

        cap = cv2.VideoCapture(actual_path)
        if not cap.isOpened():
            return

        ret, frame = cap.read()
        if not ret:
            cap.release()
            return

        frame_h, frame_w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 初始化追踪器
        self.tracker = YOLOTracker()
        self.tracker.colorful_trail = self.colorful_trail

        # 计数线
        count_line_y = int(frame_h * 0.60)
        self.tracker.count_line = ((0, count_line_y), (frame_w, count_line_y))
        self.tracker.count_enabled = True

        # 敏感区域
        zone_polygon = [
            (int(frame_w * rx), int(frame_h * ry))
            for rx, ry in self.ZONE_RATIO
        ]
        self.tracker.set_zone_polygon(zone_polygon)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.running = True
        while self.running:
            if self.paused:
                self.msleep(50)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            # 追踪（含变道分析、轨迹开关）
            annotated, pred_boxes = self.tracker.track(
                frame, draw_trail=self.show_trail,
                analyze_lane_change=self.show_lane_change
            )

            # 敏感区域
            if self.show_zone:
                annotated = self.tracker.draw_zone(annotated)
                current_inside = self.tracker.check_zone_intrusions(
                    pred_boxes, self.tracker.frame_id, fps
                )
                if current_inside:
                    for box in pred_boxes:
                        x1, y1, x2, y2, lbl, track_id = box
                        if track_id in current_inside:
                            snap_path = self.tracker.capture_snapshot(
                                annotated, track_id, x1, y1, x2, y2, lbl)
                            if snap_path:
                                self.snapshot_taken.emit(snap_path)
                            # 仅将文字标红（保持框原色）
                            cv2.putText(annotated, f"{lbl}({track_id})",
                                        (int(x1), int(y1) - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 计数线
            if self.show_count:
                annotated = self.tracker.draw_count_line(annotated)

            # FPS
            annotated = self.tracker.draw_fps(annotated)

            # 画面缩放到显示尺寸
            display_w = min(frame_w, 960)
            annotated = adaptive_resize(annotated, display_w)

            # 发送信号更新 GUI
            self.change_pixmap.emit(annotated)
            self.update_fps.emit(self.tracker.fps_smooth)
            self.update_count.emit({
                cls: self.tracker.count_data.get(cls, {'in': 0, 'out': 0})
                for cls in ['car', 'bus', 'truck', 'person']
            })

            # 入侵通知
            if self.tracker.zone_intrusions:
                intruders = [f"ID:{tid}({st['label']})"
                            for tid, st in self.tracker.zone_intrusions.items()]
                self.update_zone.emit(intruders)

            self.frame_processed.emit(self.tracker.frame_id, total_frames)

        cap.release()
        self.tracker = None
        self.running = False

    def stop(self):
        self.running = False
        self.paused = False

    def pause(self):
        self.paused = not self.paused


# ============================================================
# 主窗口
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("期末报告 — 目标追踪与分类计数系统")
        self.setMinimumSize(1050, 650)
        self.resize(1200, 700)

        # 状态
        self.thread = None
        self.current_video = None

        self._init_ui()

    def _init_ui(self):
        # ---- 整体容器 ----
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 6)
        main_layout.setSpacing(6)

        # ---- 顶栏标题 ----
        title_bar = QHBoxLayout()
        title_label = QLabel("🎯 目标追踪与分类计数系统")
        title_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        title_label.setStyleSheet("color: #1a365d; padding: 2px 0;")
        title_bar.addWidget(title_label)
        title_bar.addStretch()
        main_layout.addLayout(title_bar)

        # ============================================================
        # 主体：左影片 + 右面板
        # ============================================================
        body_layout = QHBoxLayout()
        body_layout.setSpacing(8)

        # --------------------------------------------------
        # 左侧：影片 + 底部控制
        # --------------------------------------------------
        left = QVBoxLayout()
        left.setSpacing(6)

        self.video_label = QLabel("请开启影片 (点击下方 [📂开启])")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("QLabel { background-color: #1a1a1a; color: #888; "
                                        "border: 2px solid #333; font-size: 18px; }")
        left.addWidget(self.video_label, stretch=1)

        # --- 控制行 ---
        row_ctrl = QHBoxLayout()
        row_ctrl.setSpacing(6)

        self.btn_open = QPushButton("📂 开启")
        self.btn_open.clicked.connect(self.open_video)
        row_ctrl.addWidget(self.btn_open)

        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        row_ctrl.addWidget(self.btn_play)

        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_stop.setEnabled(False)
        row_ctrl.addWidget(self.btn_stop)

        row_ctrl.addStretch()

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("font-weight: bold; color: #0a0; padding: 2px 8px;")
        row_ctrl.addWidget(self.fps_label)

        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setStyleSheet("color: #aaa; padding: 2px 8px;")
        row_ctrl.addWidget(self.frame_label)

        left.addLayout(row_ctrl)

        # --- 选项行 ---
        row_opt = QHBoxLayout()
        row_opt.setSpacing(10)

        self.chk_trail = QCheckBox("显示轨迹")
        self.chk_trail.setChecked(True)
        self.chk_trail.stateChanged.connect(self._on_option_change)
        row_opt.addWidget(self.chk_trail)

        self.chk_colorful = QCheckBox("彩色轨迹")
        self.chk_colorful.setChecked(True)
        self.chk_colorful.stateChanged.connect(self._on_option_change)
        row_opt.addWidget(self.chk_colorful)

        self.chk_count = QCheckBox("计数线")
        self.chk_count.setChecked(True)
        self.chk_count.stateChanged.connect(self._on_option_change)
        row_opt.addWidget(self.chk_count)

        self.chk_zone = QCheckBox("敏感区域")
        self.chk_zone.setChecked(True)
        self.chk_zone.stateChanged.connect(self._on_option_change)
        row_opt.addWidget(self.chk_zone)

        self.chk_lane = QCheckBox("变道分析")
        self.chk_lane.setChecked(False)
        self.chk_lane.stateChanged.connect(self._on_option_change)
        row_opt.addWidget(self.chk_lane)

        row_opt.addStretch()
        left.addLayout(row_opt)

        body_layout.addLayout(left, stretch=1)

        # --------------------------------------------------
        # 右侧：信息面板
        # --------------------------------------------------
        right = QVBoxLayout()
        right.setSpacing(6)

        # 车辆计数统计
        count_group = QGroupBox("📊 车辆计数统计")
        count_grid = QGridLayout()
        count_grid.setSpacing(4)
        count_grid.setContentsMargins(8, 12, 8, 8)

        count_grid.addWidget(QLabel("车种"), 0, 0)
        count_grid.addWidget(QLabel("进入 ↑"), 0, 1)
        count_grid.addWidget(QLabel("离开 ↓"), 0, 2)
        count_grid.addWidget(QLabel("合计"), 0, 3)

        self.count_labels = {}
        for i, cls_name in enumerate(['car', 'bus', 'truck', 'person'], start=1):
            self.count_labels[cls_name] = {}
            count_grid.addWidget(QLabel(f"  {CLS_NAMES.get(cls_name, cls_name)}"), i, 0)
            lbl_in = QLabel("0")
            lbl_in.setStyleSheet("color: #0f0; font-weight: bold;")
            count_grid.addWidget(lbl_in, i, 1)
            self.count_labels[cls_name]['in'] = lbl_in

            lbl_out = QLabel("0")
            lbl_out.setStyleSheet("color: #f00; font-weight: bold;")
            count_grid.addWidget(lbl_out, i, 2)
            self.count_labels[cls_name]['out'] = lbl_out

            lbl_total = QLabel("0")
            lbl_total.setStyleSheet("font-weight: bold;")
            count_grid.addWidget(lbl_total, i, 3)
            self.count_labels[cls_name]['total'] = lbl_total

        count_group.setLayout(count_grid)
        right.addWidget(count_group)

        # --- 参数控制 ---
        param_group = QGroupBox("⚙ 参数控制")
        param_layout = QVBoxLayout()
        param_layout.setSpacing(4)
        param_layout.setContentsMargins(8, 12, 8, 8)

        # 置信度
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("置信度:"))
        self.lbl_conf_val = QLabel("0.35")
        self.lbl_conf_val.setStyleSheet("font-weight: bold; color: #2b6e9c;")
        conf_row.addWidget(self.lbl_conf_val)
        param_layout.addLayout(conf_row)

        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(5, 95)
        self.slider_conf.setValue(35)
        self.slider_conf.valueChanged.connect(self._on_param_conf_changed)
        param_layout.addWidget(self.slider_conf)

        # 轨迹长度
        trail_row = QHBoxLayout()
        trail_row.addWidget(QLabel("轨迹长度:"))
        self.lbl_trail_val = QLabel("80")
        self.lbl_trail_val.setStyleSheet("font-weight: bold; color: #2b6e9c;")
        trail_row.addWidget(self.lbl_trail_val)
        param_layout.addLayout(trail_row)

        self.slider_trail = QSlider(Qt.Horizontal)
        self.slider_trail.setRange(10, 200)
        self.slider_trail.setValue(80)
        self.slider_trail.valueChanged.connect(self._on_param_trail_changed)
        param_layout.addWidget(self.slider_trail)

        param_group.setLayout(param_layout)
        right.addWidget(param_group)

        # 敏感区域入侵状态
        zone_group = QGroupBox("🚨 敏感区域入侵状态")
        zone_layout = QVBoxLayout()
        zone_layout.setContentsMargins(8, 12, 8, 8)
        self.zone_text = QTextEdit()
        self.zone_text.setReadOnly(True)
        self.zone_text.setText("等待检测...")
        self.zone_text.setStyleSheet("font-size: 12px;")
        zone_layout.addWidget(self.zone_text)
        zone_group.setLayout(zone_layout)
        right.addWidget(zone_group)

        # --- 入侵截图 ---
        snap_group = QGroupBox("📸 入侵截图")
        snap_layout = QVBoxLayout()
        snap_layout.setContentsMargins(8, 12, 8, 8)
        snap_layout.setSpacing(4)
        self.lbl_snap_count = QLabel("截图数: 0")
        self.lbl_snap_count.setStyleSheet("font-size: 12px; color: #555;")
        snap_layout.addWidget(self.lbl_snap_count)
        self.btn_open_snap = QPushButton("📂 打开截图文件夹")
        self.btn_open_snap.clicked.connect(self._open_snapshots_folder)
        snap_layout.addWidget(self.btn_open_snap)
        snap_group.setLayout(snap_layout)
        right.addWidget(snap_group)

        # 备注
        note_group = QGroupBox("📝 备注")
        note_layout = QVBoxLayout()
        note_layout.setContentsMargins(8, 12, 8, 8)
        note_text = QLabel(
            "[📂开启] 打开影片\n"
            "[▶ 播放]  [⏸ 暂停]  [⏹ 停止]\n"
            "• 黄线 = 计数参考线\n"
            "• 红区 = 敏感监控区域\n"
            "• 彩色轨迹 = 每辆车不同色"
        )
        note_text.setStyleSheet("color: #aaa; font-size: 12px;")
        note_layout.addWidget(note_text)
        note_group.setLayout(note_layout)
        right.addWidget(note_group)

        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(300)
        body_layout.addWidget(right_widget)

        main_layout.addLayout(body_layout, stretch=1)

    # ==================== 影片控制 ====================
    def open_video(self):
        """开启影片对话框"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择影片", VIDEOS_DIR,
            "影片 (*.mp4 *.avi *.mov *.mkv);;全部 (*.*)"
        )
        if path:
            self.start_video(path)

    def start_video(self, path):
        """开始播放影片"""
        self.stop_video()
        self.current_video = path
        self.btn_play.setText("⏸ 暂停")
        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self._run_thread()

    def _run_thread(self):
        """启动处理线程"""
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()

        self.thread = VideoThread(self.current_video)
        self.thread.show_zone = self.chk_zone.isChecked()
        self.thread.show_count = self.chk_count.isChecked()
        self.thread.show_trail = self.chk_trail.isChecked()
        self.thread.show_lane_change = self.chk_lane.isChecked()
        self.thread.colorful_trail = self.chk_colorful.isChecked()
        self.thread.change_pixmap.connect(self._update_image)
        self.thread.update_count.connect(self._update_count_display)
        self.thread.update_fps.connect(lambda v: self.fps_label.setText(f"FPS: {v:.1f}"))
        self.thread.update_zone.connect(self._update_zone_display)
        self.thread.frame_processed.connect(
            lambda cur, total: self.frame_label.setText(f"Frame: {cur}/{total}")
        )
        self.thread.snapshot_taken.connect(self._on_snapshot_taken)
        self.thread.finished.connect(self._on_thread_finished)
        self.thread.start()

    def toggle_play(self):
        """播放/暂停切换"""
        if self.thread and self.thread.isRunning():
            self.thread.pause()
            is_paused = self.thread.paused
            self.btn_play.setText("▶ 播放" if is_paused else "⏸ 暂停")

    def stop_video(self):
        """停止影片"""
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
        self.thread = None
        self.btn_play.setText("▶ 播放")
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.fps_label.setText("FPS: --")
        self.frame_label.setText("Frame: 0 / 0")
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("请开启影片 (点击下方 [📂开启])")
        self._reset_count_display()
        self.zone_text.setText("等待检测...")
        self._snap_count = 0
        self.lbl_snap_count.setText("截图数: 0")

    def _on_option_change(self):
        """选项变更时同步到线程"""
        if self.thread and self.thread.tracker:
            self.thread.show_zone = self.chk_zone.isChecked()
            self.thread.show_count = self.chk_count.isChecked()
            self.thread.show_trail = self.chk_trail.isChecked()
            self.thread.show_lane_change = self.chk_lane.isChecked()
            self.thread.colorful_trail = self.chk_colorful.isChecked()
            if self.thread.tracker:
                self.thread.tracker.colorful_trail = self.chk_colorful.isChecked()

    # ==================== 参数控制 ====================
    def _on_param_conf_changed(self, val):
        v = val / 100.0
        self.lbl_conf_val.setText(f"{v:.2f}")
        if self.thread and self.thread.tracker:
            self.thread.tracker.conf = v

    def _on_param_trail_changed(self, val):
        self.lbl_trail_val.setText(str(val))
        if self.thread and self.thread.tracker:
            self.thread.tracker.trail_length = val

    # ==================== GUI 更新 ====================
    def _update_image(self, cv_img):
        """更新影片画面"""
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _update_count_display(self, data):
        """更新计数面板"""
        for cls_name in ['car', 'bus', 'truck', 'person']:
            d = data.get(cls_name, {'in': 0, 'out': 0})
            total = d['in'] + d['out']
            if cls_name in self.count_labels:
                self.count_labels[cls_name]['in'].setText(str(d['in']))
                self.count_labels[cls_name]['out'].setText(str(d['out']))
                self.count_labels[cls_name]['total'].setText(str(total))

    def _reset_count_display(self):
        for cls_name in ['car', 'bus', 'truck', 'person']:
            if cls_name in self.count_labels:
                self.count_labels[cls_name]['in'].setText("0")
                self.count_labels[cls_name]['out'].setText("0")
                self.count_labels[cls_name]['total'].setText("0")

    def _update_zone_display(self, intruders):
        """更新敏感区域状态"""
        if intruders:
            text = "🚨 入侵中:\n" + "\n".join(f"  • {i}" for i in intruders[-5:])
            self.zone_text.setText(text)
            self.zone_text.setStyleSheet("color: #f00; font-weight: bold;")
        else:
            self.zone_text.setText("✅ 区域内无入侵")
            self.zone_text.setStyleSheet("color: #0f0;")

    def _on_thread_finished(self):
        """线程结束处理"""
        self.btn_play.setText("▶ 播放")
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(False)

    # ==================== 截图 ====================
    def _on_snapshot_taken(self, path):
        """截图完成时更新计数"""
        if not hasattr(self, '_snap_count'):
            self._snap_count = 0
        self._snap_count += 1
        self.lbl_snap_count.setText(f"截图数: {self._snap_count}")

    def _open_snapshots_folder(self):
        """打开截图文件夹"""
        from yolo_tracker_base import SNAPSHOTS_DIR
        snap_dir = os.path.abspath(SNAPSHOTS_DIR)
        if os.path.exists(snap_dir):
            os.startfile(snap_dir)
        else:
            os.makedirs(snap_dir, exist_ok=True)
            os.startfile(snap_dir)

    def closeEvent(self, event):
        self.stop_video()
        event.accept()


# ============================================================
# 启动
# ============================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
