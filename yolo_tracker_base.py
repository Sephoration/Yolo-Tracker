# yolo_tracker_base.py (期末报告增强版)
"""
YOLO Tracker 基础类 — 期末报告整合版
支援功能：
  - 基础追踪、轨迹、方向分析 (Part A/B/C)
  - 动态轨迹颜色 (每 ID 不同色)
  - 分车型计数 (car/truck/bus)
  - 敏感区域监控 (多边形 + 射线投射)
  - 异常对象截图 (进入区域自动裁切保存)
  - FPS 显示
"""

import cv2
import math
import time
import hashlib
from datetime import datetime
import numpy as np
import os
from ultralytics import YOLO


# ========== 项目路径 ==========
def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = get_project_root()
PROJECT_DIR = os.path.dirname(os.path.dirname(PROJECT_ROOT))  # Course_Deep_Learning
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
VIDEOS_DIR = os.path.join(PROJECT_ROOT, 'videos')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'videos_output')
SNAPSHOTS_DIR = os.path.join(PROJECT_ROOT, 'snapshots')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

# ========== 模型搜寻路径（多来源） ==========
DEFAULT_MODEL_PATHS = [
    os.path.join(MODELS_DIR, 'yolo26n.pt'),
    os.path.join(MODELS_DIR, 'yolo26s.pt'),
    os.path.join(MODELS_DIR, 'yolov8n.pt'),
    os.path.join(MODELS_DIR, 'yolov9t.pt'),
    # 跨资料夹搜寻
    os.path.join(PROJECT_ROOT, '..', 'Exp_Midterm', 'models', 'yolo26s.pt'),
    os.path.join(PROJECT_ROOT, '..', 'Exp_Midterm', 'models', 'yolo26n.pt'),
    os.path.join('..', '..', 'YOLO26Tracking', 'models', 'yolo26n.pt'),
]
if PROJECT_DIR:
    DEFAULT_MODEL_PATHS += [
        os.path.join(PROJECT_DIR, 'YOLO26Tracking', 'models', 'yolo26n.pt'),
        os.path.join(PROJECT_DIR, 'YOLO26Tracking', 'models', 'yolov8n.pt'),
    ]


# ========== 全局配置 ==========
OBJ_LIST = ['car', 'bus', 'truck', 'person']
CLS_NAMES = {
    'car': '汽车',
    'bus': '公交',
    'truck': '卡车',
    'person': '人员',
}
COLORS = {
    'person': (0, 255, 0),
    'car': (255, 0, 0),
    'bus': (0, 0, 255),
    'truck': (0, 255, 255)
}
TRAIL_COLOR = (255, 0, 255)     # 预设轨迹颜色（未启用多彩时使用）


# ============================================================
# 几何辅助函数
# ============================================================
class Point:
    """二维点类"""
    def __init__(self, x, y):
        self.x = x
        self.y = y


def isInsidePolygon(pt, polygon):
    """
    射线投射法判断点是否在多边形内部
    Args:
        pt: Point 物件
        polygon: [(x1,y1), (x2,y2), ...]
    Returns: True=在内部
    """
    x, y = pt.x, pt.y
    n = len(polygon)
    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)):
            intersect_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < intersect_x:
                inside = not inside
    return inside


def drawAndFillPolygon(frame, polygon, color, alpha=0.3):
    """绘制半透明填充多边形 + 边框"""
    overlay = frame.copy()
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=3)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def generate_trail_color(track_id):
    """根据 Track ID 产生固定的独特颜色 (BGR)"""
    h = hashlib.md5(str(track_id).encode()).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    # 确保不要太暗
    if r + g + b < 200:
        r = min(255, r + 100)
        g = min(255, g + 100)
        b = min(255, b + 100)
    return (b, g, r)  # OpenCV 是 BGR


def adaptive_resize(frame, max_width):
    """按最大宽度等比例缩放"""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        return cv2.resize(frame, (int(max_width), int(h * scale)))
    return frame


# ============================================================
# YOLO Tracker 主类
# ============================================================
class YOLOTracker:

    def __init__(self, model_path=None, device=None):
        if model_path is None:
            model_path = self._find_model()
        if model_path is None:
            raise FileNotFoundError(
                f"未找到模型文件！请放入 {MODELS_DIR} 或 Exp_Midterm/models/ 目录"
            )

        self.device = device if device else ('cuda' if __import__('torch').cuda.is_available() else 'cpu')
        print(f"加载模型: {model_path}")
        print(f"使用设备: {self.device}")
        self.model = YOLO(model_path)
        self.img_size = 640
        self.conf = 0.35
        self.iou = 0.70

        # ---- 轨迹记录 ----
        self.trails = {}
        self.lost_counter = {}
        self.trail_length = 80
        self.lost_threshold = 20

        # ---- Part A: 方向分析（保留期中相容性） ----
        self.angle_history = {}
        self.angle_diffs = {}
        self.analysis_records = []
        self.frame_id = 0

        # ---- Part B: 监控区间 ----
        self.valid_y_min = 120
        self.valid_y_max = 600
        self.angle_step = 3          # 期中版：跳帧3帧
        self.perspective_enabled = True
        self.short_window_size = 8               # 期中版
        self.short_acc_threshold = 25            # 期中版
        self.short_consistent_ratio = 0.55       # 期中版
        self.long_window_size = 20               # 期中版
        self.long_acc_threshold = 35
        self.long_consistent_ratio = 0.50        # 期中版
        self.long2_window_size = 60
        self.long2_acc_threshold = 20
        self.long2_consistent_ratio = 0.40
        self.trajectory_window_size = 30         # 期中版
        self.min_lateral_shift = 40              # 期中版 40px
        self.min_lateral_ratio = 0.15
        self.min_x_consistent_ratio = 0.50       # 期中版
        self.short_net_threshold = 15
        self.long_net_threshold = 25
        self.long2_net_threshold = 20
        self.smooth_alpha = 0.35                 # 期中版 0.35
        self.trails_smooth = {}
        self.abnormal_angle_diff_threshold = 20.0  # 期中版 20.0
        self.min_speed = 2.0
        self.lateral_axis = 'x'
        self.lane_change_results = {}

        # ---- 期末：动态轨迹颜色 ----
        self.colorful_trail = True          # True=每ID不同色, False=统一色
        self.trail_color_cache = {}         # {track_id: (B,G,R)}

        # ---- 期末：FPS 计算 ----
        self.fps_values = []
        self.fps_smooth = 0.0
        self._last_time = time.time()

        # ---- 期末：计数系统 ----
        self.count_line = None              # ((x1,y1),(x2,y2)) 计数参考线
        self.count_enabled = False
        self.count_data = {
            'car':    {'in': 0, 'out': 0},
            'bus':    {'in': 0, 'out': 0},
            'truck':  {'in': 0, 'out': 0},
            'person': {'in': 0, 'out': 0},
        }
        self._count_history = {}            # {track_id: {'last_side': 'above'/'below', 'counted': False}}

        # ---- 期末：敏感区域 ----
        self.zone_polygon = None            # [(x1,y1), (x2,y2), ...]
        self.zone_color = (0, 0, 255)       # 红色 BGR
        self.zone_alpha = 0.3
        self.zone_enabled = False
        self.zone_intrusions = {}           # {track_id: {...}}
        self.zone_inside_ids = set()        # 上一帧在区域内者

        # ---- 期末：异常截图 ----
        self.snapshot_enabled = True
        self.snapshot_dir = SNAPSHOTS_DIR
        self._snapshot_logged = set()       # 已截图过的 track_id 避免重复

    # ==================== 模型查找 ====================
    def _find_model(self):
        for path in DEFAULT_MODEL_PATHS:
            abs_path = os.path.abspath(os.path.join(PROJECT_ROOT, path)) if not os.path.isabs(path) else path
            if os.path.exists(abs_path):
                return abs_path
        # 直接搜寻 models 目录
        if os.path.exists(MODELS_DIR):
            pts = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')])
            if pts:
                return os.path.join(MODELS_DIR, pts[-1])
        return None

    # ==================== 轨迹颜色 ====================
    def get_trail_color(self, track_id):
        """取得该 track_id 的轨迹颜色（缓存）"""
        if not self.colorful_trail:
            return TRAIL_COLOR
        if track_id not in self.trail_color_cache:
            self.trail_color_cache[track_id] = generate_trail_color(track_id)
        return self.trail_color_cache[track_id]

    def reset(self):
        self.trails.clear()
        self.trails_smooth.clear()
        self.lost_counter.clear()
        self.angle_history.clear()
        self.angle_diffs.clear()
        self.analysis_records.clear()
        self.lane_change_results.clear()
        self.trail_color_cache.clear()
        self._count_history.clear()
        self.zone_intrusions.clear()
        self.zone_inside_ids.clear()
        self._snapshot_logged.clear()
        self.frame_id = 0

    # ==================== 监控区间 ====================
    def in_valid_y_zone(self, y):
        return self.valid_y_min <= y <= self.valid_y_max

    def get_perspective_scale(self, cy):
        if not self.perspective_enabled:
            return 1.0
        ratio = (cy - self.valid_y_min) / max(self.valid_y_max - self.valid_y_min, 1)
        ratio = max(0.0, min(1.0, ratio))
        return 0.7 + 0.3 * ratio

    # ==================== Part C: 指数平滑 ====================
    def exponential_smooth_point(self, prev, curr, alpha):
        if prev is None:
            return curr
        return (alpha * curr[0] + (1 - alpha) * prev[0],
                alpha * curr[1] + (1 - alpha) * prev[1])

    def update_smooth_trail(self, track_id, center):
        if track_id not in self.trails_smooth:
            self.trails_smooth[track_id] = [center]
            return center
        prev = self.trails_smooth[track_id][-1]
        sc = self.exponential_smooth_point(prev, center, self.smooth_alpha)
        self.trails_smooth[track_id].append(sc)
        if len(self.trails_smooth[track_id]) > self.trail_length:
            self.trails_smooth[track_id] = self.trails_smooth[track_id][-self.trail_length:]
        return sc

    # ==================== 角度分析（期中相容） ====================
    def get_angle_diff_deg(self, a1, a2):
        d = a2 - a1
        if d > 180:
            d -= 360
        elif d < -180:
            d += 360
        return d

    def judge_lane_change_by_window(self, diffs, ws, acc_th, ratio, scale=1.0):
        if len(diffs) < ws:
            return False, None
        w = diffs[-ws:]
        pos = [d for d in w if d > 0]
        neg = [d for d in w if d < 0]
        if sum(pos) >= acc_th * scale and len(pos) / len(w) >= ratio:
            return True, 'right'
        if sum(abs(d) for d in neg) >= acc_th * scale and len(neg) / len(w) >= ratio:
            return True, 'left'
        return False, None

    def judge_lane_change_by_triple_window(self, diffs, scale=1.0):
        for ws, acc, r, name in [
            (self.short_window_size, self.short_acc_threshold, self.short_consistent_ratio, 'short'),
            (self.long_window_size, self.long_acc_threshold, self.long_consistent_ratio, 'long'),
            (self.long2_window_size, self.long2_acc_threshold, self.long2_consistent_ratio, 'long2'),
        ]:
            ok, _ = self.judge_lane_change_by_window(diffs, ws, acc, r, scale)
            if ok:
                return True, name
        return False, None

    # ==================== 净角度变化检查（期中完整版） ====================
    def judge_net_angle_change(self, diffs, ws, net_th, scale=1.0):
        if len(diffs) < ws:
            return False
        w = diffs[-ws:]
        pos = sum(d for d in w if d > 0)
        neg = sum(abs(d) for d in w if d < 0)
        return abs(pos - neg) >= net_th * scale

    def judge_net_angle_change_by_triple_window(self, diffs, trigger_type, scale=1.0):
        if trigger_type == 'short':
            return self.judge_net_angle_change(diffs, self.short_window_size, self.short_net_threshold, scale)
        if trigger_type == 'long':
            return self.judge_net_angle_change(diffs, self.long_window_size, self.long_net_threshold, scale)
        if trigger_type == 'long2':
            return self.judge_net_angle_change(diffs, self.long2_window_size, self.long2_net_threshold, scale)
        return False

    # ==================== 横向位移确认（期中完整版） ====================
    def judge_lateral_shift_by_trail(self, trail, scale=1.0):
        if len(trail) < self.trajectory_window_size:
            return False
        recent = trail[-self.trajectory_window_size:]
        sx, sy = recent[0]
        ex, ey = recent[-1]
        dx, dy = ex - sx, ey - sy

        if self.lateral_axis == 'y':
            lateral_disp, forward_disp = dy, dx
            axis_diffs = [recent[i][1] - recent[i-1][1] for i in range(1, len(recent))]
        else:
            lateral_disp, forward_disp = dx, dy
            axis_diffs = [recent[i][0] - recent[i-1][0] for i in range(1, len(recent))]

        scaled_pixel = self.min_lateral_shift * scale
        scaled_ratio = self.min_lateral_ratio * scale

        if abs(lateral_disp) < scaled_pixel:
            return False
        if abs(lateral_disp) / (abs(forward_disp) + 1e-6) < scaled_ratio:
            return False
        if len(axis_diffs) == 0:
            return False

        pc = sum(1 for d in axis_diffs if d > 0)
        nc = sum(1 for d in axis_diffs if d < 0)
        cr = pc / len(axis_diffs) if lateral_disp > 0 else nc / len(axis_diffs) if lateral_disp < 0 else 0
        return cr >= self.min_x_consistent_ratio

    def get_motion_vector(self, p1, p2):
        return (p2[0] - p1[0], p2[1] - p1[1])

    def get_vector_angle_deg(self, vec):
        vx, vy = vec
        if vx == 0 and vy == 0:
            return None
        return math.degrees(math.atan2(vy, vx))

    # ==================== 核心追踪 ====================
    def track(self, frame, draw_trail=False, analyze_lane_change=False):
        """
        对一帧图像进行追踪
        Args:
            frame: 输入图像
            draw_trail: 是否画轨迹线
            analyze_lane_change: 是否进行变道分析
        Returns:
            annotated_frame, pred_boxes
        """
        self.frame_id += 1

        # ---- FPS 计算 ----
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            self.fps_values.append(1.0 / dt)
            if len(self.fps_values) > 30:
                self.fps_values.pop(0)
            self.fps_smooth = sum(self.fps_values) / len(self.fps_values)

        # ---- YOLO 追踪 ----
        results = self.model.track(
            frame, persist=True, device=self.device,
            imgsz=self.img_size, conf=self.conf, iou=self.iou
        )

        pred_boxes = []
        current_ids = []

        if results[0].boxes and results[0].boxes.id is not None:
            for box in results[0].boxes:
                class_id = int(box.cls.cpu().item())
                lbl = self.model.names[class_id]
                if lbl not in OBJ_LIST:
                    continue

                xyxy = box.xyxy.cpu()[0].numpy()
                x1, y1, x2, y2 = xyxy
                track_id = int(box.id.cpu().item())
                current_ids.append(track_id)

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                center = (cx, cy)

                if draw_trail or analyze_lane_change:
                    if track_id in self.trails:
                        self.trails[track_id].append(center)
                    else:
                        self.trails[track_id] = [center]
                    self.lost_counter[track_id] = 0
                    if len(self.trails[track_id]) > self.trail_length:
                        self.trails[track_id] = self.trails[track_id][-self.trail_length:]
                    smooth = self.update_smooth_trail(track_id, center)
                else:
                    smooth = center

                # ---- 计数系统（跨线检测） ----
                if self.count_enabled and self.count_line:
                    self._update_counting(track_id, cx, cy, lbl)

                # ---- 变道分析（保留） ----
                if analyze_lane_change:
                    self._analyze_lane_change(frame, track_id, lbl, cx, cy,
                                              x1, y1, x2, y2, smooth)

                pred_boxes.append((x1, y1, x2, y2, lbl, track_id))

        # ---- 清理消失目标 ----
        if draw_trail or analyze_lane_change:
            remove_ids = []
            for tid in list(self.trails.keys()):
                if tid not in current_ids:
                    self.lost_counter[tid] = self.lost_counter.get(tid, 0) + 1
                    if self.lost_counter[tid] > self.lost_threshold:
                        remove_ids.append(tid)
            for tid in remove_ids:
                for d in [self.trails, self.trails_smooth, self.lost_counter,
                          self.angle_history, self.angle_diffs,
                          self.lane_change_results]:
                    d.pop(tid, None)
                self.trail_color_cache.pop(tid, None)
                self._count_history.pop(tid, None)
                self.zone_intrusions.pop(tid, None)

        # ---- 绘制 ----
        annotated = self._draw_boxes(frame.copy(), pred_boxes)
        if draw_trail:
            annotated = self._draw_trails(annotated)
        if analyze_lane_change:
            annotated = self._draw_lane_change_status(annotated, pred_boxes)

        return annotated, pred_boxes

    def _update_counting(self, track_id, cx, cy, lbl):
        """跨线计数逻辑"""
        (lx, ly), (rx, ry) = self.count_line
        # 判断在线的哪一侧
        # 水平线简化：用 y 是否大于 line_y
        line_y = ly  # 水平参考线
        side = 'below' if cy > line_y else 'above'

        if track_id not in self._count_history:
            self._count_history[track_id] = {'last_side': side, 'counted': False}

        hist = self._count_history[track_id]
        # 当从 above → below 跨越：out（往下）
        # 当从 below → above 跨越：in（往上）
        if hist['last_side'] == 'above' and side == 'below' and not hist['counted']:
            direction = 'out'
            lbl_key = lbl if lbl in self.count_data else 'car'
            self.count_data[lbl_key][direction] += 1
            hist['counted'] = True
            print(f"[计数] {lbl}(ID:{track_id}) {direction} → "
                  f"{lbl_key}={self.count_data[lbl_key]}")
        elif hist['last_side'] == 'below' and side == 'above' and not hist['counted']:
            direction = 'in'
            lbl_key = lbl if lbl in self.count_data else 'car'
            self.count_data[lbl_key][direction] += 1
            hist['counted'] = True
            print(f"[计数] {lbl}(ID:{track_id}) {direction} → "
                  f"{lbl_key}={self.count_data[lbl_key]}")
        elif hist['last_side'] != side:
            # 方向改变但已计数过，重设标记
            hist['counted'] = False

        hist['last_side'] = side

    def _analyze_lane_change(self, frame, track_id, lbl, cx, cy,
                              x1, y1, x2, y2, smooth_center):
        """变道分析 — 三关验证（角度 + 净角度 + 横向位移，参照期中完整版）"""
        if not self.in_valid_y_zone(cy):
            return

        trail_use = self.trails_smooth.get(track_id, self.trails.get(track_id, []))
        if len(trail_use) < self.angle_step + 1:
            return

        p1 = trail_use[-1 - self.angle_step]
        p2 = trail_use[-1]
        vx, vy = self.get_motion_vector(p1, p2)
        motion_mag = math.sqrt(vx**2 + vy**2)
        if motion_mag < self.min_speed:
            return

        angle = self.get_vector_angle_deg((vx, vy))
        if angle is None:
            return

        if track_id not in self.angle_history:
            self.angle_history[track_id] = []
        self.angle_history[track_id].append(angle)

        if track_id not in self.angle_diffs:
            self.angle_diffs[track_id] = []
        if len(self.angle_history[track_id]) > 1:
            ad = self.get_angle_diff_deg(self.angle_history[track_id][-2], angle)
            if abs(ad) <= self.abnormal_angle_diff_threshold:
                self.angle_diffs[track_id].append(ad)

        p_scale = self.get_perspective_scale(cy)

        # 第一关：三窗口角度检查
        angle_ok, trigger_type = self.judge_lane_change_by_triple_window(
            self.angle_diffs.get(track_id, []), p_scale)

        if angle_ok and trigger_type:
            # 第二关：净角度变化（过滤振荡）
            net_angle_ok = self.judge_net_angle_change_by_triple_window(
                self.angle_diffs.get(track_id, []), trigger_type, p_scale)

            # 第三关：横向位移确认
            lateral_ok = self.judge_lateral_shift_by_trail(trail_use, p_scale)

            # 三关全过 → 确认为变道
            if net_angle_ok and lateral_ok:
                if track_id not in self.lane_change_results:
                    self.lane_change_results[track_id] = True

    # ==================== 绘制方法 ====================
    def _draw_boxes(self, im, boxes):
        for x1, y1, x2, y2, lbl, track_id in boxes:
            color = (0, 0, 255) if self.lane_change_results.get(track_id, False) \
                    else COLORS.get(lbl, (128, 128, 128))
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            tag = f"{lbl}({track_id})"
            if self.lane_change_results.get(track_id, False):
                tag += " LC"
            cv2.putText(im, tag, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return im

    def _draw_trails(self, im):
        """
        绘制轨迹线
        - 如果 self.colorful_trail = True → 每 ID 不同颜色
        - 否则使用统一 TRAIL_COLOR
        """
        for tid, trail in self.trails_smooth.items():
            if len(trail) < 2:
                continue
            color = self.get_trail_color(tid) if self.colorful_trail else TRAIL_COLOR
            for i in range(1, len(trail)):
                cv2.line(im,
                         (int(trail[i-1][0]), int(trail[i-1][1])),
                         (int(trail[i][0]), int(trail[i][1])),
                         color, 2)
        return im

    def _draw_lane_change_status(self, im, boxes):
        for x1, y1, x2, y2, lbl, track_id in boxes:
            if self.lane_change_results.get(track_id, False):
                cv2.putText(im, "LANE CHANGE", (int(x1), int(y1) - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return im

    # ==================== 计数绘制 ====================
    def draw_count_line(self, frame, line_color=(0, 255, 255), thickness=2):
        """绘制计数参考线"""
        if self.count_line is None:
            return frame
        (lx, ly), (rx, ry) = self.count_line
        cv2.line(frame, (int(lx), int(ly)), (int(rx), int(ry)), line_color, thickness)
        # 标注方向
        mid_x, mid_y = (lx + rx) // 2, (ly + ry) // 2
        cv2.putText(frame, "IN ^", (int(mid_x) - 50, int(mid_y) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)
        cv2.putText(frame, "OUT v", (int(mid_x) + 10, int(mid_y) + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)
        return frame

    # ==================== 敏感区域绘制 ====================
    def set_zone_polygon(self, polygon):
        """设置敏感区域多边形"""
        self.zone_polygon = polygon
        self.zone_enabled = (polygon is not None)

    def draw_zone(self, frame):
        """绘制敏感区域（半透明）"""
        if self.zone_polygon is None:
            return frame
        frame = drawAndFillPolygon(frame, self.zone_polygon, self.zone_color, self.zone_alpha)
        # 区域标签
        cx = int(sum(p[0] for p in self.zone_polygon) / len(self.zone_polygon))
        cy = int(sum(p[1] for p in self.zone_polygon) / len(self.zone_polygon))
        cv2.putText(frame, "SENSITIVE ZONE", (cx - 70, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.zone_color, 2)
        return frame

    def check_zone_intrusions(self, pred_boxes, frame_id, fps):
        """
        检测敏感区域入侵
        Returns: set — 当前在区域内的 track_id 集合
        """
        if self.zone_polygon is None:
            return set()

        current_inside = set()

        for box in pred_boxes:
            x1, y1, x2, y2, lbl, track_id = box
            # 使用边界框中心点判定，中心进入区域才算入侵(>1/2车身)
            center = Point(x=(x1+x2)/2, y=(y1+y2)/2)
            if isInsidePolygon(center, self.zone_polygon):
                current_inside.add(track_id)
                just_entered = (track_id not in self.zone_inside_ids)

                if track_id not in self.zone_intrusions:
                    self.zone_intrusions[track_id] = {
                        'entered_frame': frame_id,
                        'total_frames_inside': 0,
                        'alerted': False,
                        'label': lbl,
                    }

                state = self.zone_intrusions[track_id]
                state['total_frames_inside'] += 1
                state['label'] = lbl

                if just_entered:
                    print(f"[区域] {lbl}(ID:{track_id}) 进入敏感区域 (Frame {frame_id})")

        # 更新上一帧状态
        self.zone_inside_ids = current_inside

        return current_inside

    # ==================== 异常截图 ====================
    def capture_snapshot(self, frame, track_id, x1, y1, x2, y2, lbl):
        """进入区域时对目标裁切截图，保存为 ID_时间.jpg（每 ID 仅存一次）"""
        if not self.snapshot_enabled:
            return None
        if track_id in self._snapshot_logged:
            return None

        # 边界保护
        h, w = frame.shape[:2]
        x1_i, y1_i = max(0, int(x1)), max(0, int(y1))
        x2_i, y2_i = min(w, int(x2)), min(h, int(y2))
        if x2_i <= x1_i or y2_i <= y1_i:
            return None

        crop = frame[y1_i:y2_i, x1_i:x2_i]
        if crop.size == 0:
            return None

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{track_id}_{ts}.jpg"
        save_path = os.path.join(self.snapshot_dir, filename)
        cv2.imwrite(save_path, crop)
        self._snapshot_logged.add(track_id)
        print(f"[截图] {lbl}(ID:{track_id}) 已保存 → {save_path}")
        return save_path

    # ==================== 工具 ====================
    def draw_fps(self, frame, x=10, y=25):
        """绘制 FPS"""
        color = (0, 255, 0) if self.fps_smooth > 20 else (0, 255, 255) if self.fps_smooth > 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {self.fps_smooth:.1f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def get_count_summary(self):
        """取得计数摘要文字"""
        lines = []
        for cls_name in ['car', 'bus', 'truck', 'person']:
            d = self.count_data.get(cls_name, {'in': 0, 'out': 0})
            total = d['in'] + d['out']
            if total > 0:
                lines.append(f"{cls_name}: {d['in']}↑ {d['out']}↓ = {total}")
        return '\n'.join(lines) if lines else "无数据"

    def export_csv(self, csv_path="analysis_records.csv"):
        import csv
        if not self.analysis_records:
            print("没有分析数据可导出")
            return
        if not os.path.isabs(csv_path) and not csv_path.startswith('.'):
            csv_path = os.path.join(OUTPUT_DIR, csv_path)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        fieldnames = ['frame_id', 'track_id', 'center_x', 'center_y',
                      'smooth_x', 'smooth_y', 'vx', 'vy', 'angle',
                      'angle_diff', 'angle_ok', 'net_angle_ok', 'lateral_ok']
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.analysis_records)
        print(f"分析数据已导出到: {csv_path} ({len(self.analysis_records)} 条)")
