# final_main.py
"""
期末报告 — 目标追踪与分类计数 + 敏感区域监控
"""

import cv2
import os

from yolo_tracker_base import (
    YOLOTracker, VIDEOS_DIR, OUTPUT_DIR, PROJECT_ROOT,
    adaptive_resize, CLS_NAMES
)

SAVE_VIDEO = True
OUTPUT_NAME = "final_result.mp4"
DISPLAY_WIDTH = 1280

COUNT_LINE_Y_RATIO = 0.60

ZONE_POLYGON_RATIO = [
    (0.15, 0.95),
    (0.85, 0.95),
    (0.75, 0.40),
    (0.25, 0.40),
]


def find_videos():
    dirs = [
        VIDEOS_DIR,
        os.path.join(PROJECT_ROOT, '..', 'Exp_Midterm', 'videos'),
        os.path.join(PROJECT_ROOT, '..', '..', 'YOLO26Tracking', 'videos'),
    ]
    found = []
    for d in dirs:
        abs_d = os.path.abspath(d)
        if os.path.exists(abs_d):
            for f in sorted(os.listdir(abs_d)):
                if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    found.append(os.path.join(abs_d, f))
    return found


def main():
    videos = find_videos()
    if not videos:
        print(f"未找到影片，请放入 {VIDEOS_DIR} 目录")
        return

    video_path = videos[0]
    print(f"使用影片: {video_path}")

    tracker = YOLOTracker()
    tracker.colorful_trail = True

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开影片")
        return

    ret, frame = cap.read()
    if not ret:
        print("无法读取影片帧")
        return
    frame_height, frame_width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"分辨率: {frame_width}x{frame_height}  FPS: {fps:.1f}")

    count_line_y = int(frame_height * COUNT_LINE_Y_RATIO)
    tracker.count_line = ((0, count_line_y), (frame_width, count_line_y))
    tracker.count_enabled = True

    zone_polygon = [
        (int(frame_width * rx), int(frame_height * ry))
        for rx, ry in ZONE_POLYGON_RATIO
    ]
    tracker.set_zone_polygon(zone_polygon)
    print(f"敏感区域顶点: {zone_polygon}")

    win_name = 'Final Project - Tracking + Counting + Zone'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, DISPLAY_WIDTH,
                     int(frame_height * (DISPLAY_WIDTH / frame_width)))
    cv2.moveWindow(win_name, 100, 50)

    out = None
    if SAVE_VIDEO:
        out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (frame_width, frame_height))
        print(f"输出影片 → {out_path}")

    print("\n按下 ESC 或关闭视窗退出\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated, pred_boxes = tracker.track(frame, draw_trail=True)

        annotated = tracker.draw_zone(annotated)
        current_inside = tracker.check_zone_intrusions(pred_boxes,
                                                       tracker.frame_id, fps)
        if current_inside:
            for box in pred_boxes:
                x1, y1, x2, y2, lbl, track_id = box
                if track_id in current_inside:
                    tracker.capture_snapshot(annotated, track_id, x1, y1, x2, y2, lbl)
                    cv2.rectangle(annotated,
                                  (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 0, 255), 3)
                    cv2.putText(annotated, f"INTRUSION! {lbl}({track_id})",
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        annotated = tracker.draw_count_line(annotated)
        annotated = tracker.draw_fps(annotated)

        stats_x = frame_width - 260
        stats_y = 10
        overlay = annotated.copy()
        cv2.rectangle(overlay, (stats_x, stats_y),
                      (stats_x + 250, stats_y + 140), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0)
        cv2.putText(annotated, "=== Vehicle Count ===",
                    (stats_x + 10, stats_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        yy = stats_y + 50
        for cls_name in ['car', 'bus', 'truck', 'person']:
            d = tracker.count_data.get(cls_name, {'in': 0, 'out': 0})
            color = (255, 0, 0) if cls_name == 'car' else \
                    (0, 0, 255) if cls_name == 'bus' else \
                    (0, 255, 255) if cls_name == 'truck' else (0, 255, 0)
            text = f"{CLS_NAMES.get(cls_name, cls_name)}: ↑{d['in']}  ↓{d['out']}"
            cv2.putText(annotated, text, (stats_x + 10, yy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            yy += 28

        display = adaptive_resize(annotated, DISPLAY_WIDTH)
        cv2.imshow(win_name, display)

        if out:
            out.write(annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # ---- 结束 ----
    print(f"\n{'='*50}")
    print("最终统计摘要")
    print(f"{'='*50}")
    print(tracker.get_count_summary())
    if tracker.zone_intrusions:
        print(f"\n敏感区域入侵: {len(tracker.zone_intrusions)} 个目标")
        for tid, st in tracker.zone_intrusions.items():
            print(f"  ID:{tid} ({st['label']}) - "
                  f"累计 {st['total_frames_inside']} 帧")
    print(f"{'='*50}")

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("期末报告程序完成")


if __name__ == "__main__":
    main()
