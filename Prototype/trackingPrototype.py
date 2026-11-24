import os
import csv
import time
import cv2
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# 60 FPS WEBCAM & HARDWARE CONSTANTS
USE_YOLO = True
TRACK_PEOPLE = True                   # Enabled: Tracks humans (Class 0)
TRACK_RED_WHITE_TARGET = True         # Enabled: Tracks traditional red-and-white bullseye targets (Class 99)
SIM_FPS = 60.0                        # Hardware control loop rate (60 Hz)
FRAME_BUDGET_MS = 1000.0 / SIM_FPS    # 16.67 ms maximum allowed latency per frame
LOOKAHEAD_FRAMES = 30                 # 30 frames = 0.50 seconds of predicted path at 60 FPS
LEAD_INTERCEPT_FRAMES = 15            # 15 frames = 250 ms lead aim point at 60 FPS

# Class IDs: 0=Person, 99=Red/White Target
CLASS_NAMES = {
    0: "PERSON",
    99: "TARGET"
}

# PART 1: 60Hz KALMAN FILTER & PREDICTOR
class TargetTrackerKF:
    def __init__(self, initial_x, initial_y):
        self.kf = cv2.KalmanFilter(4, 2)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        self.kf.processNoiseCov = np.diag([0.1, 0.1, 2.0, 2.0]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.05
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self.kf.statePost = np.array([
            [np.float32(initial_x)],
            [np.float32(initial_y)],
            [0.0],
            [0.0]
        ], np.float32)

        self.prev_pos = (initial_x, initial_y)
        self.prev_vel = (0.0, 0.0)
        self.accel = (0.0, 0.0)
        self.updates_count = 0

    def predict(self):
        pred = self.kf.predict()
        self.accel = (self.accel[0] * 0.90, self.accel[1] * 0.90)
        return float(pred[0][0]), float(pred[1][0])

    def update(self, x, y):
        self.updates_count += 1
        if self.updates_count == 1:
            vx = np.float32(x - self.prev_pos[0])
            vy = np.float32(y - self.prev_pos[1])
            self.kf.statePre = np.array([[np.float32(x)], [np.float32(y)], [vx], [vy]], np.float32)
            self.kf.statePost = np.array([[np.float32(x)], [np.float32(y)], [vx], [vy]], np.float32)
            self.prev_vel = (float(vx), float(vy))
        else:
            measurement = np.array([[np.float32(x)], [np.float32(y)]], np.float32)
            self.kf.correct(measurement)
            
            vx = float(self.kf.statePost[2][0])
            vy = float(self.kf.statePost[3][0])
            
            raw_ax = np.clip(vx - self.prev_vel[0], -4.0, 4.0)
            raw_ay = np.clip(vy - self.prev_vel[1], -4.0, 4.0)
            self.accel = (
                0.82 * self.accel[0] + 0.18 * raw_ax,
                0.82 * self.accel[1] + 0.18 * raw_ay
            )
            self.prev_vel = (vx, vy)
            
        self.prev_pos = (float(self.kf.statePost[0][0]), float(self.kf.statePost[1][0]))

    def get_velocity_px_per_sec(self):
        vx_fps = float(self.kf.statePost[2][0]) * SIM_FPS
        vy_fps = float(self.kf.statePost[3][0]) * SIM_FPS
        return vx_fps, vy_fps

    def get_future_path(self, steps=LOOKAHEAD_FRAMES):
        cx = float(self.kf.statePost[0][0])
        cy = float(self.kf.statePost[1][0])
        vx = float(self.kf.statePost[2][0])
        vy = float(self.kf.statePost[3][0])
        ax, ay = self.accel

        future_pts = []
        for k in range(1, steps + 1):
            fx = cx + (vx * k) + (0.5 * ax * (k ** 2))
            fy = cy + (vy * k) + (0.5 * ay * (k ** 2))
            future_pts.append((int(fx), int(fy)))
        return future_pts


# PART 2: SINGLE TARGET TRACK OBJECT
class Track:
    def __init__(self, track_id, center, bbox, embedding, cls_id, conf):
        self.track_id = track_id
        self.kf = TargetTrackerKF(center[0], center[1])
        self.prediction = center
        self.bbox = list(bbox)
        self.embedding = embedding
        self.cls_id = cls_id
        self.conf = conf
        self.skipped_frames = 0
        self.hits = 1
        self.is_coasting = False
        self.future_path = []


# PART 3: HYBRID MULTI-TARGET WEBCAM TRACKER
class UnifiedPrototypePipeline:
    def __init__(self, use_yolo=USE_YOLO, yolo_weights="yolov8n.pt"):
        self.use_yolo = use_yolo
        self.yolo = None
        if self.use_yolo:
            from ultralytics import YOLO
            print("Loading YOLO model...")
            self.yolo = YOLO(yolo_weights, task='detect')
            self.yolo.to('cuda')

        if TRACK_PEOPLE:
            self.target_classes = [0, 32, 39, 41, 67]
        else:
            self.target_classes = [32, 39, 41, 67]

        # fuse red rings of a bullseye into one solid target
        self.target_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        self.tracks = []
        self.next_id = 1
        
        self.max_pixel_dist = 220.0
        self.crossover_proximity = 100.0
        self.max_skipped_frames = 15
        self.lambda_weight = 0.7
        self.box_smooth_alpha = 0.60

    def extract_embedding(self, frame, bbox):
        h, w, _ = frame.shape
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return np.zeros(128, dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
        return hist / (np.linalg.norm(hist) + 1e-6)

    def _detect_red_white_targets(self, frame):
        """Detects traditional red-and-white concentric/bullseye targets in <1.5 ms."""
        target_boxes = []
        target_centers = []
        target_confs = []

        # Downscale by 50% for fast HSV processing
        scale = 0.5
        inv_scale = 2.0
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Red HSV masks
        red_low1 = np.array([0, 105, 70], dtype=np.uint8)
        red_high1 = np.array([12, 255, 255], dtype=np.uint8)
        red_low2 = np.array([165, 105, 70], dtype=np.uint8)
        red_high2 = np.array([180, 255, 255], dtype=np.uint8)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, red_low1, red_high1),
            cv2.inRange(hsv, red_low2, red_high2)
        )

        # White HSV mask
        white_low = np.array([0, 0, 155], dtype=np.uint8)
        white_high = np.array([180, 80, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, white_low, white_high)

        # Fuse red rings together so the outer ring forms one unified contour
        red_closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self.target_kernel, iterations=2)
        contours, _ = cv2.findContours(red_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_s, w_s = small.shape[:2]
        raw_candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 140 or area > (h_s * w_s * 0.60):
                continue

            x, y, w_box, h_box = cv2.boundingRect(cnt)
            aspect_ratio = float(w_box) / max(1.0, float(h_box))
            # Bullseye / square targets have an aspect ratio near 1.0
            if not (0.55 <= aspect_ratio <= 1.80):
                continue

            # Check convex hull solidity (eliminates irregular background clutter)
            hull = cv2.convexHull(cnt)
            hull_area = max(1.0, cv2.contourArea(hull))
            solidity = area / hull_area
            if solidity < 0.60:
                continue

            # Verify presence of BOTH red and white pixels inside the target region
            box_area = float(w_box * h_box)
            red_pixels = cv2.countNonZero(red_mask[y:y + h_box, x:x + w_box])
            white_pixels = cv2.countNonZero(white_mask[y:y + h_box, x:x + w_box])

            red_ratio = red_pixels / box_area
            white_ratio = white_pixels / box_area

            if red_ratio >= 0.15 and white_ratio >= 0.12:
                conf_score = min(0.98, 0.55 + (red_ratio + white_ratio) * 0.45)
                fx1, fy1 = float(x * inv_scale), float(y * inv_scale)
                fx2, fy2 = float((x + w_box) * inv_scale), float((y + h_box) * inv_scale)
                raw_candidates.append(([fx1, fy1, fx2, fy2], conf_score))

        # Suppress inner rings if an outer target ring already encloses them
        for i, (box_a, conf_a) in enumerate(raw_candidates):
            is_inner_ring = False
            for j, (box_b, _) in enumerate(raw_candidates):
                if i != j:
                    if box_a[0] >= box_b[0] - 10 and box_a[1] >= box_b[1] - 10 and \
                       box_a[2] <= box_b[2] + 10 and box_a[3] <= box_b[3] + 10:
                        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
                        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
                        if area_a < area_b:
                            is_inner_ring = True
                            break
            if not is_inner_ring:
                target_boxes.append(box_a)
                target_centers.append(((box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0))
                target_confs.append(conf_a)

        return target_boxes, target_centers, target_confs

    def _detect_targets(self, frame):
        bboxes = []
        centers = []
        cls_ids = []
        confs = []
        
        # YOLO Detection (Humans + Handheld Objects)
        # imgsz=480 improves small-object & human detection accuracy while keeping latency low
        if self.use_yolo and self.yolo is not None:
            results = self.yolo(
                frame,
                imgsz=480,
                verbose=False,
                conf=0.40,
                iou=0.45,
                classes=self.target_classes
            )[0]
            
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                c_id = int(box.cls[0].item())
                conf_val = float(box.conf[0].item())
                bboxes.append([float(x1), float(y1), float(x2), float(y2)])
                centers.append(((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0))
                cls_ids.append(c_id)
                confs.append(conf_val)

        # Dedicated Red-and-White Target Detector (Class 99)
        if TRACK_RED_WHITE_TARGET:
            t_boxes, t_centers, t_confs = self._detect_red_white_targets(frame)
            for tb, tc, tcf in zip(t_boxes, t_centers, t_confs):
                bboxes.append(tb)
                centers.append(tc)
                cls_ids.append(99)
                confs.append(tcf)
                
        return bboxes, centers, cls_ids, confs

    def process_frame(self, frame):
        # PREDICT
        predictions = []
        for t in self.tracks:
            pred_x, pred_y = t.kf.predict()
            dx = pred_x - t.prediction[0]
            dy = pred_y - t.prediction[1]
            t.bbox = [t.bbox[0] + dx, t.bbox[1] + dy, t.bbox[2] + dx, t.bbox[3] + dy]
            t.prediction = (pred_x, pred_y)
            t.is_coasting = True
            t.future_path = t.kf.get_future_path(steps=LOOKAHEAD_FRAMES)
            predictions.append(t.prediction)

        # DETECT
        bboxes, centers, cls_ids, confs = self._detect_targets(frame)

        if len(self.tracks) == 0:
            for bbox, center, cid, cf in zip(bboxes, centers, cls_ids, confs):
                emb = self.extract_embedding(frame, bbox)
                self._spawn_track(center, bbox, emb, cid, cf)
            return self.tracks

        if len(bboxes) == 0:
            for t in self.tracks:
                t.skipped_frames += 1
            self._purge_stale_tracks()
            return self.tracks

        # COST MATRIX 
        motion_cost = distance.cdist(np.array(predictions), np.array(centers), metric='euclidean')

        need_reid = False
        if len(predictions) >= 2:
            if np.any(distance.pdist(np.array(predictions)) < self.crossover_proximity):
                need_reid = True

        det_embeddings = [self.extract_embedding(frame, b) for b in bboxes] if need_reid else None

        if need_reid:
            track_embeddings = [t.embedding for t in self.tracks]
            norm_motion = np.clip(motion_cost / self.max_pixel_dist, 0.0, 1.0)
            app_cost = distance.cdist(np.array(track_embeddings), np.array(det_embeddings), metric='cosine')
            total_cost = (self.lambda_weight * norm_motion) + ((1.0 - self.lambda_weight) * app_cost)
        else:
            total_cost = motion_cost.copy()

        for r, track in enumerate(self.tracks):
            for c, det_cls in enumerate(cls_ids):
                if track.cls_id != det_cls:
                    total_cost[r, c] += 10000.0
                    motion_cost[r, c] += 10000.0

        # HUNGARIAN ALGORITHM ASSIGNMENT
        row_inds, col_inds = linear_sum_assignment(total_cost)

        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(bboxes)))

        for r, c in zip(row_inds, col_inds):
            if motion_cost[r, c] < self.max_pixel_dist:
                track = self.tracks[r]
                track.kf.update(centers[c][0], centers[c][1])
                
                smooth_cx, smooth_cy = track.kf.prev_pos
                track.prediction = (smooth_cx, smooth_cy)
                track.future_path = track.kf.get_future_path(steps=LOOKAHEAD_FRAMES)
                track.conf = confs[c]
                
                old_w = track.bbox[2] - track.bbox[0]
                old_h = track.bbox[3] - track.bbox[1]
                new_w = bboxes[c][2] - bboxes[c][0]
                new_h = bboxes[c][3] - bboxes[c][1]
                w_s = self.box_smooth_alpha * new_w + (1.0 - self.box_smooth_alpha) * old_w
                h_s = self.box_smooth_alpha * new_h + (1.0 - self.box_smooth_alpha) * old_h
                
                track.bbox = [smooth_cx - w_s / 2.0, smooth_cy - h_s / 2.0,
                              smooth_cx + w_s / 2.0, smooth_cy + h_s / 2.0]
                
                track.skipped_frames = 0
                track.hits += 1
                track.is_coasting = False
                if need_reid:
                    track.embedding = 0.85 * track.embedding + 0.15 * det_embeddings[c]
                
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)

        # LIFECYCLE CLEANUP
        for r in unmatched_tracks:
            self.tracks[r].skipped_frames += 1

        for c in unmatched_dets:
            emb = det_embeddings[c] if need_reid else self.extract_embedding(frame, bboxes[c])
            self._spawn_track(centers[c], bboxes[c], emb, cls_ids[c], confs[c])

        self._purge_stale_tracks()
        return self.tracks

    def _spawn_track(self, center, bbox, embedding, cls_id, conf):
        self.tracks.append(Track(self.next_id, center, bbox, embedding, cls_id, conf))
        self.next_id += 1

    def _purge_stale_tracks(self):
        self.tracks = [t for t in self.tracks if t.skipped_frames <= self.max_skipped_frames]


# PART 4: LIVE WEBCAM LOOP & TELEMETRY LOGGER
def main():
    output_video_path = "tracked_output_60fps.mp4"
    output_csv_path = "tracker_results_60fps.csv"

    print("Opening live webcam (Camera 0)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam (Camera 0). Check if another app is using it.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_video_path, fourcc, SIM_FPS, (width, height))

    pipeline = UnifiedPrototypePipeline()

    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "frame_idx", "sim_time_sec", "track_id", "class_name", "center_x", "center_y",
            "vx_px_per_s", "vy_px_per_s", "pred_lead_x", "pred_lead_y",
            "tracking_mode", "frame_latency_ms", "within_60fps_budget"
        ])

        frame_idx = 0
        print("Webcam tracking active! Detecting PERSON, TARGET (Red/White), and handheld objects. Press 'q' to stop.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            height, width = frame.shape[:2]

            start_time = time.perf_counter()
            active_tracks = pipeline.process_frame(frame)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            sim_time_sec = frame_idx / SIM_FPS
            within_budget = latency_ms <= FRAME_BUDGET_MS

            # Bottom HUD Banner
            banner_h = min(45, height)
            sub_slice = frame[height - banner_h:height, 0:width]
            dark_rect = np.full(sub_slice.shape, 25, dtype=np.uint8)
            frame[height - banner_h:height, 0:width] = cv2.addWeighted(dark_rect, 0.75, sub_slice, 0.25, 0)

            for track in active_tracks:
                x1, y1, x2, y2 = map(int, track.bbox)
                cx, cy = int(track.prediction[0]), int(track.prediction[1])
                vx_s, vy_s = track.kf.get_velocity_px_per_sec()
                cls_name = CLASS_NAMES.get(track.cls_id, f"CLS_{track.cls_id}")

                # 1. Draw PREDICTED FUTURE PATH (Cyan Dashed Trajectory + Lead Crosshair)
                lead_x, lead_y = cx, cy
                if len(track.future_path) > 0:
                    prev_pt = (cx, cy)
                    for step_idx, f_pt in enumerate(track.future_path):
                        if step_idx % 2 == 0:
                            cv2.line(frame, prev_pt, f_pt, (255, 220, 0), 2, cv2.LINE_AA)
                        if (step_idx + 1) % 5 == 0:
                            cv2.circle(frame, f_pt, 3, (255, 180, 0), -1, cv2.LINE_AA)
                        prev_pt = f_pt

                    lead_idx = min(LEAD_INTERCEPT_FRAMES - 1, len(track.future_path) - 1)
                    lead_x, lead_y = track.future_path[lead_idx]
                    cv2.circle(frame, (lead_x, lead_y), 8, (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.line(frame, (lead_x - 12, lead_y), (lead_x + 12, lead_y), (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.line(frame, (lead_x, lead_y - 12), (lead_x, lead_y + 12), (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, "LEAD (+250ms)", (lead_x + 10, lead_y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

                if track.is_coasting:
                    color = (0, 140, 255)  # Orange when coasting on Kalman prediction
                    mode_str = "KALMAN_BLUR_PREDICT"
                elif track.cls_id == 99:
                    color = (0, 255, 255)  # High-visibility Yellow/Gold lock for Red-White Bullseye Target
                    mode_str = "BULLSEYE_LOCKED"
                else:
                    color = (40, 210, 40)  # Green lock for Person / YOLO objects
                    mode_str = "TARGET_LOCKED"

                # 2. Draw current target bounding box and center aiming dot
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)

                label = f"ID:{track.track_id} {cls_name} ({track.conf:.0%}) [{mode_str}]"
                cv2.putText(frame, label, (max(5, x1), max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

                writer.writerow([
                    frame_idx, round(sim_time_sec, 4), track.track_id, cls_name,
                    round(cx, 2), round(cy, 2), round(vx_s, 1), round(vy_s, 1),
                    round(lead_x, 2), round(lead_y, 2), mode_str,
                    round(latency_ms, 2), within_budget
                ])

            # Bottom HUD Bar
            budget_status = "60Hz OK" if within_budget else "CPU MODE"
            hud = (
                f"Webcam | Frame {frame_idx:04d} | "
                f"Latency: {latency_ms:.1f} ms [{budget_status}] | "
                f"Cyan: Predicted Path | Magenta: Lead"
            )
            cv2.putText(frame, hud, (12, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            out_video.write(frame)
            cv2.imshow("Turret Vision Prototype - Live Webcam (Press Q to Quit)", frame)

            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    print(f"Done! Processed {frame_idx} frames.")
    print(f"Saved webcam video to '{output_video_path}' and telemetry to '{output_csv_path}'.")

if __name__ == "__main__":
    main()