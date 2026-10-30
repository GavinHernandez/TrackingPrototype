import os
import csv
import time
import cv2
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# 60 FPS SIMULATION CONSTANTS
USE_YOLO = False
SIM_FPS = 60.0                        # hardware control loop rate (60 Hz)
FRAME_BUDGET_MS = 1000.0 / SIM_FPS    # 16.67 ms maximum allowed latency per frame
LOOKAHEAD_FRAMES = 30                 # 30 frames = 0.50 seconds of predicted path at 60 FPS
LEAD_INTERCEPT_FRAMES = 15            # 15 frames = 250 ms lead aim point at 60 FPS


# 60Hz KALMAN FILTER AND FUTURE PREDICTOR
class TargetTrackerKF:
    def __init__(self, initial_x, initial_y):
        # 4 state variables: x, y, vx_per_frame, vy_per_frame and 2 measurements: x, y
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
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.02
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self.kf.statePost = np.array([
            [np.float32(initial_x)],
            [np.float32(initial_y)],
            [0.0],
            [0.0]
        ], np.float32)

        self.prev_pos = (initial_x, initial_y)
        self.prev_vel = (0.0, 0.0)
        self.accel = (0.0, 0.0)  # Smoothed (ax, ay) per frame^2 for curved trajectories
        self.updates_count = 0

    def predict(self):
        pred = self.kf.predict()
        self.accel = (self.accel[0] * 0.92, self.accel[1] * 0.92)
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
            
        self.prev_pos = (x, y)

    def get_velocity_px_per_sec(self):
        """Returns target velocity scaled to 60 FPS (pixels/second) for hardware control."""
        vx_fps = float(self.kf.statePost[2][0]) * SIM_FPS
        vy_fps = float(self.kf.statePost[3][0]) * SIM_FPS
        return vx_fps, vy_fps

    def get_future_path(self, steps=LOOKAHEAD_FRAMES):
        """Projects future (x, y) coordinates over the next N frames at 60 FPS."""
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


# SINGLE TARGET TRACK OBJECT
class Track:
    def __init__(self, track_id, center, bbox, embedding):
        self.track_id = track_id
        self.kf = TargetTrackerKF(center[0], center[1])
        self.prediction = center
        self.bbox = list(bbox)
        self.embedding = embedding
        self.skipped_frames = 0
        self.hits = 1
        self.is_coasting = False
        self.future_path = []  # Only stores future predicted path (past trail removed)


# LOW-LATENCY MULTI-TARGET TRACKER
class UnifiedPrototypePipeline:
    def __init__(self, use_yolo=USE_YOLO, yolo_weights="yolov8n.pt"):
        self.use_yolo = use_yolo
        self.yolo = None
        if self.use_yolo:
            from ultralytics import YOLO
            print("Loading YOLO model (imgsz=320 for 60 FPS inference)...")
            self.yolo = YOLO(yolo_weights)
        else:
            print(f"Running in 60 FPS Hardware Simulation Mode (Budget: {FRAME_BUDGET_MS:.2f} ms/frame)...")

        self.detect_scale = 0.25
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=40, varThreshold=20, detectShadows=False
        )
        self.prev_small_gray = None
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

        self.tracks = []
        self.next_id = 1
        
        self.max_pixel_dist = 350.0
        self.crossover_proximity = 120.0
        self.max_skipped_frames = 15
        self.lambda_weight = 0.7
        self.box_smooth_alpha = 0.65

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

    def _merge_nearby_boxes(self, bboxes, merge_dist=140.0):
        if len(bboxes) <= 1:
            return bboxes
            
        merged = [list(b) for b in bboxes]
        changed = True
        while changed:
            changed = False
            new_merged = []
            used = [False] * len(merged)
            for i in range(len(merged)):
                if used[i]:
                    continue
                bx1, by1, bx2, by2 = merged[i]
                cx1, cy1 = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                for j in range(i + 1, len(merged)):
                    if used[j]:
                        continue
                    jx1, jy1, jx2, jy2 = merged[j]
                    cx2, cy2 = (jx1 + jx2) / 2.0, (jy1 + jy2) / 2.0
                    overlap = not (bx2 < jx1 - 30 or bx1 > jx2 + 30 or by2 < jy1 - 30 or by1 > jy2 + 30)
                    if overlap or np.hypot(cx1 - cx2, cy1 - cy2) < merge_dist:
                        bx1, by1 = min(bx1, jx1), min(by1, jy1)
                        bx2, by2 = max(bx2, jx2), max(by2, jy2)
                        cx1, cy1 = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                        used[j] = True
                        changed = True
                new_merged.append([bx1, by1, bx2, by2])
            merged = new_merged
        return merged

    def _detect_targets(self, frame):
        bboxes = []
        h_full, w_full = frame.shape[:2]

        if self.use_yolo and self.yolo is not None:
            results = self.yolo(frame, imgsz=320, verbose=False, conf=0.25)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                if (x2 - x1) < w_full * 0.85 and (y2 - y1) < h_full * 0.85:
                    bboxes.append([float(x1), float(y1), float(x2), float(y2)])

        scale = self.detect_scale
        inv_scale = 1.0 / scale
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor.apply(small_frame)

        if self.prev_small_gray is not None:
            frame_diff = cv2.absdiff(small_gray, self.prev_small_gray)
            _, diff_mask = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
            combined_mask = cv2.bitwise_or(fg_mask, diff_mask)
        else:
            combined_mask = fg_mask
        self.prev_small_gray = small_gray

        closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.morph_kernel)

        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_area_max = small_frame.shape[0] * small_frame.shape[1] * 0.70
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 12 < area < small_area_max:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                bboxes.append([
                    float(x * inv_scale),
                    float(y * inv_scale),
                    float((x + w_box) * inv_scale),
                    float((y + h_box) * inv_scale)
                ])

        bboxes = self._merge_nearby_boxes(bboxes, merge_dist=140.0)
        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in bboxes]
        return bboxes, centers

    def process_frame(self, frame):
        # 1. PREDICT
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

        # 2. DETECT
        bboxes, centers = self._detect_targets(frame)

        if len(self.tracks) == 0:
            for bbox, center in zip(bboxes, centers):
                emb = self.extract_embedding(frame, bbox)
                self._spawn_track(center, bbox, emb)
            return self.tracks

        if len(bboxes) == 0:
            for t in self.tracks:
                t.skipped_frames += 1
            self._purge_stale_tracks()
            return self.tracks

        # 3. COST MATRIX
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
            total_cost = motion_cost

        # 4. HUNGARIAN ALGORITHM ASSIGNMENT
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

        # 5. LIFECYCLE CLEANUP
        for r in unmatched_tracks:
            self.tracks[r].skipped_frames += 1

        for c in unmatched_dets:
            emb = det_embeddings[c] if need_reid else self.extract_embedding(frame, bboxes[c])
            self._spawn_track(centers[c], bboxes[c], emb)

        self._purge_stale_tracks()
        return self.tracks

    def _spawn_track(self, center, bbox, embedding):
        self.tracks.append(Track(self.next_id, center, bbox, embedding))
        self.next_id += 1

    def _purge_stale_tracks(self):
        self.tracks = [t for t in self.tracks if t.skipped_frames <= self.max_skipped_frames]


# 60 FPS VIDEO LOOP & TELEMETRY LOGGER
def main():
    input_video_path = r"C:\Users\ipos\Fake_Frames\10 yards_10 in_10 mph_1 ms_CROP_1066x1066_20261029_185017_672\Scene_Uncompressed.avi"
    output_video_path = "tracked_output_60fps.mp4"
    output_csv_path = "tracker_results_60fps.csv"

    if not os.path.exists(input_video_path):
        print(f"ERROR: Could not find video file at:\n{input_video_path}")
        return

    print(f"Opening practice video at locked {SIM_FPS:.0f} FPS simulation rate:\n{input_video_path}")
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("ERROR: OpenCV could not open the .avi file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Force output video encoding to exact 60.0 FPS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_video_path, fourcc, SIM_FPS, (width, height))

    pipeline = UnifiedPrototypePipeline()

    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "frame_idx", "sim_time_sec", "track_id", "center_x", "center_y",
            "vx_px_per_s", "vy_px_per_s", "pred_lead_x", "pred_lead_y",
            "tracking_mode", "frame_latency_ms", "within_60fps_budget"
        ])

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.perf_counter()
            active_tracks = pipeline.process_frame(frame)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            sim_time_sec = frame_idx / SIM_FPS
            within_budget = latency_ms <= FRAME_BUDGET_MS

            # Bottom HUD Banner
            banner_h = 50
            sub_slice = frame[height - banner_h:height, 0:width]
            dark_rect = np.full(sub_slice.shape, 25, dtype=np.uint8)
            frame[height - banner_h:height, 0:width] = cv2.addWeighted(dark_rect, 0.75, sub_slice, 0.25, 0)

            for track in active_tracks:
                x1, y1, x2, y2 = map(int, track.bbox)
                cx, cy = int(track.prediction[0]), int(track.prediction[1])
                vx_s, vy_s = track.kf.get_velocity_px_per_sec()

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

                    # Draw Magenta "LEAD" reticle at LEAD_INTERCEPT_FRAMES (250 ms ahead at 60 FPS)
                    lead_idx = min(LEAD_INTERCEPT_FRAMES - 1, len(track.future_path) - 1)
                    lead_x, lead_y = track.future_path[lead_idx]
                    cv2.circle(frame, (lead_x, lead_y), 8, (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.line(frame, (lead_x - 12, lead_y), (lead_x + 12, lead_y), (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.line(frame, (lead_x, lead_y - 12), (lead_x, lead_y + 12), (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, "LEAD (+250ms)", (lead_x + 10, lead_y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

                if track.is_coasting:
                    color = (0, 140, 255)
                    mode_str = "KALMAN_BLUR_PREDICT"
                else:
                    color = (40, 210, 40)
                    mode_str = "TARGET_LOCKED"

                # 2. Draw current target bounding box and center aiming dot
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)

                label = f"ID:{track.track_id} [{mode_str}] ({cx},{cy})"
                cv2.putText(frame, label, (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

                writer.writerow([
                    frame_idx, round(sim_time_sec, 4), track.track_id,
                    round(cx, 2), round(cy, 2), round(vx_s, 1), round(vy_s, 1),
                    round(lead_x, 2), round(lead_y, 2), mode_str,
                    round(latency_ms, 2), within_budget
                ])

            # Bottom HUD Bar 
            budget_status = "60Hz OK" if within_budget else "OVER BUDGET"
            hud = (
                f"60FPS Sim | T: {sim_time_sec:05.2f}s (Frame {frame_idx:04d}) | "
                f"Latency: {latency_ms:.1f}/{FRAME_BUDGET_MS:.1f} ms [{budget_status}] | "
                f"Cyan: Predicted Path | Magenta: Lead Aim"
            )
            cv2.putText(frame, hud, (15, height - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            out_video.write(frame)
            cv2.imshow("Turret Vision Prototype - 60 FPS Hardware Loop (Press Q to Quit)", frame)

            frame_idx += 1
            # Dynamically pace playback so the live window runs at 60 FPS (16.67 ms per frame total)
            wait_ms = max(1, int(round(FRAME_BUDGET_MS - latency_ms)))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    print(f"Done! Processed {frame_idx} frames ({frame_idx / SIM_FPS:.2f} seconds at 60 FPS).")
    print(f"Saved 60 FPS video to '{output_video_path}' and hardware telemetry to '{output_csv_path}'.")

if __name__ == "__main__":
    main()