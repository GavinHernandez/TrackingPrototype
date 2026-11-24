import time
import platform
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# ==========================================
# CONFIGURATION (JETSON PLUG-AND-PLAY)
# ==========================================
SPLIT_SINGLE_CAM_TEST = False  # False = Use 2 physical cameras on the Jetson
                               # True  = Split 1 webcam into Cam 1 & Cam 2 for laptop testing
USE_CSI_CAMERAS = False        # Set True ONLY if using MIPI CSI ribbon cameras

# On Jetson Linux, 2 USB cameras usually map to /dev/video0 and /dev/video2
# (CSI ribbon cameras use sensor-id 0 and 1; Windows webcams use 0 and 1)
CAM1_SOURCE = 0
CAM2_SOURCE = 1 if (USE_CSI_CAMERAS or platform.system() != "Linux") else 2

REID_MATCH_THRESHOLD = 0.35       # Lower = stricter match (0.0 = identical, 1.0 = different)
GLOBAL_MEMORY_SEC = 10.0          # Seconds to remember a lost target for handoff
TARGET_CLASSES = [0, 39, 41, 67]  # 0=Person, 39=Bottle, 41=Cup, 67=Cell Phone (Set None for all)
DISPLAY_W, DISPLAY_H = 520, 390   # Size of each camera panel in the side-by-side dashboard


# ==========================================
# 1. CAMERA LOADER (WINDOWS + JETSON LINUX)
# ==========================================
def open_camera(source, use_csi=False, width=640, height=480, fps=30):
    """Opens USB or MIPI CSI cameras cleanly on both Windows and Jetson Linux."""
    if use_csi:
        gst_pipeline = (
            f"nvarguscamerasrc sensor-id={source} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"format=NV12, framerate={fps}/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )
        return cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if platform.system() == "Windows":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
        return cap

    # Jetson / Linux USB Camera (V4L2 + MJPG prevents USB bandwidth saturation)
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


# ==========================================
# 2. FILTERPY LINEAR KALMAN FILTER
# ==========================================
class TargetKF:
    def __init__(self, initial_x, initial_y, dt=0.033):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([initial_x, initial_y, 0., 0.])
        self.kf.F = np.array([[1., 0., dt, 0.],
                              [0., 1., 0., dt],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])
        self.kf.H = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])
        self.kf.R = np.diag([0.5, 0.5])
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=2.0, block_size=2)
        self.kf.P *= 10.0

    def update(self, z_x, z_y):
        self.kf.predict()
        self.kf.update(np.array([z_x, z_y]))

    def get_lead_intercept(self, lookahead_frames=10, dt=0.033):
        time_ahead = lookahead_frames * dt
        pred_x = self.kf.x[0] + (self.kf.x[2] * time_ahead)
        pred_y = self.kf.x[1] + (self.kf.x[3] * time_ahead)
        return int(pred_x), int(pred_y)


# ==========================================
# 3. GLOBAL CROSS-CAMERA ReID MANAGER
# ==========================================
class GlobalTrackManager:
    PALETTE = [
        (50, 255, 50),    # Bright Green
        (255, 220, 0),    # Cyan
        (0, 230, 255),    # Yellow
        (255, 100, 255),  # Pink/Magenta
        (0, 140, 255),    # Orange
        (255, 150, 80),   # Sky Blue
    ]

    def __init__(self, reid_threshold=0.35, max_memory_sec=10.0):
        self.reid_threshold = reid_threshold
        self.max_memory_sec = max_memory_sec
        self.next_global_id = 1
        self.local_to_global = {}   # (cam_id, local_id) -> global_id
        self.gallery = {}           # global_id -> dict(embedding, cls_id, last_seen, last_cam, color)
        self.last_handoff_msg = "Waiting for cross-camera handoff..."
        self.last_handoff_time = 0.0

    @staticmethod
    def extract_embedding(frame, bbox):
        """Extracts a 3-strip (top/mid/bottom) HSV appearance vector for fast ReID."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 9 or crop.shape[1] < 9:
            return None

        # Trim 10% border to reduce background bleed
        ch, cw = crop.shape[:2]
        crop = crop[int(ch * 0.1):int(ch * 0.9), int(cw * 0.1):int(cw * 0.9)]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        strips = np.array_split(hsv, 3, axis=0)

        features = []
        for strip in strips:
            hist = cv2.calcHist([strip], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
            norm = np.linalg.norm(hist)
            if norm > 1e-6:
                hist /= norm
            features.append(hist)

        vec = np.concatenate(features).astype(np.float32)
        total_norm = np.linalg.norm(vec)
        return (vec / total_norm) if total_norm > 1e-6 else None

    def resolve_frame_ids(self, cam_id, frame, boxes, local_ids, class_ids, now):
        """Maps all local tracker IDs in a frame to persistent Global IDs."""
        resolved = []

        active_gids_in_cam = set()
        for lid in local_ids:
            key = (cam_id, int(lid))
            if key in self.local_to_global:
                active_gids_in_cam.add(self.local_to_global[key])

        for box, lid, cid in zip(boxes, local_ids, class_ids):
            lid, cid = int(lid), int(cid)
            key = (cam_id, lid)
            emb = self.extract_embedding(frame, box)

            # Case 1: Already mapped in this camera
            if key in self.local_to_global:
                gid = self.local_to_global[key]
                entry = self.gallery[gid]
                entry["last_seen"] = now
                entry["last_cam"] = cam_id
                if emb is not None and entry["embedding"] is not None:
                    updated = 0.85 * entry["embedding"] + 0.15 * emb
                    entry["embedding"] = updated / (np.linalg.norm(updated) + 1e-6)
                resolved.append((gid, entry["color"], False))
                continue

            # Case 2: New local track -> search gallery for cross-camera match
            best_gid = None
            best_dist = 1.0

            if emb is not None:
                for gid, entry in self.gallery.items():
                    if gid in active_gids_in_cam:
                        continue
                    if entry["cls_id"] != cid:
                        continue
                    if (now - entry["last_seen"]) > self.max_memory_sec:
                        continue
                    if entry["embedding"] is None:
                        continue

                    dist = float(1.0 - np.dot(emb, entry["embedding"]))
                    if dist < best_dist:
                        best_dist = dist
                        best_gid = gid

            if best_gid is not None and best_dist < self.reid_threshold:
                prev_cam = self.gallery[best_gid]["last_cam"]
                self.local_to_global[key] = best_gid
                active_gids_in_cam.add(best_gid)

                is_handoff = (prev_cam != cam_id)
                if is_handoff:
                    match_pct = (1.0 - best_dist) * 100.0
                    self.last_handoff_msg = (
                        f"HANDOFF CONFIRMED: GID {best_gid} "
                        f"(CAM {prev_cam} -> CAM {cam_id}) | Similarity: {match_pct:.1f}%"
                    )
                    self.last_handoff_time = now

                self.gallery[best_gid]["last_seen"] = now
                self.gallery[best_gid]["last_cam"] = cam_id
                resolved.append((best_gid, self.gallery[best_gid]["color"], is_handoff))
            else:
                gid = self.next_global_id
                self.next_global_id += 1
                color = self.PALETTE[(gid - 1) % len(self.PALETTE)]
                self.gallery[gid] = {
                    "embedding": emb,
                    "cls_id": cid,
                    "last_seen": now,
                    "last_cam": cam_id,
                    "color": color,
                }
                self.local_to_global[key] = gid
                active_gids_in_cam.add(gid)
                resolved.append((gid, color, False))

        return resolved


# ==========================================
# 4. CAMERA PROCESSING & DRAWING HELPER
# ==========================================
def process_camera(cam_id, frame, model, global_mgr, kf_registry, now, device):
    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        classes=TARGET_CLASSES,
        conf=0.40,
        device=device,
        verbose=False
    )

    visible_gids = []

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        local_ids = results[0].boxes.id.int().cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().numpy()

        resolved = global_mgr.resolve_frame_ids(cam_id, frame, boxes, local_ids, class_ids, now)

        for box, lid, cid, (gid, color, is_handoff) in zip(boxes, local_ids, class_ids, resolved):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            visible_gids.append(gid)

            kf_key = (cam_id, gid)
            if kf_key not in kf_registry:
                kf_registry[kf_key] = TargetKF(cx, cy)
            else:
                kf_registry[kf_key].update(cx, cy)

            lead_x, lead_y = kf_registry[kf_key].get_lead_intercept()
            cls_name = model.names.get(int(cid), str(cid)).upper()

            thickness = 4 if is_handoff else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"GID:{gid} (L:{int(lid)}) {cls_name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)
            cv2.circle(frame, (lead_x, lead_y), 7, (255, 0, 255), -1)
            cv2.line(frame, (int(cx), int(cy)), (lead_x, lead_y), (255, 0, 255), 2)

    cv2.rectangle(frame, (0, 0), (210, 32), (20, 20, 20), -1)
    cv2.putText(frame, f"CAMERA {cam_id} | Active: {visible_gids}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    return frame, visible_gids


# ==========================================
# 5. MAIN DUAL-CAMERA DASHBOARD LOOP
# ==========================================
def main():
    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else "cpu"
    hw_label = f"GPU ({torch.cuda.get_device_name(0)})" if use_gpu else "CPU ONLY"
    print(f"Hardware Acceleration: {hw_label}")

    print("Loading two independent YOLOv8 + BoT-SORT trackers...")
    model_cam1 = YOLO("yolov8n.pt")
    model_cam2 = YOLO("yolov8n.pt")

    cap1 = open_camera(CAM1_SOURCE, use_csi=USE_CSI_CAMERAS)
    if not cap1.isOpened():
        print(f"ERROR: Could not open Camera 1 (source={CAM1_SOURCE}).")
        return

    cap2 = None
    if not SPLIT_SINGLE_CAM_TEST:
        cap2 = open_camera(CAM2_SOURCE, use_csi=USE_CSI_CAMERAS)
        if not cap2.isOpened():
            print(
                f"ERROR: Could not open Camera 2 (source={CAM2_SOURCE}). "
                "Run 'v4l2-ctl --list-devices' on the Jetson to check your camera indices."
            )
            cap1.release()
            return

    global_mgr = GlobalTrackManager(
        reid_threshold=REID_MATCH_THRESHOLD,
        max_memory_sec=GLOBAL_MEMORY_SEC
    )
    kf_registry = {}

    mode_label = "SIMULATED SPLIT-CAM" if SPLIT_SINGLE_CAM_TEST else "DUAL PHYSICAL CAMS"
    print(f"Running in [{mode_label}] mode. Press 'q' to quit.")

    while True:
        now = time.perf_counter()

        if SPLIT_SINGLE_CAM_TEST:
            ok, full_frame = cap1.read()
            if not ok or full_frame is None:
                print("ERROR: Lost feed from Camera 1.")
                break
            h, w = full_frame.shape[:2]
            mid = w // 2
            gap = 20
            f1 = full_frame[:, :mid - gap].copy()
            f2 = full_frame[:, mid + gap:].copy()
        else:
            ok1, f1 = cap1.read()
            ok2, f2 = cap2.read()
            if not ok1 or f1 is None or not ok2 or f2 is None:
                print(f"ERROR: Frame grab failed (Cam 1 ok={ok1}, Cam 2 ok={ok2}).")
                break

        f1 = cv2.resize(f1, (DISPLAY_W, DISPLAY_H))
        f2 = cv2.resize(f2, (DISPLAY_W, DISPLAY_H))

        f1, gids1 = process_camera(1, f1, model_cam1, global_mgr, kf_registry, now, device)
        f2, gids2 = process_camera(2, f2, model_cam2, global_mgr, kf_registry, now, device)

        divider = np.full((DISPLAY_H, 6, 3), (80, 80, 80), dtype=np.uint8)
        combined = np.hstack([f1, divider, f2])

        dash_h = 85
        dashboard = np.full((dash_h, combined.shape[1], 3), (25, 25, 25), dtype=np.uint8)

        recent_handoff = (now - global_mgr.last_handoff_time) < 2.0
        msg_color = (50, 255, 50) if recent_handoff else (220, 220, 220)

        cv2.putText(dashboard, f"MODE: {mode_label} | {hw_label} | ReID: {REID_MATCH_THRESHOLD}",
                    (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(dashboard, global_mgr.last_handoff_msg,
                    (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.60, msg_color, 2, cv2.LINE_AA)
        cv2.putText(dashboard, f"CAM 1 GIDs: {gids1}    |    CAM 2 GIDs: {gids2}    |    Press 'Q' to Quit",
                    (15, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        final_view = np.vstack([combined, dashboard])
        cv2.imshow("PATS II - Dual-Camera Global Handoff Monitor", final_view)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap1.release()
    if cap2 is not None:
        cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()