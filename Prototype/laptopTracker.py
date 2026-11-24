import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter  # Swapped to the standard linear filter
from filterpy.common import Q_discrete_white_noise

# 1. Advanced Trajectory Predictor (FilterPy KF)
class TargetKF:
    def __init__(self, initial_x, initial_y, dt=0.033): # dt = 33ms for ~30 FPS webcam
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State: [x, y, velocity_x, velocity_y]
        self.kf.x = np.array([initial_x, initial_y, 0., 0.])
        
        # State Transition Matrix
        self.kf.F = np.array([[1., 0., dt, 0.],
                               [0., 1., 0., dt],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]])
        
        # Measurement Function
        self.kf.H = np.array([[1., 0., 0., 0.],
                               [0., 1., 0., 0.]])
        
        # Covariance Matrices
        self.kf.R = np.diag([0.5, 0.5]) 
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=2.0, block_size=2)
        self.kf.P *= 10.0 

    def update(self, z_x, z_y):
        self.kf.predict()
        self.kf.update(np.array([z_x, z_y]))
        
    def get_lead_intercept(self, lookahead_frames=15, dt=0.033):
        time_ahead = lookahead_frames * dt
        pred_x = self.kf.x[0] + (self.kf.x[2] * time_ahead)
        pred_y = self.kf.x[1] + (self.kf.x[3] * time_ahead)
        return int(pred_x), int(pred_y)

def main():
    # 2. Load the YOLO model (Downloads yolov8n.pt automatically on first run)
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    
    # 3. Open your laptop webcam (or replace 0 with a video file path)
    cap = cv2.VideoCapture(0)
    
    # Global registry to hold EKFs for each target ID
    active_targets = {}

    print("Starting tracker. Press 'q' to quit.")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # 4. Execute YOLO + BoT-SORT Tracking
        # persist=True keeps IDs across frames
        # tracker="botsort.yaml" enables ReID and camera-motion compensation
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)
        
        # 5. Extract results and run the EKF Math
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().numpy()
            
            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                # Calculate center of the bounding box
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                
                # Manage EKF lifecycle
                if track_id not in active_targets:
                    active_targets[track_id] = TargetKF(cx, cy)
                else:
                    active_targets[track_id].update(cx, cy)
                    
                # Predict future lead point (e.g., 15 frames ahead)
                lead_x, lead_y = active_targets[track_id].get_lead_intercept()
                
                # 6. Draw on the frame
                # Bounding Box
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                # Track ID
                cv2.putText(frame, f"ID: {track_id}", (int(box[0]), int(box[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Lead Aim Point (Magenta)
                cv2.circle(frame, (lead_x, lead_y), 8, (255, 0, 255), -1)
                cv2.line(frame, (lead_x, int(cy)), (int(cx), int(cy)), (255, 0, 255), 1)

        # Show the frame
        cv2.imshow("VS Code Laptop Prototype", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()