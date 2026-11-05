import cv2
import numpy as np
from pypylon import pylon


class Basler:
    def __init__(self, _dbg=False):
        self._dbg = _dbg  # Init debugging bool
        # Connect to the first available Basler camera
        try:
            self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.cam.Open()
            if self._dbg:  # Debugging
                print(f"Connected to: {self.cam.GetDeviceInfo().GetModelName()}")
        except Exception as e:
            print(f"Error connecting to Basler camera: {e}")
            print("Ensure the camera is plugged into a USB 3.0 port, "
                  "matching network subnet, and closed in pylon Viewer.")
            exit()
        # Force the camera to output a standard format OpenCV expects
        self.cam.PixelFormat.SetValue("RGB8")
        # Get resolution
        self.width, self.height = self.cam.Width.GetValue(), self.cam.Height.GetValue()
        # Check FPS current, min, and max
        self.fps_current = self.cam.ResultingFrameRate.GetValue()
        self.fps_min, self.fps_max = self.cam.AcquisitionFrameRate.GetMin(), self.cam.AcquisitionFrameRate.GetMax()
        if self._dbg:  # Debugging
            print("Basler initialized.")

    def setFPS(self, target_fps):
        pass  # need to code

    def initGrab(self):
        # Start grabbing frames (GrabStrategy_LatestImageOnly keeps latency low)
        try:
            self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception as e:
            print(f"Error connecting to Basler camera: {e}")
            print("Ensure the camera is plugged into a USB 3.0 port, "
                  "matching network subnet, and closed in pylon Viewer.")
            exit()
        if self._dbg:  # Debugging
            print("Basler.initGrab successful.")

    def grabFrame(self):
        frame = np.zeros(shape=[self.height, self.width, 3], dtype=np.uint8)  # init frame var
        if self.cam.IsGrabbing():
            # Grab a frame with a 5000ms timeout threshold
            grabResult = self.cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                # Access the raw NumPy array directly from the buffer
                # and convert Basler RGB to OpenCV BGR
                frame = cv2.cvtColor(grabResult.Array, cv2.COLOR_RGB2BGR)
                if self._dbg:  # Debugging
                    print("Basler.grabFrame successful.")
            grabResult.Release()
            if self._dbg:  # Debugging
                print("Basler.grabFrame returning frame.")
        elif self._dbg:  # Debugging
            print("Basler.grabFrame unsuccessful.")
            print("Basler.grabFrame returning empty frame.")
        return frame

    def close(self):
        self.cam.StopGrabbing()
        self.cam.Close()
        if self._dbg:  # Debugging
            print("Basler.close successful.")


class WebCam:
    def __init__(self, target_res=(1920, 1080), target_fps=60, _dbg=False):
        self._dbg = _dbg  # Init debugging bool
        # Init first available camera, CAP_DSHOW prevents windows loading lag
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # Verify camera opened
        if not self.cam.isOpened():
            raise RuntimeError("Unable to open device.")
        # Request specific resolution and frame rate
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, target_res[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, target_res[1])
        self.cam.set(cv2.CAP_PROP_FPS, target_fps)
        # Read hardware settings
        self.width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_current = self.cam.get(cv2.CAP_PROP_FPS)
        if self._dbg:  # Debugging
            print("WebCam initialized.")
            print(f"    ----- WebCam settings -----\n"
                  f"fps={self.fps_current}    WxH={self.width}x{self.height}")

    def grabFrame(self):
        ret, frame = self.cam.read()
        # If frame failed, handle error
        if not ret:
            frame = np.zeros(shape=[self.height, self.width, 3], dtype=np.uint8)
            if self._dbg:
                print("WebCam.grabFrame returning empty frame.")
            # raise RuntimeError("WebCam.grabFrame unsuccessful. Unable to read frame.")
        if self._dbg:  # Debugging
            print("WebCam.grabFrame successful.")
            print("WebCam.grabFrame returning frame.")
        return frame

    def close(self):
        self.cam.release()
        if self._dbg:  # Debugging
            print("WebCam.close successful.")


if __name__ == '__main__':
    # camera = Basler()
    # camera.initGrab()
    camera = WebCam(target_res=(640, 480), target_fps=60, _dbg=True)
    while True:
        cv2.imshow("Frame", camera.grabFrame())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.close()
    cv2.destroyAllWindows()
