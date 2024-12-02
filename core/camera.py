"""Camera interface for capturing frames."""
from picamera2 import Picamera2
import numpy as np
from dataclasses import asdict
from ..config.config import CameraConfig

class Camera:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.camera = None
        self._initialize_camera()
    
    def _initialize_camera(self):
        self.camera = Picamera2(self.config.camera_id)
        config = self.camera.create_preview_configuration(main={"size": (self.config.width, self.config.height), "format": "RGB888"})
        self.camera.configure(config)
        self.camera.start()
    
    def capture_frame(self) -> np.ndarray:
        return self.camera.capture_array()
    
    def cleanup(self):
        if self.camera:
            self.camera.close()