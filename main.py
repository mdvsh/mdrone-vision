import numpy as np
import logging
from .config.config import CameraConfig, TagConfig, UARTConfig
from .core.camera import Camera
from .core.tag_detector import TagDetector
from .comms.uart_interface import UARTInterface
import time

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    camera_config = CameraConfig()
    tag_config = TagConfig()
    uart_config = UARTConfig()
    
    try:
        calib_data = np.load('calibration/camera_calibration.npz')
        camera_matrix = calib_data['camera_matrix']
        dist_coeffs = calib_data['dist_coeffs']
    except Exception as e:
        logging.error(f"Failed to load camera calibration: {e}")
        return
    
    camera = Camera(camera_config)
    detector = TagDetector(tag_config, camera_matrix, dist_coeffs)
    uart = UARTInterface(uart_config)
    
    logging.info("Drone landing vision system initialized")
    
    try:
        while True:
            frame = camera.capture_frame()
            detections = detector.detect(frame)
            
            if detections:
                for detection in detections:
                    if uart.send_detection(detection):
                        logging.info(
                            f"Tag {detection.tag_id}: "
                            f"pos=({detection.position[0]:.2f}, "
                            f"{detection.position[1]:.2f}, "
                            f"{detection.position[2]:.2f}) "
                            f"rpy=({detection.rotation[0]:.2f}, "
                            f"{detection.rotation[1]:.2f}, "
                            f"{detection.rotation[2]:.2f})"
                        )
            else:
                uart.send_no_detection()
            
            # rate control
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        camera.cleanup()
        uart.cleanup()

if __name__ == '__main__':
    main()