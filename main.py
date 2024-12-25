import numpy as np
import logging
from config.config import CameraConfig, TagConfig, UARTConfig
from core.camera import Camera
from core.tag_detector import TagDetector
from comms.uart_interface import UARTInterface
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
                        pos_x = float(detection.position[0])
                        pos_y = float(detection.position[1])
                        pos_z = float(detection.position[2])
                        rot_x = float(detection.rotation[0])
                        rot_y = float(detection.rotation[1])
                        rot_z = float(detection.rotation[2])

                        log_msg = (
                            f"Tag {detection.tag_id}: "
                            f"pos=({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f}) "
                            f"rpy=({rot_x:.2f}, {rot_y:.2f}, {rot_z:.2f})"
                        )
                        logging.info(log_msg)
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
