"""AprilTag detection and processing."""
import cv2
import numpy as np
from apriltag import apriltag
from dataclasses import dataclass
from ..config.config import TagConfig

@dataclass
class TagDetection:
		tag_id: int
		position: np.ndarray  # [x, y, z]
		rotation: np.ndarray  # [roll, pitch, yaw]
		corners: np.ndarray   # Corner points in image

class TagDetector:
		def __init__(self, config: TagConfig, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
				self.config = config
				self.camera_matrix = camera_matrix
				self.dist_coeffs = dist_coeffs
				self.detector = apriltag(config.tag_family, threads=1)
				
				self.tag_points = {
						config.landing_tag_id: self._create_tag_points(config.landing_tag_size),
						config.precision_tag_id: self._create_tag_points(config.precision_tag_size)
				}
		
		def _create_tag_points(self, size: float) -> np.ndarray:
				"""Create 3D reference points for a tag."""
				half_size = size / 2
				return np.array([
						[-half_size,  half_size, 0],
						[ half_size,  half_size, 0],
						[ half_size, -half_size, 0],
						[-half_size, -half_size, 0]
				], dtype=np.float32)
		
		def detect(self, frame: np.ndarray) -> list[TagDetection]:
				"""Detect and process AprilTags in frame."""
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				detections = self.detector.detect(gray)
				
				results = []
				for det in detections:
						tag_id = det['id']
						if tag_id not in self.tag_points:
								continue
								
						corners = np.array(det['lb-rb-rt-lt'], dtype=np.float32)
						object_points = self.tag_points[tag_id]
						
						# Estimate pose
						ret, rvec, tvec = cv2.solvePnP(
								object_points, corners,
								self.camera_matrix, self.dist_coeffs,
								flags=cv2.SOLVEPNP_IPPE_SQUARE
						)
						
						euler_angles = get_euler_angles(rvec)
						
						results.append(TagDetection(
								tag_id=tag_id,
								position=tvec.flatten(),
								rotation=euler_angles,
								corners=corners
						))
				
				return results

# helper

def get_euler_angles(rvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector to Euler angles (roll, pitch, yaw)."""
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rmat[2, 1], rmat[2, 2])
        y = np.arctan2(-rmat[2, 0], sy)
        z = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        x = np.arctan2(-rmat[1, 2], rmat[1, 1])
        y = np.arctan2(-rmat[2, 0], sy)
        z = 0

    return np.array([np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)])