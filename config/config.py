from dataclasses import dataclass

@dataclass
class CameraConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    camera_id: int = 0

@dataclass
class TagConfig:
    landing_tag_id: int = 37      # Large landing pad tag ID
    precision_tag_id: int = 73    # Small precision tag ID
    landing_tag_size: float = 120  # mm
    precision_tag_size: float = 40 # mm
    tag_family: str = "tagCustom48h12"

@dataclass
class UARTConfig:
    port: str = "/dev/ttyAMA10"
    baudrate: int = 115200
    header: int = 0xAA
    msg_detection: int = 0x01
    msg_no_detection: int = 0x02