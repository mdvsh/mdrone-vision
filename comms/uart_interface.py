"""UART communication interface."""
import serial
import struct
import time
from ..config.config import UARTConfig
from ..core.tag_detector import TagDetection

class UARTInterface:
    def __init__(self, config: UARTConfig):
        self.config = config
        self.ser = serial.Serial(config.port, config.baudrate)
    
    def send_detection(self, detection: TagDetection) -> bool:
        timestamp = int(time.time() * 1000)
        try:
            data = struct.pack('!BBQB6f',
                self.config.header,
                self.config.msg_detection,
                timestamp,
                detection.tag_id,
                *detection.position,
                *detection.rotation
            )
            self.ser.write(data)
            return True
        except Exception as e:
            return False
    
    def send_no_detection(self) -> bool:
        timestamp = int(time.time() * 1000)
        try:
            data = struct.pack('!BBQ',
                self.config.header,
                self.config.msg_no_detection,
                timestamp
            )
            self.ser.write(data)
            return True
        except Exception as e:
            return False
    
    def cleanup(self):
        if self.ser and self.ser.is_open:
            self.ser.close()