"""UART communication interface."""
import serial
import struct
import time
import logging

class UARTInterface:
    def __init__(self, config):
        self.config = config
        self.ser = serial.Serial(
            port=config.port,
            baudrate=config.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )
        # Flush any existing data
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def _print_bytes(self, data: bytes, msg: str = ""):
        print(f"{msg} [{len(data)} bytes]:")
        print(" ".join([f"{b:02X}" for b in data]))

    def send_detection(self, detection) -> bool:
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
            self._print_bytes(data, "Sending detection packet")
            self.ser.write(data)
            return True
        except Exception as e:
            logging.error(f"Failed to send detection: {e}")
            return False

    def send_no_detection(self) -> bool:
        timestamp = int(time.time() * 1000)
        try:
            data = struct.pack('!BBQ',
                self.config.header,
                self.config.msg_no_detection,
                timestamp
            )
            self._print_bytes(data, "Sending no-detection packet")
            self.ser.write(data)
            return True
        except Exception as e:
            logging.error(f"Failed to send no-detection: {e}")
            return False

    def cleanup(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
