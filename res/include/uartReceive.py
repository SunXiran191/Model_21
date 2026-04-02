import serial
import threading
import time

USB_FRAME_HEAD = 0x42
USB_FRAME_LENMIN = 8
USB_FRAME_LENMAX = 12

USB_ADDR_CARCTRL = 1
USB_ADDR_BUZZER = 4
USB_ADDR_LED = 5
USB_ADDR_KEY = 6

class Buzzer(int):
    BUZZER_OK = 0
    BUZZER_WARNNING = 1
    BUZZER_FINISH = 2
    BUZZER_DING = 3
    BUZZER_START = 4

class Uart:
    def __init__(self, port: str):
        self.port_name = port
        self.is_open = False
        self.serial_port = None
        self.thread_rec = None
        
        self.keypress = False
        self.pitch_angle = 0.0
        
        self._running = False

    def __del__(self):
        self.close()

    def open(self) -> int:
        try:
            # 初始化串口，根据实际波特率调节
            self.serial_port = serial.Serial(self.port_name, baudrate=115200, timeout=1)
            self.is_open = True
            return 0
        except serial.SerialException as e:
            print(f"Failed to open port {self.port_name}, {e}")
            return -1

    def receive_bytes(self, timeout_ms=0) -> int:
        # 等待读取字节
        if self.serial_port and self.is_open:
            data = self.serial_port.read(1)
            if data:
                return data[0]
        return -1

    def transmit_byte(self, data: int) -> int:
        if self.serial_port and self.is_open:
            return self.serial_port.write(bytes([data]))
        return 0

    def start_receive(self):
        if not self.is_open:
            return
        
        self._running = True
        self.thread_rec = threading.Thread(target=self._receive_loop)
        self.thread_rec.daemon = True
        self.thread_rec.start()

    def _receive_loop(self):
        while self._running:
            # 持续运行串口解析逻辑
            pass

    def close(self):
        self._running = False
        if self.thread_rec and self.thread_rec.is_alive():
            self.thread_rec.join(timeout=1.0)
        
        if self.serial_port and self.is_open:
            self.serial_port.close()
            self.is_open = False
