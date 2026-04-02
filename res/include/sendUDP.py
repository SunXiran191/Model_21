import socket
import time
import json

class UdpTelemetrySender:
    def __init__(self, ip: str, port: int):
        self.is_initialized = False
        self.server_addr = (ip, port)
        
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.is_initialized = True
        except Exception as e:
            print(f"[UDP Sender] Socket 创建失败! {e}")

    def __del__(self):
        if hasattr(self, 'sock'):
            self.sock.close()

    def send_data(self, id_str: str, x: float, y: float, z: float, yaw: float):
        if not self.is_initialized:
            return

        # 获取Unix时间戳 (秒，带小数代表毫秒)
        timestamp = time.time()

        # 构造 JSON 对象
        data = {
            "id": id_str,
            "timestamp": round(timestamp, 3),
            "x": round(x, 3),
            "y": round(y, 3),
            "z": round(z, 3),
            "pitch": 0.0,
            "yaw": round(yaw, 3),
            "roll": 0.0
        }

        # 转换为 JSON 字符串并发送
        try:
            msg = json.dumps(data).encode('utf-8')
            self.sock.sendto(msg, self.server_addr)
        except Exception as e:
            print(f"[UDP Sender] 发送失败! {e}")
