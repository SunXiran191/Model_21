import time
import threading

class Logger:
    def __init__(self, filename: str, log_interval_ms: int = 0):
        self.interval_ms = log_interval_ms
        self.log_mutex = threading.Lock()
        
        try:
            self.log_file = open(filename, 'a', encoding='utf-8')
            # 写入 CSV 表头
            self.log_file.write("AbsoluteTime(ms),X,Y,Angle,Speed\n")
        except Exception as e:
            print(f"Error: Cannot open log file: {filename}. {e}")
            self.log_file = None
            
        self.last_log_time = time.time() * 1000

    def __del__(self):
        if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
            self.log_file.close()

    def log(self, x: float, y: float, angle: float, speed: float):
        if not self.log_file or self.log_file.closed:
            return

        # 获取当前绝对系统时间（毫秒）
        now_ms = time.time() * 1000

        # 检查是否达到了设定的保存周期
        if self.interval_ms > 0:
            if (now_ms - self.last_log_time) < self.interval_ms:
                return
        
        self.last_log_time = now_ms

        # 加锁防止多线程写入冲突
        with self.log_mutex:
            # 格式化写入文件 (X 保留2位小数)
            self.log_file.write(f"{int(now_ms)},{x:.2f},{y},{angle},{speed}\n")
            # 强制刷入硬盘（需要时取消注释）
            # self.log_file.flush()
