import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import json
import gc
import numpy as np
from rknnlite.api import RKNNLite
import time
from threading import Thread, Lock
import struct
import mmap

cv2.setUseOptimized(True)
cv2.setNumThreads(1)


# ============================================================
#  Python 线程绑核 — 避免在 A76/A55 间飘移导致 cache 颠簸
#  read_stream (轮询 SHM，轻量) → 小核 core 0
#  infer_worker (图像预处理，CPU 密集) → 大核 core 7
# ============================================================
def _bind_to_core(core_id: int) -> None:
    try:
        import os
        os.sched_setaffinity(0, [core_id])
    except (AttributeError, PermissionError):
        pass  # 非 Linux 或权限不足时静默跳过


# ============================================================
#  配置文件路径（可通过命令行参数覆盖）
#  默认从脚本所在目录向上到项目根目录的 res/configs/config_1.2m.json
#  用法: python infer.py [config_path]
# ============================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _resolve_config_path() -> str:
    """从 project_config.json 读取 selected_code_dir + selected_config，返回实际配置文件的路径。"""
    _PROJECT_CONFIG = os.path.join(_SCRIPT_DIR, "../../../../project_config.json")
    try:
        with open(_PROJECT_CONFIG, "r", encoding="utf-8") as _pf:
            _pj = json.load(_pf)
        code_dir = _pj.get("selected_code_dir", "")
        config_name = _pj.get("selected_config", "")
        if code_dir and config_name:
            if config_name.startswith("/"):
                return config_name
            return os.path.normpath(os.path.join(code_dir, "res/configs", config_name))
    except Exception as e:
        print(f"[WARN] failed to read project_config.json: {e}")
    # fallback 默认值
    return os.path.join(_SCRIPT_DIR, "../../../res/configs/config_1.2m.json")

DEFAULT_CONFIG_PATH = _resolve_config_path()

def _load_config(config_path: str) -> dict:
    """读取 JSON 配置文件，返回字典；失败时返回空 dict。"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # 支持 jsonpath，提取内层 config
        jp = cfg.get("jsonpath", "")
        if jp.startswith("$."):
            keys = jp.split(".")[1:]
            _nested = cfg
            for k in keys:
                _nested = _nested.get(k, {})
            # 如果 jsonpath 提取后为空，说明配置在顶层，直接用原始 cfg
            if _nested:
                cfg = _nested
        return cfg
    except Exception as e:
        print(f"[WARN] failed to load config from {config_path}: {e}")
        return {}

import sys as _sys
_config_path = DEFAULT_CONFIG_PATH
if len(_sys.argv) > 1:
    _config_path = _sys.argv[1]

# 读取 inferseg_debug 标志（控制 infer.py 自身的输出）
_INFERSEG_DEBUG = False
try:
    _PROJ_CFG = os.path.join(_SCRIPT_DIR, "../../../../project_config.json")
    with open(_PROJ_CFG, "r", encoding="utf-8") as _pf:
        _pj = json.load(_pf)
    _INFERSEG_DEBUG = _pj.get("debug", False) or _pj.get("inferseg_debug", False)
except Exception:
    pass
_config = _load_config(_config_path)

# 模型路径：从 config 文件中读取
_model_path_raw = _config.get("infer_seg_model_path", "")
if _model_path_raw:
    if os.path.isabs(_model_path_raw):
        MODEL_PATH = _model_path_raw
    else:
        MODEL_PATH = os.path.normpath(os.path.join(_SCRIPT_DIR, _model_path_raw))
else:
    print("[ERROR] infer_seg_model_path not found in config")
    MODEL_PATH = ""

print(f"[INFO] using config: {_config_path}")
print(f"[INFO] seg model: {MODEL_PATH}")

IMG_SIZE = 320

SHM_NAME = "shm_ar_video"
SHM_MAP_BYTES = 3*640*480 + 16
SHM_HEADER_SIZE = 16

SHM_OUT_NAME = "shm_ar_seg"
SHM_OUT_BYTES = 3*640*480 + 16
SHM_OUT_HEADER_SIZE = 16

NUM_WORKERS = 1

run_flag = True

latest_frame = None
latest_frame_id = 0
latest_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

frame_lock = Lock()
mask_lock = Lock()
infer_lock = Lock()

infer_count = 0


class ShmWriter:
    def __init__(self, name, map_bytes):
        self.name = name
        self.map_bytes = map_bytes
        self.fd = None
        self.mm = None

    def open(self):
        path = f"/dev/shm/{self.name}"
        try:
            self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
            os.ftruncate(self.fd, self.map_bytes)
            self.mm = mmap.mmap(self.fd, self.map_bytes, access=mmap.ACCESS_WRITE)
            return True
        except OSError as exc:
            print(f"[SHM_OUT] open failed: {exc}")
            self.close()
            return False

    def close(self):
        if self.mm is not None:
            self.mm.close()
            self.mm = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def write_mask(self, mask, frame_id=0):
        if self.mm is None:
            if not self.open():
                return False

        if mask is None or mask.size == 0:
            return False

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        h, w = mask.shape[:2]
        data = mask.tobytes()
        data_len = len(data)

        if SHM_OUT_HEADER_SIZE + data_len > self.map_bytes:
            print(f"[SHM_OUT] mask too large: {data_len} bytes")
            return False

        try:
            self.mm.seek(0)
            self.mm.write(struct.pack("<QII", frame_id, w, h))
            self.mm.write(data)
            self.mm.flush()
            return True
        except OSError as exc:
            print(f"[SHM_OUT] write failed: {exc}")
            return False


def get_core_mask():
    try:
        return RKNNLite.NPU_CORE_0_1  # core0+core1 给分割，core2 留给目标检测
    except AttributeError:
        return 3


def decode_mask(output):
    output = np.squeeze(output)

    if output.ndim == 2:
        if output.dtype.kind in "fc":
            min_val = float(output.min())
            max_val = float(output.max())
            threshold = 0.5 if 0.0 <= min_val and max_val <= 1.0 else 0.0
            return (output > threshold).astype(np.uint8)
        return output.astype(np.uint8)

    if output.ndim == 3:
        if output.shape[0] == 2:
            return np.argmax(output, axis=0).astype(np.uint8)
        if output.shape[-1] == 2:
            return np.argmax(output, axis=-1).astype(np.uint8)
        if output.shape[0] == 1:
            return (output[0] > 0.0).astype(np.uint8)
        if output.shape[-1] == 1:
            return (output[..., 0] > 0.0).astype(np.uint8)

    raise ValueError(f"Unsupported output shape: {output.shape}")


class ShmReader:
    def __init__(self, name, map_bytes):
        self.name = name
        self.map_bytes = map_bytes
        self.fd = None
        self.mm = None
        self.last_fid = -1
        self.mapped_bytes = 0

    def open(self):
        path = f"/dev/shm/{self.name}"
        try:
            self.fd = os.open(path, os.O_RDONLY)
            stat = os.fstat(self.fd)
            if stat.st_size <= 0:
                print("[SHM] open failed: shm size is 0")
                self.close()
                return False
            self.mapped_bytes = min(self.map_bytes, stat.st_size)
            self.mm = mmap.mmap(self.fd, self.mapped_bytes, access=mmap.ACCESS_READ)
            return True
        except OSError as exc:
            print(f"[SHM] open failed: {exc}")
            self.close()
            return False

    def close(self):
        if self.mm is not None:
            self.mm.close()
            self.mm = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.mapped_bytes = 0

    def read_frame(self):
        if self.mm is None:
            if not self.open():
                return None

        try:
            fid, w, h = struct.unpack_from("<QII", self.mm, 0)
        except struct.error:
            return None

        if fid == self.last_fid:
            return None
        if w == 0 or h == 0:
            return None

        data_len = int(w * h * 3)
        if SHM_HEADER_SIZE + data_len > self.mapped_bytes:
            return None

        data = self.mm[SHM_HEADER_SIZE:SHM_HEADER_SIZE + data_len]
        if len(data) != data_len:
            return None

        frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
        # 注意：不做垂直翻转！C++ produce_thread 已统一处理翻转。
        # 模型训练时使用原始相机图像（赛道在上方），翻转会破坏输入分布导致分割失败。
        # 仅做一次连续内存拷贝，避免后续 mmap 数据被覆写。
        frame = frame.copy()
        self.last_fid = fid
        return frame


def read_stream():
    global run_flag, latest_frame, latest_frame_id
    # 绑到小核 core 0（轮询 SHM 仅做内存拷贝，轻量操作）
    _bind_to_core(0)
    print("[INFO] Starting shared memory reader...")
    reader = ShmReader(SHM_NAME, SHM_MAP_BYTES)

    while run_flag:
        frame = reader.read_frame()
        if frame is None:
            time.sleep(0.002)
            continue

        with frame_lock:
            latest_frame = frame
            latest_frame_id += 1

        # 帧已到达，不需要额外 sleep 增加延迟

    reader.close()
    print("[INFO] Shared memory reader stopped.")


def infer_worker(thread_id, core_mask):
    global run_flag, latest_mask, infer_count
    # 绑到大核 core 7（图像预处理 resize/BGR2RGB 是 CPU 密集操作）
    _bind_to_core(7)
    print(f"[Worker {thread_id}] Starting on core mask: {core_mask}...")
    rknn = RKNNLite()

    ret = rknn.load_rknn(MODEL_PATH)
    if ret != 0:
        print(f"[Worker {thread_id}] load_rknn failed: {ret}")
        return

    ret = rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
        print(f"[Worker {thread_id}] init_runtime failed: {ret}")
        rknn.release()
        return

    last_seq = -1
    shm_writer = ShmWriter(SHM_OUT_NAME, SHM_OUT_BYTES)
    gc_counter = 0

    while run_flag:
        with frame_lock:
            current_seq = latest_frame_id
            frame = latest_frame

        if frame is None or current_seq == last_seq:
            time.sleep(0.002)
            continue

        last_seq = current_seq

        # read_frame 已返回 RGB 格式，无需重复转换
        # 垂直翻转，赛道在上方 !!!!!!!!!!!!!!!!!!!!
        #frame = cv2.flip(frame, 1)  # 垂直翻转，赛道在上方 !!!!!!!!!!!!!!!!!!!!
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img = np.ascontiguousarray(img[None, ...], dtype=np.uint8)

        # 每 100 帧执行一次垃圾回收，防止 numpy/opencv 临时对象累积
        gc_counter += 1
        if gc_counter >= 100:
            gc_counter = 0
            gc.collect()

        outputs = rknn.inference(inputs=[img], data_format=["nhwc"])
        if outputs is None or len(outputs) == 0:
            continue

        try:
            pred_mask = decode_mask(outputs[0])
        except Exception as e:
            print(f"[Worker {thread_id}] decode failed: {e}")
            continue

        if pred_mask.shape != (IMG_SIZE, IMG_SIZE):
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        with mask_lock:
            latest_mask = pred_mask

        shm_writer.write_mask(pred_mask, current_seq)

        with infer_lock:
            infer_count += 1

    shm_writer.close()
    rknn.release()
    print(f"[Worker {thread_id}] Stopped.")


if __name__ == "__main__":
    cam_thread = Thread(target=read_stream, daemon=True)
    cam_thread.start()
    time.sleep(1)

    core_mask = get_core_mask()
    worker_threads = []

    for i in range(NUM_WORKERS):
        t = Thread(target=infer_worker, args=(i, core_mask), daemon=True)
        t.start()
        worker_threads.append(t)

    fps_start = time.time()
    display_counter = 0
    display_fps = 0
    npu_fps = 0

    print("[INFO] Main loop started. Press q to quit.")

    while run_flag:
        # 显示代码已全部注释，主循环不再需要逐帧轮询渲染，
        # 改为长时间休眠，让 infer_worker 线程独立工作即可。
        time.sleep(1.0)
        # 每秒打印一次推理计数作为保活信号
        if _INFERSEG_DEBUG:
            print(f"[INFO] infer worker alive, masks produced so far: {infer_count}")

    cam_thread.join()
    for t in worker_threads:
        t.join()

    cv2.destroyAllWindows()
