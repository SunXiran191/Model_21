#!/usr/bin/env python3
"""
navigator_api.py — 岔路语义导航 Python 子进程入口
====================================================
由 C++ OCR 线程通过 popen 调用:

  仅 OCR 模式:  python3 navigator_api.py --ocr-only <crop_image>
  完整流水线:   python3 navigator_api.py <crop_image>

输出: 识别文字 ("--ocr-only" 模式) 或 "直行"/"右转" (完整模式)

环境变量:
  BAIDU_API_KEY     - 百度千帆 API Key
  BAIDU_SECRET_KEY  - 百度千帆 Secret Key
"""

import os
import sys
import json
import numpy as np
import cv2

# ─── 路径配置 ──────────────────────────────────────────
# navigator_api.py 位于 src/inference/ocr/
# 模型/字典位于原始 workspace 目录下，从脚本向上 5 级到达 workspace 根
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../../../../.."))
OCR_MODEL = os.path.join(WORKSPACE_ROOT,
    "Model_21-6.15/modelInference_py/model/ppocr_v4_rec_rk3588.rknn")
OCR_DICT  = os.path.join(WORKSPACE_ROOT, "objDetect_1/ppocr_keys_v1.txt")

# ─── 百度千帆 v2 API 凭证 (OpenAI 兼容) ──────────────────
# 在 IAM 控制台获取: https://console.bce.baidu.com/iam/#/iam/apikey/list
BAIDU_API_KEY    = os.environ.get("BAIDU_API_KEY",    "bce-v3/****-encrypted:U2FsdGVkX1+Z7H3Kj9mQwR5tVnB4cXy8pL6oI2uN0fA=" )

# ─── v2 API 端点 (OpenAI 兼容) ─────────────────────────
CHAT_URL = "https://qianfan.baidubce.com/v2/chat/completions"

# ─── LLM Prompt ───────────────────────────────────────
SYSTEM_PROMPT = (
    "你是智能驾驶导航助手。你面前只有直行（左道）和岔路（右道）两条道路。"
    "根据道路指示牌文字判断直行或右转。"
    "注意：OCR可能有错字（如"道"误为"造"），请根据语义推理。"
    "规则：左侧/左道封闭或施工→走右道→右转；右侧/右道封闭或施工→走左道→直行。"
    "仅返回\"直行\"或\"右转\"或\"无法判断\"三个词。"
)


# ============================================================
#  OCR
# ============================================================
def load_ocr():
    """加载 PP-OCRv4 RKNN 模型 + 字典"""
    from rknnlite.api import RKNNLite

    if not os.path.exists(OCR_MODEL):
        print(f"[OCR] model not found: {OCR_MODEL}", file=sys.stderr)
        sys.exit(1)

    with open(OCR_DICT, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n").rstrip("\r") for line in f]
    char_list = ["blank"] + lines + [" "]  # 6625 classes

    rknn = RKNNLite()
    if rknn.load_rknn(OCR_MODEL) != 0:
        print("[OCR] load_rknn failed", file=sys.stderr)
        sys.exit(1)
    if rknn.init_runtime() != 0:
        rknn.release()
        print("[OCR] init_runtime failed", file=sys.stderr)
        sys.exit(1)

    return rknn, char_list


def ocr_recognize(image_path, rknn, char_list):
    """PP-OCRv4 识别: 预处理 → NPU推理 → CTC解码"""
    img = cv2.imread(image_path)
    if img is None:
        return ""

    h, w = img.shape[:2]
    ratio = 48.0 / h
    new_w = int(round(w * ratio))
    if new_w > 320:
        new_w = 320

    img_r = cv2.resize(img, (new_w, 48), interpolation=cv2.INTER_LINEAR)
    pad_w = 320 - new_w
    if pad_w > 0:
        img_r = np.pad(img_r, ((0, 0), (0, pad_w), (0, 0)), mode="constant")

    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(img_r, 0).astype(np.uint8)

    outputs = rknn.inference(inputs=[inp])
    probs = outputs[0][0]  # (T, C)

    # CTC greedy decode
    idx = np.argmax(probs, axis=-1)
    decoded, prev = [], 0
    for i in idx:
        if i == 0:
            prev = 0
            continue
        if i == prev:
            continue
        decoded.append(char_list[i] if i < len(char_list) else "?")
        prev = i
    return "".join(decoded)


# ============================================================
#  API (千帆 v2 OpenAI 兼容)
# ============================================================
def _http_post(url, body=None, timeout=10):
    """HTTP POST with Bearer Auth"""
    import urllib.request

    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data)
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {BAIDU_API_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"[API] error: {e}", file=sys.stderr)
        return {}


def call_ernie(ocr_text):
    """v2 直接调用，无需 token 交换"""
    body = {
        "model": "ernie-4.5-turbo-32k",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"指示牌文字：{ocr_text}"}
        ],
        "temperature": 0.1,
    }
    result = _http_post(CHAT_URL, body, timeout=15)
    # OpenAI 格式: choices[0].message.content
    choices = result.get("choices", [])
    if choices:
        answer = choices[0].get("message", {}).get("content", "")
    else:
        answer = ""

    if "无法判断" in answer:
        return "无法判断"
    if "直行" in answer:
        return "直行"
    if "右转" in answer:
        return "右转"
    if "直" in answer or "前" in answer:
        return "直行"
    return "无法判断"


def rule_based(ocr_text):
    """无 API Key 时的关键词规则兜底（含 OCR 误识别变体）"""
    # 左侧/左道封闭 → 走右侧 → 右转
    for kw in ["左道封闭", "左侧封闭", "左道施工", "左侧施工",
               "左道关闭", "左方封闭", "左方施工",
               "左造施工", "左造封闭", "左倒施工"]:
        if kw in ocr_text:
            return "右转"
    # 右侧/右道封闭 → 走左侧 → 直行
    for kw in ["右道封闭", "右侧封闭", "右道施工", "右侧施工",
               "右道关闭", "右方封闭", "右方施工"]:
        if kw in ocr_text:
            return "直行"
    for kw in ["右转", "右侧", "右", "绕行", "转入"]:
        if kw in ocr_text:
            return "右转"
    for kw in ["直行", "直", "前", "继", "通过"]:
        if kw in ocr_text:
            return "直行"
    return "右转"


# ============================================================
#  main
# ============================================================
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="裁剪文本图像路径")
    parser.add_argument("--ocr-only", action="store_true",
                        help="仅 OCR，不调用 API")
    parser.add_argument("--max-retries", type=int, default=30,
                        help="OCR 最大重试次数 (默认 10)")
    parser.add_argument("--max-api-rounds", type=int, default=10,
                        help="OCR+API 流水线最大轮次 (默认 5)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("右转")
        sys.exit(0)

    # Step 1: 加载 OCR 模型（只加载一次）
    rknn, char_list = load_ocr()
    try:
        # ── OCR + API 流水线（带重试） ──
        #  - OCR 失败/空/短 → 重试 OCR（最多 max_retries 次）
        #  - API 返回"无法判断" → 重新 OCR + API（最多 max_api_rounds 轮）
        pipeline_done = False
        for api_round in range(args.max_api_rounds):
            if api_round > 0:
                print(f"[Pipeline] retry round #{api_round}", file=sys.stderr)

            # ---- 1. OCR 识别（带重试） ----
            text = ""
            ocr_ok = False
            for ocr_attempt in range(args.max_retries):
                if ocr_attempt > 0:
                    print(f"[OCR] retry #{ocr_attempt}", file=sys.stderr)

                text = ocr_recognize(args.image, rknn, char_list)
                text = text.strip()
                print(f"[OCR] attempt #{ocr_attempt}: '{text}'", file=sys.stderr)

                # 至少需要 2 个字符
                if len(text) >= 2:
                    ocr_ok = True
                    break

                if ocr_attempt < args.max_retries - 1:
                    time.sleep(0.1)  # 短暂等待后重试

            if args.ocr_only:
                print(text if ocr_ok else "")
                sys.exit(0 if ocr_ok else 1)

            if not ocr_ok:
                print(f"[Pipeline] OCR exhausted retries, fallback 右转", file=sys.stderr)
                print("右转")
                sys.exit(0)

            # ---- 2. API 语义分析 ----
            decision = call_ernie(text)
            print(f"[API] decision: '{decision}'", file=sys.stderr)

            if decision == "无法判断":
                print("[Pipeline] API → 无法判断, re-running OCR+API...", file=sys.stderr)
                continue  # 重新 OCR + API
            elif decision in ("直行", "右转"):
                print(decision)
                pipeline_done = True
                break
            else:
                # 兜底：关键词规则
                decision = rule_based(text)
                print(f"[Pipeline] API unclear, rule fallback → {decision}", file=sys.stderr)
                print(decision)
                pipeline_done = True
                break

        if not pipeline_done:
            print(f"[Pipeline] all rounds exhausted, fallback 直行", file=sys.stderr)
            print("直行")

    finally:
        rknn.release()
