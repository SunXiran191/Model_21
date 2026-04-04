import cv2
import time
import threading
from infer_wrap import InferWrap
from lineTrack import LineTracker  # 导入巡线类

def main():
    print("正在初始化 NPU 推理引擎...")
    infer = InferWrap(model_dir="model", TPEs=3)
    
    print("正在初始化 传统视觉巡线器...")
    tracker = LineTracker()
    
    # 连接视频流（可以换成 0 连接本地 USB 摄像头）
    stream_url = "http://192.168.137.168:8080/ar_feed"
    print(f"正在连接视频流: {stream_url}")
    cap = CameraStream(stream_url)
    
    time.sleep(1.0)
    ret, _ = cap.read()
    if not ret:
        print("无法读取视频流，请检查URL地址或网络连接。")
        return
        
    print("视频流连接成功，开始实时推理... (按 'q' 退出)")
    
    fps = 0.0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
            
        # 统一缩放尺寸，适配 LineTracker 调优过的分辨率
        frame = cv2.resize(frame, (640, 480))
        
        # 步骤 1: 传统图像巡线 (在送入推理前处理最新帧)
        lane_points, mask = tracker.extract_arrows(frame)
        
        # 提取到赛道点后进行平滑拟合，并直接绘制在 frame 上
        if len(lane_points) > 3:
            filtered_line = tracker.fit_trajectory_lowess(len(lane_points), lane_points, frame)
            # 画黄点 (源点)
            #for pt in lane_points[:-1]:
                #cv2.circle(frame, pt, 4, (0, 255, 0), -1) 
            # 画蓝线 (拟合后的线段)
            for i in range(len(filtered_line) - 1):
                cv2.line(frame, filtered_line[i], filtered_line[i + 1], (255, 0, 0), 3)

        # 步骤 2: 目标检测推理
        # 这里传入的 frame 已经被画上了巡线轨迹。
        # infer() 会把当前帧送入队列，并返回 TPEs (3) 帧之前的图像。
        result_img, flag = infer(frame)
        
        if flag and result_img is not None:
            frame_count += 1
            current_time = time.time()
            if current_time - start_time >= 1.0:
                fps = frame_count / (current_time - start_time)
                frame_count = 0
                start_time = current_time
            
            # 在图像左上角绘制实时FPS
            cv2.putText(result_img, f'FPS: {fps:.1f}', (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        
            # 窗口显示最终结果 (包含巡线 + 目标检测框)
            cv2.imshow("first view", result_img)
            # 如果需要调试二值化巡线图像，可以取消下面这行注释：
            # cv2.imshow("Line Mask Debug", mask)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("正在退出...")
            break

    cap.release()
    cv2.destroyAllWindows()
    infer.release()
    print("资源释放完毕。")

if __name__ == '__main__':
    main()