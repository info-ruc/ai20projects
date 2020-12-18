# from tello import Tello
import time
import av
import numpy as np
import cv2
import math
import sys
import tellopy
import traceback

# 初始化避障阈值，光流计算等参数
v = 5
l_0 = 0.5
theta = math.atan(14.3 / 12.3) / 2
tantheta = math.tan(theta)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=200,
                      qualityLevel=0.2,
                      minDistance=7,
                      blockSize=7)

# 初始化实时避障计算参数
track_len_limit = 5
detect_interval = 5
tracks = []
tracks_len = []
frame_idx = 0
command_count = 0
command_left_count = 0
command_right_count = 0
x_resol = 0
y_resol = 0

drone = tellopy.Tello()

try:
    drone.connect()
    drone.wait_for_connection(60.0)

    # 多次尝试读取tello视频帧
    retry = 3
    container = None
    while container is None and 0 < retry:
        retry -= 1
        try:
            # container = "C:/Users/Flanker/Personal/PyCharm/test1.mp4"
            container = av.open(drone.get_video_stream())
        except av.AVError as ave:
            print(ave)
            print('retry...')

    # 起飞并等待稳定
    drone.takeoff()
    time.sleep(1)

    # 跳帧机制，启动时舍弃一定量的帧，防止视频读取延时
    frame_skip = 300

    # 实时监测部分
    while True:
        for frame_raw in container.decode(video=0):
            # 跳帧机制，防止视频读取延时
            if 0 < frame_skip:
                frame_skip = frame_skip - 1
                continue
            start_time = time.time()
            # frame = cv2.cvtColor(np.array(frame_raw.to_image()), cv2.COLOR_RGB2BGR)
            # 第一次运算检测并计算分辨率相关参数，后续不再计算
            if frame_idx == 0:
                frame = cv2.cvtColor(np.array(frame_raw.to_image()), cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                shape = np.shape(frame)
                x_resol = shape[1]
                y_resol = shape[0]
                d = x_resol / 2 / tantheta
                t_interval = 1 / 30
                xi = x_resol / 2 * v * t_interval / tantheta / d
                obst_thre = xi / (l_0 / v / t_interval - 1)
            else:
                frame = cv2.resize(cv2.cvtColor(np.array(frame_raw.to_image()), cv2.COLOR_RGB2BGR),
                                   (0, 0), fx=0.5, fy=0.5)
            # 转化为灰度图像
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 当检测到合格角点后进行光流跟踪
            if len(tracks) > 0:
                img0, img1 = prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                # 分别计算当前帧角点与前一帧角点的位置
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None,
                                                       **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None,
                                                        **lk_params)
                # 用角点回溯与前一帧实际角点的位置变化的值与阈值比较，过大则舍弃
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                new_tracks_len = []
                left_track = []
                right_track = []
                left_tr_len = []
                left_obst = 0
                right_tr_len = []
                right_obst = 0
                for tr, tr_len, (x, y), good_flag in zip(tracks, tracks_len, p1.reshape(-1, 2),
                                                         good):  # 将跟踪正确的点列入成功跟踪点
                    # 保留正确的角点
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    # 计算每个角点的光流向量的模长
                    tr_len.append(math.sqrt(
                        (tr[-1][0] - tr[-2][0]) ** 2
                        + (tr[-1][1] - tr[-2][1]) ** 2))
                    # 限制保留的旧角点及光流向量的模长的数量
                    if len(tr) > track_len_limit:
                        del tr[0]
                        del tr_len[0]
                    # 左分区的角点
                    if (100/2 < x < (x_resol / 2)) & (y < (y_resol - 300/2)):
                        left_track.append(tr)
                        left_tr_len.append(tr_len)
                        # 计算各个角点前五帧的平均光流向量的模长
                        left_tr_length = 0
                        for i in range(len(tr_len)):
                            left_tr_length = left_tr_length + tr_len[i]
                        left_tr_length = left_tr_length / 5 / abs(tr[-1][1] - x_resol / 2)
                        # 判断光流向量的模长是否大于阈值，并画圆
                        if left_tr_length > obst_thre:
                            left_obst = left_obst + 1
                            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                        else:
                            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                    # 右分区的角点
                    elif ((x_resol / 2) < x < (x_resol - 100/2)) & (y < (y_resol - 300/2)):
                        right_track.append(tr)
                        right_tr_len.append(tr_len)
                        # 计算各个角点前五帧的平均光流向量的模长
                        right_tr_length = 0
                        for i in range(len(tr_len)):
                            right_tr_length = right_tr_length + tr_len[i]
                        right_tr_length = right_tr_length / 5 / abs(tr[-1][1] - x_resol / 2)
                        # 判断光流向量的模长是否大于阈值，并画圆
                        if right_tr_length > obst_thre:
                            right_obst = right_obst + 1
                            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                        else:
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    new_tracks_len.append(tr_len)
                    new_tracks.append(tr)
                # 向右避障
                if left_obst - right_obst > 15:
                    command_right_count = command_right_count + 1
                    # print("left")
                    if command_right_count == 5:
                        drone.forward(0)
                        drone.right(40)
                        command_right_count = 0
                        command_left_count = 0
                        time.sleep(1)
                        drone.right(0)
                        time.sleep(0.5)
                        # frame_skip = 50
                # 向左避障
                elif right_obst - left_obst > 15:
                    command_left_count = command_left_count + 1
                    # print("right")
                    if command_left_count == 5:
                        drone.forward(0)
                        drone.left(40)
                        command_left_count = 0
                        command_right_count = 0
                        time.sleep(1)
                        drone.left(0)
                        time.sleep(0.5)
                        # frame_skip = 50
                # 前进
                else:
                    command_count = command_count + 1
                    # print("go")
                    # drone.forward(40)
                    if command_count == 20:
                        drone.forward(30)
                        command_count = 0

                tracks = new_tracks
                tracks_len = new_tracks_len
                # 以上一振角点为初始点，当前帧跟踪到的点为终点划线
                # cv2.polylines(frame, [np.int32(tr) for tr in left_track], False,
                #               (255, 0, 0))
                # cv2.polylines(frame, [np.int32(tr) for tr in right_track], False,
                #               (0, 255, 0))

            # 每5帧筛选一次特征点,用以提高计算速度，保证帧率
            if frame_idx % detect_interval == 0:
                # 新建和视频帧大小相同的空白图像蒙版
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                # 在角点所在像素画圆
                # for x, y in [np.int32(tr[-1]) for tr in tracks]:
                #     cv2.circle(mask, (x, y), 5, 0, -1)
                # 筛选合格的角点
                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)  # 像素级别角点检测
                # 将检测到合格的角点加入跟踪序列
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中
                        tracks_len.append([0])

            # 增加视频帧序列，传递灰度图像，可视化视频
            frame_idx += 1
            prev_gray = frame_gray
            cv2.imshow('track', frame)
            # 按q退出程序
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # 跳帧机制，防止视频读取延时
            if frame_raw.time_base < 1.0 / 60:
                time_base = 1.0 / 60
            else:
                time_base = frame_raw.time_base
            frame_skip = int((time.time() - start_time) / time_base)

except Exception as ex:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print(ex)
finally:
    drone.land()
    drone.quit()
    cv2.destroyAllWindows()
