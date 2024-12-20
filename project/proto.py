import cv2
import math
import numpy as np
from ultralytics import YOLO
from collections import deque

# モデル読み込み
object_model = YOLO('../models/yolov8n.pt')  
hand_model = YOLO('../models/yolov8nHandOnly.pt')  

cap = cv2.VideoCapture(1)

# 過去フレーム情報を保持するdeque（最大6フレーム）
prev_hand_centers = deque(maxlen=10)
prev_obj_centers = deque(maxlen=10)

prev_holding_state = False
holding_frame_count = 0

# 閾値設定例
dist_threshold = 50.0       # 手と物体の中心間距離閾値
cosine_threshold = 0.7      # ベクトル類似度閾値
hold_confirm_frames = 1     # このフレーム数連続で条件を満たせば「持っている」と判定
release_confirm_frames = 9  # 解除のための連続フレーム数

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 物体検出（マウス(64)とボトル(39)のみに限定）
    object_results = object_model(frame, conf=0.5, classes=[39,64])  
    # 手検出
    hand_results = hand_model(frame, conf=0.5)

    # YOLO結果の抽出
    objects = []
    if len(object_results) > 0:
        for box in object_results[0].boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            # 上でclasses=[39,64]としたためcls_idは39か64のみ
            objects.append((x_min, y_min, x_max, y_max, conf, cls_id))

    hands = []
    if len(hand_results) > 0:
        for box in hand_results[0].boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            hands.append((x_min, y_min, x_max, y_max, conf, cls_id))

    # 手領域の平均座標を求める
    hand_center = None
    if len(hands) > 0:
        centers = []
        for (hx1, hy1, hx2, hy2, hconf, hcls) in hands:
            cx = (hx1 + hx2) / 2.0
            cy = (hy1 + hy2) / 2.0
            centers.append((cx, cy))
        if len(centers) > 0:
            mean_x = sum(c[0] for c in centers) / len(centers)
            mean_y = sum(c[1] for c in centers) / len(centers)
            hand_center = (mean_x, mean_y)

    # 物体中心座標（ここでは単純に最初の物体を対象にする）
    obj_center = None
    if len(objects) > 0:
        ox1, oy1, ox2, oy2, oconf, ocls = objects[0]
        ocx = (ox1 + ox2) / 2.0
        ocy = (oy1 + oy2) / 2.0
        obj_center = (ocx, ocy, ocls)

    # 現フレームの情報を追加
    if hand_center is not None:
        prev_hand_centers.append(hand_center)
    else:
        prev_hand_centers.append(None)

    if obj_center is not None:
        prev_obj_centers.append(obj_center)
    else:
        prev_obj_centers.append(None)

    holding_candidate = False

    # ベクトル計算は過去6フレーム分溜まってから行う
    if len(prev_hand_centers) > 1 and len(prev_obj_centers) > 1:
        valid_hand_positions = [pos for pos in prev_hand_centers if pos is not None]
        valid_obj_positions = [pos for pos in prev_obj_centers if pos is not None]

        if len(valid_hand_positions) > 1 and len(valid_obj_positions) > 1:
            # 最も古い有効な位置と最新位置を用いてベクトル計算
            oldest_hand = valid_hand_positions[0]
            newest_hand = valid_hand_positions[-1]

            oldest_obj = valid_obj_positions[0]
            newest_obj = valid_obj_positions[-1]

            hand_dx = newest_hand[0] - oldest_hand[0]
            hand_dy = newest_hand[1] - oldest_hand[1]

            obj_dx = newest_obj[0] - oldest_obj[0]
            obj_dy = newest_obj[1] - oldest_obj[1]

            if hand_center is not None and obj_center is not None:
                (ocx, ocy, ocls) = obj_center
                dist = math.sqrt((hand_center[0] - ocx)**2 + (hand_center[1] - ocy)**2)

                hand_vec_norm = math.sqrt(hand_dx**2 + hand_dy**2)
                obj_vec_norm = math.sqrt(obj_dx**2 + obj_dy**2)

                if hand_vec_norm > 0 and obj_vec_norm > 0:
                    cos_sim = (hand_dx * obj_dx + hand_dy * obj_dy) / (hand_vec_norm * obj_vec_norm)
                else:
                    cos_sim = 0.0

                if dist < dist_threshold and cos_sim > cosine_threshold:
                    holding_candidate = True

    # 状態管理
    if holding_candidate:
        holding_frame_count += 1
    else:
        holding_frame_count -= 1
        if holding_frame_count < 0:
            holding_frame_count = 0

    # ホールド状態判定
    if not prev_holding_state and holding_frame_count >= hold_confirm_frames:
        prev_holding_state = True
    elif prev_holding_state and holding_frame_count <= release_confirm_frames:
        prev_holding_state = False

    # 描画
    if hand_center is not None:
        cv2.circle(frame, (int(hand_center[0]), int(hand_center[1])), 5, (0,255,0), -1)
    if obj_center is not None:
        cv2.circle(frame, (int(obj_center[0]), int(obj_center[1])), 5, (255,0,0), -1)

    text = "Holding: " + ("Yes" if prev_holding_state else "No")
    cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
