import cv2
import sys
import mediapipe as mp
import math
import pyvirtualcam
import numpy as np
from PIL import Image


def add_transparent_overlay(background, overlay, pos=(0, 0), alpha=0.5, size_percent=0.1):
    h, w = background.shape[:2]
    overlay_w, overlay_h = overlay.size

    # size_percent를 이용해 오버레이 크기 조정
    new_overlay_size = (int(overlay_w * size_percent), int(overlay_h * size_percent))
    resize_overlay = overlay.resize(new_overlay_size, Image.BICUBIC)

    # 넘파이 배열로 변환
    np_overlay = np.array(resize_overlay)

    # RGBA 이미지를 RGB로 변환
    np_overlay = cv2.cvtColor(np_overlay, cv2.COLOR_RGBA2RGB)

    # 오버레이를 적용할 영역 선택
    y_pos = pos[1] if pos[1] >= 0 else h - new_overlay_size[1] + pos[1]
    x_pos = pos[0] if pos[0] >= 0 else w - new_overlay_size[0] + pos[0]

    # 배경 이미지에서 오버레이를 적용할 영역 선택
    roi = background[y_pos:y_pos + new_overlay_size[1], x_pos:x_pos + new_overlay_size[0]]

    # 오버레이 크기 조정
    resized_overlay = cv2.resize(np_overlay, (roi.shape[1], roi.shape[0]))

    # 오버레이 적용
    overlay_image = cv2.addWeighted(roi, 1, resized_overlay, alpha, 0)

    # 결과 이미지에 오버레이를 적용
    background[y_pos:y_pos + new_overlay_size[1], x_pos:x_pos + new_overlay_size[0]] = overlay_image

    return background


def distance(p1, p2):
    return math.dist((p1.x, p1.y), (p2.x, p2.y))


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera is not opened")
    sys.exit(1)

hands = mp_hands.Hands(max_num_hands=10)
with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
    while True:  # 무한 반복
        res, frame = cap.read()

        if not res:
            print("Camera error")
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(
                #     frame,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style(),
                # )

                points = hand_landmarks.landmark

                fingers = [0, 0, 0, 0, 0]

                if distance(points[4], points[9]) > distance(points[3], points[9]):
                    fingers[0] = 1

                for i in range(1, 5):
                    if distance(points[4 * (i + 1)], points[0]) > distance(
                            points[4 * (i + 1) - 1], points[0]
                    ):
                        fingers[i] = 1

                if fingers[0] == 1 and fingers[1:] == [0, 0, 0, 0] and points[4].y < points[0].y:
                    hand_shape = "thumbs up"
                    frame = add_transparent_overlay(frame, Image.open(f'./image/thumbs_up.png'), (int(points[4].x * frame.shape[1]), int(points[4].y * frame.shape[0])), 0.5)

                elif fingers[0] == 1 and fingers[1:] == [0, 0, 0, 0] and points[4].y > points[20].y:
                    hand_shape = "thumbs up"
                    frame = add_transparent_overlay(frame, Image.open(f'./image/thumbs_down.png'), (int(points[0].x * frame.shape[1]), int(points[0].y * frame.shape[0])), 0.5)

                elif distance(points[4], points[8]) < 0.1 and fingers[2:] == [
                    1,
                    1,
                    1,
                ]:  # 엄지 손가락과 검지 손가락이 닿아있고, 나머지 손가락 3개가 펴진 경우
                    hand_shape = "Ok"  # Ok
                    frame = add_transparent_overlay(frame, Image.open(f'./image/ok.png'), (int(points[20].x * frame.shape[1]), int(points[20].y * frame.shape[0])), 0.5, size_percent=1)

                else:  # 두 가지 모양이 아닌 경우
                    hand_shape = ""  # 내용을 출력하지 않음

        cv2.imshow("MediaPipe Hands", frame)  # 영상을 화면에 출력.

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam.send(frame)
        cam.sleep_until_next_frame()

        key = cv2.waitKey(5) & 0xFF  # 키보드 입력받기
        if key == 27:  # ESC를 눌렀을 경우
            break  # 반복문 종료

cv2.destroyAllWindows()  # 영상 창 닫기
cap.release()  # 비디오 캡처 객체 해제
