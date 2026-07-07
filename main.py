# Управление дроном Tello по жестам и позе человека

from ultralytics import YOLO
from djitellopy import Tello
import cv2
import time
import logging

# === НАСТРОЙКИ ===
DEAD_ZONE_X = 150  # Мёртвая зона по X (px)
DEAD_ZONE_Y = 100  # Мёртвая зона по Y (px)

# Индексы ключевых точек (COCO)
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

# Классы жестов
GESTURE_CLASSES = ["neutral", "go_up", "go_down", "go_left", "go_right"]
NEUTRAL_GESTURE = "neutral"
TAKEOFF_GESTURE = "go_up"
LAND_GESTURE = "go_down"
LEFT_GESTURE = "go_left"
RIGHT_GESTURE = "go_right"

# Минимальная уверенность для распознавания жеста
GESTURE_CONF_THRESH = 0.7

# Загрузка моделей
POSE_MODEL = YOLO("yolov8n-pose.pt")  # Детекция позы
GESTURE_MODEL = YOLO("best.pt")  # Распознавание жестов

# Настройка логирования
logging.getLogger("djitellopy").setLevel(logging.WARNING)
fly = Tello()

# === ФУНКЦИИ ===


def wait_for_valid_frame(frame_read):
    """Ждёт первый валидный кадр с дрона."""
    print("Ожидание первого кадра с дрона...")
    while True:
        frame = frame_read.frame
        if frame is not None and frame.size > 0:
            print("Первый кадр получен.")
            return frame
        time.sleep(0.1)


def check_gesture(frame):
    """
    Анализирует кадр и возвращает распознанный жест.
    Возвращает: 'go_up', 'go_down', 'neutral'
    """
    results = GESTURE_MODEL(frame, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return NEUTRAL_GESTURE

    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy()

    max_conf_idx = confs.argmax()
    if confs[max_conf_idx] >= GESTURE_CONF_THRESH:
        cls_id = int(clss[max_conf_idx])
        return GESTURE_CLASSES[cls_id]

    return NEUTRAL_GESTURE


def calculate_body_center(keypoints):
    """Вычисляет центр тела по плечам и бёдрам."""
    points = [
        keypoints[i] for i in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    ]
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return int(sum(xs) / 4), int(sum(ys) / 4)


def calculate_body_area(keypoints):
    """Вычисляет площадь четырёхугольника по ключевым точкам."""
    x1, y1 = keypoints[LEFT_SHOULDER]
    x2, y2 = keypoints[RIGHT_SHOULDER]
    x3, y3 = keypoints[RIGHT_HIP]
    x4, y4 = keypoints[LEFT_HIP]
    area = (
        abs(
            (x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1)
            - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1)
        )
        / 2
    )
    return int(area)


def select_target_person(keypoints_list, frame_width):
    """Выбирает самого крупного человека около центра кадра."""
    biggest_area = 0
    best_center = None

    for kpts in keypoints_list:
        if len(kpts) < 13:
            continue
        try:
            area = calculate_body_area(kpts)
            cx_body, cy_body = calculate_body_center(kpts)
            if area < 5000 or abs(cx_body - frame_width // 2) > frame_width * 0.4:
                continue
            if area > biggest_area * 1.2:
                biggest_area = area
                best_center = (cx_body, cy_body)
        except Exception as e:
            continue

    return best_center, biggest_area


def get_command(body_x, body_y, area, frame_center_x, frame_center_y):
    """Формирует команду управления дроном."""
    if body_x > frame_center_x + DEAD_ZONE_X:
        fly.send_rc_control(0, 0, 0, 60)  # Поворот направо
        return "GO RIGHT"
    elif body_x < frame_center_x - DEAD_ZONE_X:
        fly.send_rc_control(0, 0, 0, -60)  # Поворот налево
        return "GO LEFT"
    elif body_y > frame_center_y + DEAD_ZONE_Y:
        fly.send_rc_control(0, 0, -60, 0)  # Вниз
        return "GO DOWN"
    elif body_y < frame_center_y - DEAD_ZONE_Y:
        fly.send_rc_control(0, 0, 60, 0)  # Вверх
        return "GO UP"
    elif area > 30000:
        fly.send_rc_control(0, -40, 0, 0)  # Назад
        return "GO BACK"
    elif area < 7000:
        fly.send_rc_control(0, 40, 0, 0)  # Вперёд
        return "GO FORWARD"
    else:
        fly.send_rc_control(0, 0, 0, 0)  # Стоп
        return "STOP"


def annotate_frame(frame, center, area, command, frame_center, gesture):
    """Добавляет аннотации на кадр."""
    h, w = frame.shape[:2]

    # Центр кадра
    cv2.circle(frame, frame_center, 8, (0, 255, 0), -1)

    # Центр тела
    if center is not None:
        cv2.circle(frame, center, 10, (0, 0, 255), -1)
        cv2.putText(
            frame,
            "Body Center",
            (center[0] + 15, center[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Area: {area}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    # Команда и жест
    cv2.putText(
        frame,
        f"Cmd: {command}",
        (w - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (10, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )


def process_frame(frame):
    """Обработка одного кадра: поза → выбор цели → команда → аннотация."""
    my_frame = cv2.resize(frame, (640, 480))
    h, w = my_frame.shape[:2]
    frame_center = (w // 2, h // 2)

    # Детекция позы
    results = POSE_MODEL(my_frame, verbose=False)
    result = results[0]
    keypoints_list = (
        result.keypoints.xy.cpu().tolist() if result.keypoints is not None else []
    )
    annotated_frame = result.plot(boxes=False)  # Только скелеты

    # Выбор цели
    body_center, body_area = select_target_person(keypoints_list, w)

    # Жест
    gesture = check_gesture(my_frame)

    # Влево/вправо
    get_right_left(gesture)

    # Команда
    command = (
        get_command(*body_center, body_area, *frame_center) if body_center else "STOP"
    )

    # Аннотация
    annotate_frame(
        annotated_frame, body_center, body_area, command, frame_center, gesture
    )

    return annotated_frame, command, gesture


def wait_for_takeoff(frame_read):
    """Ждёт жест 'go_up' для взлёта."""
    print("Ожидание жеста для взлёта (поднимите руки — 'go_up')...")
    while True:
        frame = frame_read.frame
        if frame is None or frame.size == 0:
            continue

        my_frame = cv2.resize(frame, (640, 480))
        gesture = check_gesture(my_frame)

        # Отладочное отображение
        cv2.putText(
            my_frame,
            "Status: Waiting TAKEOFF",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            my_frame,
            f"Gesture: {gesture}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.imshow("Pose Tracking", cv2.cvtColor(my_frame, cv2.COLOR_BGR2RGB))

        if gesture == TAKEOFF_GESTURE:
            print("Жест 'TAKEOFF' распознан. Взлёт!")
            fly.takeoff()
            time.sleep(2)
            # cv2.waitKey(500)
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Выход без взлёта.")
            fly.end()
            exit(0)


def get_right_left(gesture):
    if gesture == LEFT_GESTURE:
        fly.send_rc_control(30, 0, 0, 0)
        # time.sleep(2)
    elif gesture == RIGHT_GESTURE:
        fly.send_rc_control(-30, 0, 0, 0)
        # time.sleep(2)
    # else:
    #     fly.send_rc_control(0, 0, 0, 0)


def drone_init():
    """Инициализация дрона."""
    fly.connect()
    print(f"Заряд батареи: {fly.get_battery()}%")

    fly.set_video_fps(Tello.FPS_15)
    fly.set_video_bitrate(Tello.BITRATE_AUTO)

    fly.streamoff()
    time.sleep(1)
    fly.streamon()
    time.sleep(2)

    fly.send_rc_control(0, 0, 0, 0)  # Остановить все движения


def main():
    drone_init()
    frame_read = fly.get_frame_read()

    wait_for_valid_frame(frame_read)
    time.sleep(1)

    wait_for_takeoff(frame_read)

    print("Дрон в воздухе. Управление запущено.")

    try:
        while True:
            frame = frame_read.frame
            if frame is None or frame.size == 0:
                print("Пустой кадр — пропуск...")
                continue

            annotated_frame, command, gesture = process_frame(frame)
            print(f"Command: {command}, Gesture: {gesture}")

            # Посадка по жесту
            if gesture == LAND_GESTURE:
                print("Жест 'LAND' распознан. Посадка.")
                cv2.putText(
                    annotated_frame,
                    "LANDING...",
                    (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )
                # cv2.imshow("Pose Tracking", annotated_frame)
                cv2.imshow(
                    "Pose Tracking", cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                )
                cv2.waitKey(1000)
                fly.land()
                break

            cv2.imshow(
                "Pose Tracking", cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nПрерывание по Ctrl+C.")
    finally:
        fly.send_rc_control(0, 0, 0, 0)
        fly.streamoff()
        fly.end()
        cv2.destroyAllWindows()
        print("Полёт завершён.")


if __name__ == "__main__":
    main()
