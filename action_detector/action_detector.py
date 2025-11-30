import json
import time
import logging
import uuid
from collections import defaultdict
from pattern_analiser import MotionPatternAnalyzer

logging.basicConfig(level=logging.INFO)


class Camera:
    def __init__(self, name):
        self.name = name


class ActionDetector:
    def __init__(self, cams):
        self.__cameras = [Camera(cam) for cam in cams]
        # Список того, с чем может взаимодействовать человек, наверное что-то добавится в будущем
        self.__items = ["knife", "spoon", "desk", "plate", "food", "hat"]
        # Список камер, где в данный момент времени возможно действие
        self.__action_possible_cameras = []
        # минимальный уровень confidence, при котором идёт детекция действия
        self.__detection_threshold = 0.5

        self.__pattern_analyser = MotionPatternAnalyzer()

        # Предыдущая позиция объектов камеры
        # Пример: {"Kitchen_1": {"person_1": [243.64 -> центр по x, 534.65 -> центр по y, 1764438539.9258504 -> время, \
        # [322.6, 542.54, 12.7, 98.6] -> bbox], ...}, ...}
        self.__previous_position = defaultdict(dict)

        # Данные о движении объектов на данной камере
        # Пример: {"Kitchen_1": {"person_1": [[134.36 -> центр по x, 352.756 -> центр по y, \
        # 63.865 -> скорость по x, 524.754 -> скорость по y], ...], ...}, ...}
        self.__movement_vectors = defaultdict(dict)

        # Данные о действиях, замеченных на данной камере
        self.__detected_actions = defaultdict(dict)

        # Дефолтные значения
        for c in cams:
            self.__previous_position[c] = {}
            self.__movement_vectors[c] = {}
            self.__detected_actions[c] = {
                "timestamp": time.time(),
                "state": "IDLE",
                "action_detected": False,
                "action_type": "NONE",
                "timestamp_start": 0,
                "timestamp_end": 0
            }

    def is_action_possible(self, json_data):
        """Проверяет возможно ли действие"""
        camera_id = json_data["camera_id"]
        bit_flag = 1

        for item in json_data["objects"]:
            obj_class = item["class"]
            obj_confidence = item["confidence"]

            # Для того, чтоб действие было возможно, на камере должен быть человек (повар), рука и инструмент, \
            # с которым происходит взаимодействие
            if obj_class == "person" and obj_confidence > self.__detection_threshold:
                bit_flag = bit_flag << 1
            if (obj_class == "bare_hand" and obj_confidence > self.__detection_threshold) or \
                    (obj_class == "gloved_hand" and obj_confidence > self.__detection_threshold):
                bit_flag = bit_flag << 1
            if self.__check_items(obj_class):
                bit_flag = bit_flag << 1

        if bit_flag == 8:
            for c in range(len(self.__cameras)):
                if self.__cameras[c].name == camera_id:
                    self.__action_possible_cameras.append(camera_id)
                    print(f" -- Action is possible on {camera_id}")
            return True

        # очистка данных камеры в случае, если действие невозможно
        self.__clear_cam_data(camera_id)
        return False

    def analise_motion(self, data_json):
        """Составляет паттерны движения предметов"""
        camera_id = data_json["camera_id"]
        if not self.action_possible_on_cam(camera_id):
            self.__clear_cam_data(camera_id)
            print(f" -- Error with {camera_id}: \"Action is impossible\"")
            return None

        # Так как в self.movement_vectors[camera_id] добавляются данные о движении всех объектов на данной камере,
        # то нужно задать макс. значение длинны сохраняемой истории, чтоб не было переполнения
        if len(self.__movement_vectors[camera_id]) > 20:
            self.__movement_vectors[camera_id].pop(0)

        current_timestamp = time.time()
        item_list = []
        # Сюда добавляется информация о координатах центра объекта на камере
        center_position_list = []
        for item in data_json["objects"]:
            obj_class = item["class"]
            item_list.append(obj_class)
            # Проверка на наличие нескольких объектов одного и того же класса в кадре, чтоб программа могла их различать
            if obj_class in item_list:
                obj_class = obj_class + "_" + str(item_list.count(obj_class))

            bbox = item["bbox"]
            confidence = item["confidence"]
            if confidence < self.__detection_threshold:
                continue

            current_center = self.__calculate_bbox_center(bbox)
            movement_data = tuple()
            # Проверяем, есть ли предыдущие объекты того же класса
            if (camera_id in self.__previous_position and
                    obj_class in self.__previous_position[camera_id]):

                # Берём предыдущий объект
                previous_object = self.__previous_position[camera_id][obj_class]
                if previous_object:
                    prev_center_x, prev_center_y, prev_time, prev_bbox = previous_object
                    curr_center_x, curr_center_y = current_center

                    # Вычисляем вектор движения (dx, dy)
                    dx = curr_center_x - prev_center_x
                    dy = curr_center_y - prev_center_y

                    # Вычисляем скорость (пиксели в секунду)
                    time_diff = current_timestamp - prev_time
                    if time_diff > 0:
                        speed_x = dx / time_diff
                        speed_y = dy / time_diff
                    else:
                        speed_x, speed_y = 0, 0
                    movement_data = tuple([curr_center_x, curr_center_y, speed_x, speed_y])
                    center_position_list.append([(curr_center_x, curr_center_y), obj_class, camera_id])

            # Обновляем предыдущие позиции для этого класса объектов
            if camera_id not in self.__previous_position:
                self.__previous_position[camera_id] = {}
            if obj_class not in self.__previous_position[camera_id]:
                self.__previous_position[camera_id][obj_class] = []

            if obj_class not in self.__movement_vectors[camera_id]:
                self.__movement_vectors[camera_id][obj_class] = []

            # Заносим данные о текущей позиции для следующего цикла, где она будет выступать в роли предыдущей
            self.__previous_position[camera_id][obj_class] = (*current_center, current_timestamp, bbox)
            if len(movement_data) > 0:
                self.__movement_vectors[camera_id][obj_class].append(movement_data)

        patterns = self.__pattern_analyser.analyze_motion_patterns(self.__movement_vectors[camera_id])
        return patterns, center_position_list

    def detect_action(self, patterns, center_position_list, json_data):
        """Определяет тип движения (CUT, MIX, SERVE)"""
        camera_id = json_data["camera_id"]
        if (patterns is None) or (center_position_list is None):
            print(f" -- Action detection is impossible on {camera_id}")
            return None

        # Функция для определения действия, работает в трёх режимах в зависимости от состояния камеры \
        # (idle, action_candidate, action_active)
        def define_action(mode):
            for i in range(0, len(center_position_list)):
                for j in range(i + 1, len(center_position_list)):

                    # Чтобы действие было возможно, нужно, чтоб в кадре одновременно были как рука, так и предмет
                    if (self.__is_hand(center_position_list[i][1]) and self.__is_instrument(
                            center_position_list[j][1])) or \
                            (self.__is_hand(center_position_list[j][1]) and self.__is_instrument(
                                center_position_list[i][1])):

                        x1, y1 = center_position_list[i][0]
                        x2, y2 = center_position_list[j][0]

                        # Расстояние между центрами bbox руки и инструмента в пикселях
                        distance = ((abs(x2 - x1)) ** 2 + (abs(y2 - y1)) ** 2) ** 0.5
                        if distance > 50:
                            if mode == "IDLE":
                                pass
                            elif mode == "ACTION_CANDIDATE":
                                self.__detected_actions[camera_id]["timestamp"] = -1
                            elif mode == "ACTION_ACTIVE":
                                self.__detected_actions[camera_id]["timestamp"] = -1
                            continue

                        curr_time = time.time()
                        # Если в кадре нож и рука, и паттерн их движения вертикальный (вверх-вниз) или линейный
                        if (center_position_list[i][1] == "knife" or center_position_list[j][1] == "knife") and \
                                ((patterns["knife"] == "vertical" and patterns["gloved_hand"] == "vertical") or
                                 (patterns["knife"] == "linear" and patterns["gloved_hand"] == "linear")):
                            # Если условия выполнены, то засекаем время детекта действия для измерения его \
                            # продолжительности, действие считается активны, если продлилось хотя бы 0.5 секунд
                            if mode == "IDLE":
                                self.__detected_actions[camera_id] = {"timestamp": curr_time,
                                                                      "state": "ACTION_CANDIDATE",
                                                                      "action_detected": False, "action_type": "CUT"}
                            elif mode == "ACTION_CANDIDATE":
                                pass
                            # Если действие активно, то в качестве времени конца используем время последнего детекта,
                            # так как для завершения действия, он должен прерваться хотя бы на 0.4 секунды, пока действие
                            # активно, время конца будет постоянно обновляться и достичь разницы в 0.4 секунды не получится
                            elif mode == "ACTION_ACTIVE":
                                self.__detected_actions[camera_id]["timestamp_end"] = curr_time

                        # Если в кадре тарелка и рука, и рука имеет круговой движение над тарелкой
                        elif (center_position_list[i][1] == "plate" or center_position_list[j][1] == "plate") and \
                                (patterns["plate"] == "stationary" and patterns["gloved_hand"] == "circular"):
                            if mode == "IDLE":
                                self.__detected_actions[camera_id] = {"timestamp": curr_time,
                                                                      "state": "ACTION_CANDIDATE",
                                                                      "action_detected": False, "action_type": "MIX"}
                            elif mode == "ACTION_CANDIDATE":
                                pass
                            elif mode == "ACTION_ACTIVE":
                                self.__detected_actions[camera_id]["timestamp_end"] = curr_time

                        # Если в кадре тарелка и рука, и рука перемещает тарелку по линии
                        elif (center_position_list[i][1] == "plate" or center_position_list[j][1] == "plate") and \
                                (patterns["plate"] == "linear" and patterns["gloved_hand"] == "linear"):
                            if mode == "IDLE":
                                self.__detected_actions[camera_id] = {"timestamp": curr_time,
                                                                      "state": "ACTION_CANDIDATE",
                                                                      "action_detected": False, "action_type": "SERVE"}
                            elif mode == "ACTION_CANDIDATE":
                                pass
                            elif mode == "ACTION_ACTIVE":
                                self.__detected_actions[camera_id]["timestamp_end"] = curr_time
                        # Если действие не зафиксировалось
                        else:
                            if mode == "IDLE":
                                pass
                            elif mode == "ACTION_CANDIDATE":
                                self.__detected_actions[camera_id]["timestamp"] = -1
                            elif mode == "ACTION_ACTIVE":
                                self.__detected_actions[camera_id]["timestamp"] = -1

        current_state = self.__detected_actions[camera_id]["state"]
        # В данный момент на камере нет действия, но оно возможно
        if current_state == "IDLE":
            current_time = time.time()
            define_action(current_state)

            # Если действия нет на протяжении 30 секунд, проверяем возможно ли оно, так как детект мог пропасть \
            # или сотрудник ушёл из своей рабочей зоны
            if current_time - self.__detected_actions[camera_id]["timestamp"] >= 30:
                print(f" -- No action detected on {camera_id} in 30 seconds, checking again if action is possible")
                self.is_action_possible(json_data)

        # Камера зафиксировала действие, но нам нужно убедиться, что оно продлилось хотя бы 0.5 секунд
        elif current_state == "ACTION_CANDIDATE":
            current_time = time.time()
            define_action(current_state)

            if (self.__detected_actions[camera_id]["timestamp"] != -1) and \
                    (current_time - self.__detected_actions[camera_id]["timestamp"] >= 0.5):
                self.__detected_actions[camera_id]["action_detected"] = True
                self.__detected_actions[camera_id]["timestamp_start"] = current_time
                self.__detected_actions[camera_id]["state"] = "ACTION_ACTIVE"
            else:
                # Возвращаем в дефолтное состояние
                self.__detected_actions[camera_id] = {"timestamp": time.time(), "state": "IDLE", "action_detected": False,
                                                      "action_type": "NONE", "timestamp_start": 0, "timestamp_end": 0}
        # Чтоб действие прекратилось, нужно, чтоб прошло хотя бы 0.4 секунды, для уменьшения погрешности
        elif current_state == "ACTION_ACTIVE":
            current_time = time.time()
            if (self.__detected_actions[camera_id]["timestamp"] == -1) and \
                    (current_time - self.__detected_actions[camera_id]["timestamp_end"] >= 0.4):
                self.__detected_actions[camera_id]["action_detected"] = False
                self.__detected_actions[camera_id]["state"] = "IDLE"

                # Создаём выходной пакет
                output_packet = self.make_output_packet(camera_id)
                # Возвращаем в дефолтное состояние
                self.__detected_actions[camera_id] = {"timestamp": time.time(), "state": "IDLE", "action_detected": False,
                                                      "action_type": "NONE", "timestamp_start": 0, "timestamp_end": 0}
                return output_packet
            else:
                pass

        return None

    def make_output_packet(self, camera_id):
        packet_uuid = uuid.uuid4()
        # Формирование выходного пакета
        output_packet = {
            "action_id": packet_uuid,
            # Тут должно быть employee_id из ArcFace
            "employee_id": "undefined",
            "camera_id": camera_id,
            # Хз откуда брать это, возможно, при конфигурации приложения будет захардкожено
            "zone_id": "undefined",
            "action_type": self.__detected_actions[camera_id]["action_type"],
            "timestamp_start": self.__detected_actions[camera_id]["timestamp_start"],
            "timestamp_end": self.__detected_actions[camera_id]["timestamp_end"]
        }
        return output_packet

    def action_possible_on_cam(self, cam):
        """Проверяет возможно ли действие на данной камере"""
        return cam in self.__action_possible_cameras

    # =================== тут приватные функции =================== #
    def __clear_cam_data(self, camera_id):
        if camera_id in self.__action_possible_cameras:
            self.__action_possible_cameras.remove(camera_id)
            if camera_id in self.__movement_vectors:
                self.__movement_vectors[camera_id] = {}
            if camera_id in self.__previous_position:
                self.__previous_position[camera_id] = {}

    def __get_index(self, cam):
        """Находит индекс камеры по её названию"""
        for c in range(len(self.__cameras)):
            if self.__cameras[c].name == cam:
                return c
        return -1

    def __is_instrument(self, item):
        """Проверяет на инструмент"""
        for i in self.__items:
            if str(item).startswith(i):
                return True
        return False

    def __is_hand(self, item):
        """Проверяет на руку"""
        return (str(item).startswith("bare_hand")) or (str(item).startswith("gloved_hand"))

    def __calculate_bbox_center(self, bbox):
        """Вычисляет центр bounding box"""
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def __calculate_bbox_area(self, bbox):
        """Вычисляет площадь bounding box"""
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def __check_items(self, item):
        """Проверяет есть ли в кадре нужные для потенциального действия объекты"""
        if item not in self.__items:
            return False
        return True
