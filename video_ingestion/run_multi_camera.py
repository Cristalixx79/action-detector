import cv2
import queue
import sys
import time

from ai_perception.ai_perception import PerceptionWorker
from video_ingestion import CameraWorker
import logging

logging.basicConfig(level=logging.INFO)

# ==============================
# НАСТРОЙКИ
# ==============================
CAMERAS = [
    {
        "camera_id": "Kitchen_1",
        "source": "C:/Users/BoillingMachine/PycharmProjects/Metrica/videos/gloves.mp4",  # 0 = вебкамера, либо путь к файлу "video.mp4"
    },
    {
        "camera_id": "Kitchen_2",
        "source": "C:/Users/BoillingMachine/PycharmProjects/Metrica/videos/kitchencam.mp4",  # 0 = вебкамера, либо путь к файлу "video.mp4"
    },
]

TARGET_FPS = 3
RESOLUTION = (1280, 720)

# ==============================
# ЗАПУСК
# ==============================
if __name__ == "__main__":
    frame_queue = queue.Queue(maxsize=32)
    out_queue = queue.Queue(maxsize=32)
    workers = []

    # --- Запуск video_ingestion ---
    for cam in CAMERAS:
        w = CameraWorker(
            camera_id=cam["camera_id"],
            source=cam["source"],
            out_queue=frame_queue,
            target_fps=TARGET_FPS,
            target_resolution=RESOLUTION,
            brightness_alpha=1.0,
            brightness_beta=0.0
        )
        w.daemon = True
        w.start()
        workers.append(w)
        print(f"[INFO] Started video ingestion for {cam['camera_id']}")
        time.sleep(1.5)

    # --- Запуск perception ---
    perception = PerceptionWorker(frame_queue, out_queue)
    perception.daemon = True
    perception.start()
    print("[INFO] Started AI perception module")

    CLASS_NAMES = {}
    print(f"[INFO] Loaded {len(CLASS_NAMES)} class names from YOLO")

    # ==============================
    # Основной цикл отображения
    # ==============================
    try:
        last_alive_check = time.time()

        while True:
            try:
                out = out_queue.get(timeout=2)
                print(f"[DEBUG] Got packet from Perception: keys={list(out.keys())}")
            except queue.Empty:
                # Проверка живости каждые 5 секунд
                if time.time() - last_alive_check > 5:
                    alive_cams = [w.is_alive() for w in workers]
                    alive_perc = perception.is_alive()
                    logging.info(f"[HEALTH] Cameras: {alive_cams}, Perception: {alive_perc}")

                    if not any(alive_cams):
                        logging.warning("[WARN] All CameraWorkers stopped! Restarting cameras...")
                        for cam in CAMERAS:
                            w = CameraWorker(
                                camera_id=cam["camera_id"],
                                source=cam["source"],
                                out_queue=frame_queue,
                                target_fps=TARGET_FPS,
                                target_resolution=RESOLUTION
                            )
                            w.daemon = True
                            w.start()
                            workers.append(w)
                    if not perception.is_alive():
                        logging.warning("[WARN] PerceptionWorker stopped! Restarting...")
                        perception = PerceptionWorker(frame_queue, out_queue)
                        perception.daemon = True
                        perception.start()

                    last_alive_check = time.time()
                continue


            frame = out.get("frame_raw")
            if frame is None:
                continue
            camera_id = out.get("camera_id", "Unknown")
            objects = out.get("objects", [])

            # Рисуем детекции
            for obj in objects:
                bbox = [int(v) for v in obj["bbox"]]
                x1, y1, x2, y2 = bbox
                cls_name = str(obj.get("class", "unknown"))
                conf = obj.get("confidence", 0.0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    except KeyboardInterrupt:
        print("[INFO] Stopping due to Ctrl+C")

    finally:
        print("[INFO] Stopping all workers...")
        for w in workers:
            w.stop()
        perception.stop()

        # Дождаться завершения потоков
        for w in workers:
            w.join(timeout=2.0)
        perception.join(timeout=2.0)

        print("[INFO] All stopped cleanly.")
        sys.exit(0)