import json

from action_detector import ActionDetector

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse

app = FastAPI()

# Список камер, должен совпадать со списком из video_injection, чтоб их названия совпадали
cameras = ["Kitchen_1", "Kitchen_2"]
detector = ActionDetector(cameras)


@app.post("/api/data")
def get_data(data = Body(...)):
    json_data = json.JSONDecoder().decode(data)
    camera_id = json_data["camera_id"]

    # Проверка на то, находится ли камера в списке, камер, где возможно действие
    if detector.action_possible_on_cam(camera_id):
        cam_data = detector.analise_motion(json_data)
        if cam_data is not None:
            pattern, distance_list = cam_data
            detector.detect_action(pattern, distance_list, json_data)
    else:
        detector.is_action_possible(json_data)

    return JSONResponse(status_code=200, content={})
