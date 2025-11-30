Загрузил только то, что поменял, иерархия проекта не поменялась, из нового только модуль action_detector
Чтобы запустить action_detector, нужно установить fastapi и uvicorn, зайти а папку action_detector и в терминале прописать `uvicorn data_capture:app --reload`
Откроется HTTP-соединение на `127.0.0.1:8000/api/data`
