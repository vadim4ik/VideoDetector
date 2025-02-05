import cv2
from ultralytics import YOLO

# Завантаження моделі YOLOv8
model = YOLO("yolov8n.pt")  # Модель буде завантажена автоматично

# Відкриваємо відеофайл
video_path = "video.mp4"
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(video_path)

# Відеозапис вихідного файлу
output_path = "output.avi"
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Вихід, якщо відео закінчилося

    # Запускаємо розпізнавання
    results = model(frame)

    # Малюємо об'єкти
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координати
            conf = box.conf[0]  # Ймовірність
            cls = int(box.cls[0])  # Клас об'єкта
            label = f"{model.names[cls]} {conf:.2f}"

            # Малюємо рамку та підпис
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Запис у файл
    out.write(frame)

    # Показуємо відео у вікні
    cv2.imshow("YOLOv8 Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Натисніть "q" для виходу
        break

cap.release()
out.release()
cv2.destroyAllWindows()
