import cv2
from ultralytics import YOLO

# Load YOLOv8 model - first time it will auto-download yolov8n.pt
model = YOLO("models/yolov8n-face.pt")  # Only detects faces


def detect_faces_yolo(frame, conf_threshold=0.3):
    """
    Detects faces in the frame using YOLOv8.
    Returns the modified frame with boxes and the face coordinates.
    """
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)
    faces = []

# Filter by class name
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == 0:  # class 0 = person
                x1, y1, x2, y2 = map(int, box[:4])
                faces.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, faces

# Run this file directly for webcam face test
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    print("[INFO] Starting webcam...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, faces = detect_faces_yolo(frame)

        cv2.imshow("YOLO Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
