import cv2

from ultralytics import YOLO
from workout_monitor import WorkoutMonitor

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture("videos//test_pushup.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

workout_monitor = WorkoutMonitor(
    line_thickness=2,
    view_img=True,
    pose_type="pushup",
    kpts_to_check=[6, 8, 10],
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    results = model.track(im0, verbose=False)  # Tracking recommended
    im0 = workout_monitor.start_counting(im0, results)

cv2.destroyAllWindows()