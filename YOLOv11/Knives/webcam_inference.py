from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("knives_best1.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Plot results on frame
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLO Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()