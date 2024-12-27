import cv2
from ultralytics import YOLO

# Load the model
model_path = r"C:\\Users\\Faisal\\Downloads\\Internship summer\\runs\\pose\\train52\\weights\\last.pt"
model = YOLO(model_path)

# Load the image using OpenCV
image_path = r"C:\\Users\\Faisal\\Downloads\\Internship summer 2024\\August new images w updates from disi\\Data 9\\Data\\images\\val\\ezgif-frame-014 - Copy.jpg"
image = cv2.imread(image_path)

# Perform inference with a lower confidence threshold if necessary
results = model(image, conf=0.25)

# Process results and visualize keypoints
for result in results:
    if result.keypoints is not None:
        keypoints = result.keypoints.xy.cpu().numpy()  # Access the keypoints directly

        # Debugging by print keypoints shape and type
        print("Keypoints shape:", keypoints.shape)
        print("Keypoints content:", keypoints)

        for keypoint_indx in range(keypoints.shape[1]):
            # Access keypoint coordinates
            x, y = keypoints[0, keypoint_indx]

            print(f"Keypoint {keypoint_indx}: x={x}, y={y}")

            # Draw keypoint on the image
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)  # Green circle for keypoints

            # Draw keypoint number
            cv2.putText(image, str(keypoint_indx), (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1, cv2.LINE_AA)  # Red text for numbering

    else:
        print("No keypoints detected")

# Show the image with keypoints
cv2.imshow("Keypoints", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


