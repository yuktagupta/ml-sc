from ultralytics import YOLO 
import cv2 
# Load the YOLO model 
model = YOLO("C:/Users/Admin/Desktop/Yolo-Weights/yolov8n.pt")

# Process the image without automatically showing it 
results = model("C:\\Users\\Admin\\OneDrive\\Desktop\\pract\\dog.jpg")   
# If results is a list, access the first element (which should contain the image) 
image = results[0].plot()  # Plot the results (draw bounding boxes, etc.) 
# Resize the image to a suitable size before displaying 
resized_image = cv2.resize(image, (800, 800))  # Adjust 800x800 to your preferred size 
# Create the OpenCV window with a normal resizing option 
cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL) 
# Resize the window to match the size of the image 
cv2.resizeWindow("Processed Image", resized_image.shape[1], resized_image.shape[0]) 
# Display the resized image in the window 
cv2.imshow("Processed Image", resized_image) 
# Wait for a key press and close the window 
cv2.waitKey(0) 
cv2.destroyAllWindows()