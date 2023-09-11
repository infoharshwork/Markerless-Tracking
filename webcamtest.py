import cv2
import numpy as np

# Function to detect squares in the frame
def detect_squares(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_squares = []

    # Loop over the contours
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon has 4 vertices (a square)
        if len(approx) == 4:
            # Ensure the area of the square is not too small
            area = cv2.contourArea(contour)
            if area > 100:
                detected_squares.append(approx)

    return detected_squares

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect squares in the frame
    squares = detect_squares(frame)

    # Draw squares on the frame
    for square in squares:
        cv2.drawContours(frame, [square], -1, (0, 255, 0), 2)

    # Display the frame with detected squares
    cv2.imshow("Square Detection", frame)

    # Check for user input to exit the program (press 'q')
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
