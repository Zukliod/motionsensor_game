import cv2
import numpy as np
import pyautogui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the regions of interest (ROIs) for different actions
roi_gas = (100, 300, 150, 450)  # (top, bottom, left, right)
roi_brake = (300, 500, 150, 450)

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

def detect_motion(roi, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    roi_frame = fgmask[roi[0]:roi[1], roi[2]:roi[3]]
    contours, _ = cv2.findContours(roi_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:  # Adjust the threshold as needed
            return True
    return False

def update_plot(frame):
    plt.clf()
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.draw()

# Create the main window
root = tk.Tk()
root.title("Motion Detection - Hill Climb Racing")

# Create a figure for matplotlib
fig, ax = plt.subplots()

# Create a canvas for matplotlib
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect motion in each ROI
    if detect_motion(roi_gas, frame):
        pyautogui.press('right')  # Simulate gas
    if detect_motion(roi_brake, frame):
        pyautogui.press('left')  # Simulate brake

    # Draw the ROIs on the frame
    cv2.rectangle(frame, (roi_gas[2], roi_gas[0]), (roi_gas[3], roi_gas[1]), (0, 255, 0), 2)
    cv2.rectangle(frame, (roi_brake[2], roi_brake[0]), (roi_brake[3], roi_brake[1]), (255, 0, 0), 2)

    # Update the plot with the current frame
    update_plot(frame)
    canvas.draw()

    # Display the frame
    cv2.imshow('Motion Detection - Hill Climb Racing', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Start the Tkinter main loop
root.mainloop()