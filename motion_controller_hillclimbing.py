import cv2
import numpy as np
import pyautogui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

pyautogui.FAILSAFE = False

# Initialize the webcam
cap = cv2.VideoCapture(0)

roi_punch = (0, 150, 450, 750)  # (top, bottom, left, right)
# Define the regions of interest (ROIs) for different actions
roi_jump = (0, 150, 0, 250)
roi_crouch = (300, 450, 0, 250)
roi_kick = (300, 450, 450, 750)

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Initialize state variables for key presses
punch_pressed = False
jump_pressed = False
crouch_pressed = False
kick_pressed = False

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
root.title("Motion Detection - Fighting Game")

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
    if detect_motion(roi_punch, frame):
        if not punch_pressed:
            pyautogui.keyDown(',')  # Simulate punch press
            punch_pressed = True
    else:
        if punch_pressed:
            pyautogui.keyUp(',')  # Simulate punch release
            punch_pressed = False

    if detect_motion(roi_jump, frame):
        if not jump_pressed:
            pyautogui.keyDown('up')  # Simulate jump press
            jump_pressed = True
    else:
        if jump_pressed:
            pyautogui.keyUp('up')  # Simulate jump release
            jump_pressed = False

    if detect_motion(roi_crouch, frame):
        if not crouch_pressed:
            pyautogui.keyDown('down')  # Simulate crouch press
            crouch_pressed = True
    else:
        if crouch_pressed:
            pyautogui.keyUp('down')  # Simulate crouch release
            crouch_pressed = False

    if detect_motion(roi_kick, frame):
        if not kick_pressed:
            pyautogui.keyDown('.')  # Simulate kick press
            kick_pressed = True
    else:
        if kick_pressed:
            pyautogui.keyUp('.')  # Simulate kick release
            kick_pressed = False

    # Draw the ROIs on the frame
    cv2.rectangle(frame, (roi_punch[2], roi_punch[0]), (roi_punch[3], roi_punch[1]), (0, 255, 0), 2)
    cv2.rectangle(frame, (roi_jump[2], roi_jump[0]), (roi_jump[3], roi_jump[1]), (255, 0, 0), 2)
    cv2.rectangle(frame, (roi_crouch[2], roi_crouch[0]), (roi_crouch[3], roi_crouch[1]), (0, 0, 255), 2)
    cv2.rectangle(frame, (roi_kick[2], roi_kick[0]), (roi_kick[3], roi_kick[1]), (255, 255, 0), 2)

    # Update the plot with the current frame
    update_plot(frame)
    canvas.draw()

    # Display the frame
    cv2.imshow('Motion Detection - Fighting Game', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Start the Tkinter main loop
root.mainloop()