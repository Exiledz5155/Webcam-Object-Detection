import cv2
import time
from emailing import send_email
import glob
import os
from threading import Thread

video = cv2.VideoCapture(0)
# Time pause per frame in seconds
time.sleep(1) # Avoid black frames, gives time to load camera

first_frame = None
status_list = []
count = 1


def clean_folder():
    images = glob.glob("images/*.png")
    for image in images:
        os.remove(image)


while True:
    status = 0
    check, frame = video.read()
    # Convert frames to gray for less information processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Add a blur to reduce some sharpness
    gray_frame_gau = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Store the first frame for comparison
    if first_frame is None:
        first_frame = gray_frame_gau

    # Store the differences in a frame (white is the difference)
    delta_frame = cv2.absdiff(first_frame, gray_frame_gau)

    # Value of white that is 30 or higher we reassign to 255
    thresh_frame = cv2.threshold(delta_frame,
                                 80, 255,
                                 cv2.THRESH_BINARY)[1]
    # Processing to remove black spots within object
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)
    cv2.imshow("My video", dil_frame)

    # Process the contours of the object (a list)
    contours, check = cv2.findContours(dil_frame,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # If the area is too small, ignore it
        if cv2.contourArea(contour) < 10000:
            continue
        # Extract the position, width and height of the ®rectangle
        x, y, w, h = cv2.boundingRect(contour)
        # Display a rectangle over the original frame (by the corners)
        rectangle = cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 3)
        if rectangle.any:
            status = 1
            cv2.imwrite(f"images/{count}.png", frame)
            count = count + 1
            all_images = glob.glob("images/*.png")
            index = int(len(all_images) / 2)
            image_with_object = all_images[index]

    status_list.append(status)
    status_list = status_list[-2:] # Only the last 2 items
    print(status_list)

    # When the object exists the frame
    if status_list[0] == 1 and status_list[1] == 0:
        # Creating the thread to run send email function
        # We must introduce a comma here to show it's a tuple
        email_thread = Thread(target=send_email, args=(image_with_object, ))
        email_thread.daemon = True # Execute on the background

        # Creating the thread to run clean folder function
        clean_thread = Thread(target=clean_folder)
        email_thread.daemon = True

        email_thread.start()

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video.release()

clean_thread.start()
