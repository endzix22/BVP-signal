# rozpoznawanie twarzy
# baza danych pure
# nastepnie z kanalow:r,g,b mamy osobne sygnaly
# nastepnie nalozenie ich razem i wykreslenie sygnalu wraz z preprocessingiem

"""
CLICK ENTER TO STOP THE PROGRAM FROM RUNNING
pip install dlib
pip install face-recognition or if anaconda: conda install -c conda-forge
"""

import numpy as np
import time
from matplotlib import pyplot as plt
import cv2
import face_recognition  # API to detect face

cap = cv2.VideoCapture(0)
face_locations = []
heartbeat_count = 128

# declaration of variables for charts
heartbeat_times = [time.time()] * heartbeat_count
rData = [0] * heartbeat_count
gData = [0] * heartbeat_count
bData = [0] * heartbeat_count
allData = [0] * heartbeat_count
figR = plt.figure()
axR = figR.add_subplot(111)
figG = plt.figure()
axG = figG.add_subplot(111)
figB = plt.figure()
axB = figB.add_subplot(111)
figall = plt.figure()
axall = figall.add_subplot(111)

while True:
    # ONE VIDEO FRAME (splitting video into frames)
    ret, frame = cap.read()
    # Convert the image from BGR color (which OpenCV uses) to RGB
    # color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # kolor, grubosc rectangle

        # face object cropping
        crop_img = frame[top:bottom, left:right]

        #crop_img = crop_img.squeeze()
        # taking the average for each channel separately (r,g,b)
        avgR = np.mean(crop_img[:, :, 2])
        avgG = np.mean(crop_img[:, :, 1])
        avgB = np.mean(crop_img[:, :, 0])
        """
        Once we have the camera stream, it’s pretty simple. For the selected image fragment, 
        we get the average brightness value and add it to the array along with the measurement timestamp.
        """
        # TODO 1: MOŻE WARTO DODAC FREEZE IMG SIZE
        cv2.imshow("ROI", crop_img)
        ##### PLOTTING CHARTS
        # R
        rData = rData[1:] + [avgR]
        heartbeat_times = heartbeat_times[1:] + [time.time()]
        axR.plot(heartbeat_times, rData)
        figR.canvas.draw()
        plot_img_npR = np.fromstring(figR.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        plot_img_npR = plot_img_npR.reshape(figR.canvas.get_width_height()[::-1] + (3,))
        plt.cla()

        # G
        gData = gData[1:] + [avgG]
        axG.plot(heartbeat_times, gData)
        figG.canvas.draw()
        plot_img_npG = np.fromstring(figG.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        plot_img_npG = plot_img_npG.reshape(figG.canvas.get_width_height()[::-1] + (3,))
        plt.cla()

        # B
        bData = bData[1:] + [avgB]
        axB.plot(heartbeat_times, bData)
        figB.canvas.draw()
        plot_img_npB = np.fromstring(figB.canvas.tostring_rgb(),
                                     dtype=np.uint8, sep='')
        plot_img_npB = plot_img_npB.reshape(figB.canvas.get_width_height()[::-1] + (3,))
        plt.cla()

        # RGB TOGETHER
        allData = allData[1:] + [(np.average(crop_img))]

        # TODO 2: DODAJ NA ZMIENNEJ allData preprocessing
        """
        z literatury: normalizacja wartosci srednich pikseli,
        #filtr sredniej ruchomej 
        #fft na sygnale rppg



        """
        # TODO 3: SPRAWDZ Z LITERATURY JAKIE TECHNIKI PRZETWARZANIA SYGNALU ZASTOSOWAC
        axall.plot(heartbeat_times, allData)
        figall.canvas.draw()
        plot_img_npall = np.fromstring(figall.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        plot_img_npall = plot_img_npall.reshape(figall.canvas.get_width_height()[::-1] + (3,))
        # changing the color of the chart
        # plot_img_npall = cv2.cvtColor(plot_img_npall, cv2.COLOR_BGR2HSV )
        plt.cla()

        final = cv2.hconcat((plot_img_npR, plot_img_npG, plot_img_npG))
        cv2.imshow("R,G,B", final)
        cv2.imshow("RGB together", plot_img_npall)

    # Wait for Enter key to stop
    if cv2.waitKey(25) == 13:
        break

video_capture.release()
cv2.destroyAllWindows()