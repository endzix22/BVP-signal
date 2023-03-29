# Robust BVP from facial videos
This repo monitors real time cardiac activities of a person through remote photoplethysmography(rPPG) without any physical contact with sensor, by detecing blood volume pulse induced subtle color changes from video stream through webcam sensor or a video file.

# Signal extraction
The first stage is the detection of ROI. The API face recognition was used for this.
Then the detected face area is cut out. 
Next the values are averaged separately for each of the R,G,B channels for each video frame.