# Face-Swapper-with-InsightFace-Python-Library
Python script using InsightFace library for face detection, cropping, and swapping features, creating fun face-swapped images.


The code accomplishes:

1. **Face Detection**:
    - The code initializes the FaceAnalysis class from the InsightFace library, setting it up for face detection.
    - It loads an input image (in this case, identified as 't1') and displays it.
    - It detects faces in the input image using the InsightFace model and draws bounding boxes around the detected faces.
    - The annotated image with bounding boxes is saved as "t1_output.jpg."
2. **Face Cropping and Display**:
    - The code crops and displays each detected face individually, showcasing the detected faces' regions of interest.
3. **Face Swapping**:
    - The code loads a pre-trained face-swapping model from a specified path ('inswapper_128.onnx').
    - It selects a source face (in this case, the second detected face) as the reference for swapping.
    - It iterates through all detected faces and performs face swapping using the source face as a reference.
    - The result of the face swapping is stored in the 'res' variable.
4. **Displaying the Swapped Faces**:
    - The code displays the image with the swapped faces, allowing you to see the modified image with the swapped facial features.


The code effectively demonstrates how to use the InsightFace library for face detection and swapping in Python, providing an entertaining and illustrative example of face manipulation with deep learning models.

We are swapping all the faces in the image with a specific face (Chandler's face) already in the image.

Figures that are shown in order when the program is run: original image, cropped faces, cropped face to swap with and final swapped image:
![image](https://github.com/rshwati/Face-Swapper-with-InsightFace-Python-Library/assets/136934368/cc7cca71-e4a0-4d19-a9ae-c6ef4d6c98bf)
![image](https://github.com/rshwati/Face-Swapper-with-InsightFace-Python-Library/assets/136934368/691b8449-5c29-4e35-a056-ad31854f09ce)
![image](https://github.com/rshwati/Face-Swapper-with-InsightFace-Python-Library/assets/136934368/764640e8-7e72-4c1a-9b4c-3a53142e7d19)
<img width="465" alt="Untitled" src="https://github.com/rshwati/Face-Swapper-with-InsightFace-Python-Library/assets/136934368/7e370405-dd35-49ad-b520-fcb5c36028a3">


