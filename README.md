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
