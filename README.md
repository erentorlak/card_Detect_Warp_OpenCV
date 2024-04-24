# OpenCV Quadrilateral Detection and Perspective Transformation

This project is a computer vision application using OpenCV to detect cards (Quadrilaterals), in a real-time video stream from a webcam.

## Features

- Adaptive thresholding that is robust in different lightning conditions.
- Perspective transformation of detected quadrilaterals.
- Display of processed frames with highlighted polygons.
- Output of birds-eye view of detected objects.


## How It Works

1. **Video Capture**:
   - The application starts by initializing a connection to the default webcam. Frames are captured in real-time for processing.

2. **Grayscale Conversion and Blurring**:
   - Each captured frame is first converted to grayscale. Grayscale simplifies the image data by eliminating color variance, focusing processing on intensity variance.
   - A Gaussian blur is then applied to the grayscale image. This step reduces image noise and detail by smoothing the image, which is particularly useful to prepare for edge detection.

3. **Adaptive Thresholding**:
   - After blurring, the program applies adaptive thresholding to the image. Unlike simple thresholding that uses a global threshold value, adaptive thresholding calculates thresholds for smaller regions, allowing for variations in lighting conditions across the image. This results in a binary image where the foreground (potential polygons) is separated from the background.

4. **Contour Detection**:
   - The binary image is then used to detect contours using OpenCV's `findContours` function. This function retrieves all the contours from the binary image using an external retrieval mode, which only captures the outermost contours.

5. **Contour Filtering**:
   - Not all detected contours are relevant; some may be noise or irrelevant details. The contours are filtered based on their area to exclude very small contours that are unlikely to be the target polygons.

6. **Morphological Operations**:
   - To refine the contours, morphological operations such as erosion and dilation are applied:
     - **Erosion**: This operation reduces the size of foreground objects and is used to eliminate small white noise and detach connected objects.
     - **Dilation**: After eroding, dilation is applied to restore the object size and to fill in gaps in the contours, improving the continuity of detected shapes.

7. **Polygon Approximation**:
   - For each filtered contour, the program uses `approxPolyDP` to approximate the contour to a polygon. This function reduces the number of points in a contour while maintaining its shape, allowing for an accurate and simplified representation of the contour. The approximation is tuned by specifying an epsilon value, which is a maximum distance from the contour to the approximated contour.
   - Polygons with exactly four vertices are considered as potential targets (e.g., rectangles or squares).

8. **Validation of Quadrilaterals**:
   - For each approximated quadrilateral, further checks are applied:
     - **Edge Ratio Validation**: The ratios of opposite sides of the quadrilateral are calculated and compared. A significant discrepancy in these ratios may indicate that the shape is not a perfect rectangle or square, which might be critical depending on the application. I chose a range of 30 to 150 degrees as valid angles.
     - **Angle Validation**: The angles at each vertex of the quadrilateral are calculated. Valid quadrilaterals for many practical applications (like scanning documents) are expected to have angles close to 90 degrees. This step filters out skewed or irregular quadrilaterals. I chose a 1:2 ratio for edge length.

9. **Perspective Transformation**:
   - Detected quadrilaterals undergo a perspective transformation. This process involves mapping the points of the quadrilateral to a rectangle, allowing the content within the quadrilateral to be viewed head-on. This is useful for applications like document scanning where a non-frontal image of a document needs to be transformed to look like it was captured directly from the front.

