import cv2
import sys
import os
import numpy as np  # Add this line to import numpy
from skimage import draw

def main(video_path, start_frame=0, end_frame=None, zoom_factor=1.0, rotation_angle=0, translation_x=0, translation_y=0):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the start frame and end frame
    if start_frame < 0:
        start_frame = 0
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    # Set the current frame to start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Get the directory path of the input video file
    input_dir = os.path.dirname(video_path)

    # Construct the output file path
    output_file = os.path.join(input_dir, "output.mp4")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for MP4 format
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Loop through the frames until the end_frame is reached or the video ends
    frame_number = start_frame
        
    # Check if the VideoWriter object was initialized successfully
    if not out.isOpened():
        print("Error: Couldn't initialize the VideoWriter.")
        cap.release()
        return

    # Initialize previous frame for EMA
    prev_frame = None
    prev_frame2 = None

    alpha = 0.9
    stripe_width = 50

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            # If we reached the end of the video or end_frame, break out of the loop
            break

        # Increment the frame number
        frame_number += 1
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        #if not ret:
            # If we reached the end of the video, loop back to the start frame
            #cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            #continue
        
        # Apply zoom, rotation, and translation transformations to the frame
        # Zooming
        frame = cv2.resize(frame, None, fx=zoom_factor, fy=zoom_factor)
         # Calculate the mean value of the pixels in the frame
        blurred_frame = cv2.medianBlur(frame, 5)
        mean_value = np.mean(blurred_frame)

        # Use the mean value for further processing or operations
        # For example, you can use it as a threshold value in a thresholding operation
        _, frame = cv2.threshold(blurred_frame, mean_value, 255, cv2.THRESH_BINARY)

        # Rotation
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        frame = cv2.warpAffine(frame, rotation_matrix, (width, height))

        # Translation
        translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        frame = cv2.warpAffine(frame, translation_matrix, (width, height))

        # Apply exponential moving average (EMA) for temporal smoothing
        if prev_frame is None:
            smoothed_frame = frame
        else:
            smoothed_frame = cv2.addWeighted(frame, alpha, prev_frame, 1 - alpha, 0)

        if prev_frame2 is None:
            second_smooth = frame
        else:
            second_smooth = cv2.addWeighted(frame, .1, prev_frame2, .99, 0)


        # Store the current frame for the next iteration
        prev_frame = smoothed_frame.copy()
        prev_frame2 = second_smooth.copy()

        # Apply median filter
        blurred_frame = cv2.medianBlur(smoothed_frame, 5)
        blurred_frame2 = cv2.medianBlur(second_smooth, 5)

        # Apply mean filter along y-direction
        #kernel_size_y = (3, 3)  # Adjust the kernel size as needed
        #filtered_frame = cv2.blur(blurred_frame, kernel_size_y)

        kernelY = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.float32) / 5
        
        kernelX = np.array([[0, 1, 0],
                       [0, 1, 0],
                       [0, 1, 0]], dtype=np.float32) / 3
        
        kernel = np.ones((5,5))        

        # Apply the custom kernel using filter2D
        #filtered_frame = cv2.filter2D(blurred_frame, -1, kernelX)
        erode = cv2.erode(blurred_frame, kernel, iterations = 1)
        erode2 = cv2.erode(blurred_frame2, kernel, iterations = 1)

        dilate = cv2.dilate(erode, kernel, iterations = 1)
        dilate2 = cv2.dilate(erode2, kernel, iterations = 1)
        #avg = cv2.filter2D(filtered_frame, -1, kernel)

        

        # Create a mask for the narrow vertical stripe
        mask = np.zeros((height, width), dtype=np.uint8)
        start_x = width // 2 - stripe_width // 2
        end_x = start_x + stripe_width
        mask[:, start_x:end_x] = 255

        mask2 = np.zeros((height, width), dtype=np.uint8)
        mask2[:, start_x + stripe_width :end_x + stripe_width + 100] = 255
        mask2[:, start_x - stripe_width - 100 :end_x - stripe_width] = 255

        # Apply the mask to the frame
        masked_frame = cv2.bitwise_and(dilate, dilate, mask=mask)
        masked_frame2 = cv2.bitwise_and(dilate2, dilate2, mask=mask2)

        threshold =  (masked_frame > 240).astype(np.uint8) * 255
        threshold2 =  (masked_frame2 > 240).astype(np.uint8) * 255
        # Convert the image to grayscale
        gray = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(threshold2, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        _, thresh2 = cv2.threshold(gray2, 127, 255, 0)

        #erode = cv2.erode(thresh, kernel, iterations = 1)
        dilate = cv2.dilate(thresh, kernel, iterations = 2)
        dilate2 = cv2.dilate(thresh2, kernel, iterations = 1)
        dilate2 = cv2.dilate(dilate2, kernel, iterations = 1)
        #__, contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Perform corner detection using Shi-Tomasi corner detection
        corners = cv2.goodFeaturesToTrack(gray2, maxCorners=100, qualityLevel=0.01, minDistance=5)

        # Convert corners to integers
        corners = np.intp(corners)

       
        
        

        maxi = np.maximum(dilate, dilate2)
        miny = 10000
        maxy = 0
        
        # Draw detected corners on the frame
        for corner in corners:
            x, y = corner.ravel()
            if y > maxy:
                maxy = y
            if y < miny:
                miny = y
            cv2.circle(maxi, (x, y), 3, (0, 255, 0), -1)

        stepy = (maxy-miny)/10
        stepAdj = stepy*.98

        gray_rep = ""

        for i in range(10):
            yspot = miny + 0.5*stepy + i*stepAdj 
            roi = maxi[int(yspot):int(yspot+5), 500:505]  
            avg = cv2.mean(roi)
            if avg[0] > 100:
                gray_rep += "1"
            else:
                gray_rep += "0"
         
        x_pos = 50
        y_pos = 200
        font_scale = 1.5
        font_thickness = 2
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        line_height = 40
        
        binary_rep = ""
        gray_rev = gray_rep[::-1]
        binary_rep += gray_rev[0]

        for i in range(1,10):
            if gray_rev[i] == '0':
                binary_rep += binary_rep[i - 1]
            else:
                binary_rep += '1' if binary_rep[i - 1] == '0' else '0'


        for i, bit in enumerate(gray_rep):
            cv2.putText(maxi, bit, (x_pos, y_pos + i * line_height), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        for i, bit in enumerate(binary_rep):
            cv2.putText(maxi, bit, (x_pos + 50, y_pos + i * line_height), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)    

        number = int(binary_rep, 2)
        numString = str(number)

        cv2.putText(maxi, numString, (x_pos +100, y_pos + i * line_height), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)    


        rr, cc = draw.line(250, miny, 750, miny)
        rr2, cc2 = draw.line(250, maxy, 750, maxy)
        rr3, cc3 = draw.line(500, int(miny + 0.5*stepy), 500, int(maxy - 0.5*stepy))

        ln = np.zeros_like(maxi).astype(np.uint8)
        ln2 = np.zeros_like(maxi).astype(np.uint8)
        ln3 = np.zeros_like(maxi).astype(np.uint8)
       

        ln[cc, rr] = 255
        ln2[cc2, rr2] = 255
        ln3[cc3, rr3] = 255
        lnMax = np.maximum(ln, np.maximum(ln2, ln3))
        maxi = np.maximum(maxi, lnMax)

        out.write(maxi)  # Write the frame to the output video

        # Display the masked frame
        #cv2.imshow("Video", maxi)   
        if frame_number >= end_frame:
            break

        

        # Wait for 25 milliseconds, and check for key press 'q' to exit the loop
        #if cv2.waitKey(25) & 0xFF == ord('q'):
            #break

        # Check if we reached the end frame
        #if cap.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
            # If we reached the end frame, loop back to the start frame
            #cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Release the VideoWriter object
    out.release()
    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) != 8:
        print("Usage: python script_name.py video_file_path start_frame end_frame zoom_factor rotation_angle translation_x translation_y")
        sys.exit(1)

    # Get the video file path, start frame, end frame, zoom factor, rotation angle, translation x, and translation y from the command-line arguments
    video_path = sys.argv[1]
    start_frame = int(sys.argv[2])
    end_frame = int(sys.argv[3])
    zoom_factor = float(sys.argv[4])
    rotation_angle = float(sys.argv[5])
    translation_x = float(sys.argv[6])
    translation_y = float(sys.argv[7])

    main(video_path, start_frame, end_frame, zoom_factor, rotation_angle, translation_x, translation_y)
