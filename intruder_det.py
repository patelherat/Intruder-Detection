"""
This is a background subtractor program which gives us the foreground and thus gives us any objects which is new
In this background.
@author- Tushar Kumar
@author- Herat Patel
"""

import cv2
import numpy as np
import os


"""
This function implements the foreground separation.
"""
def seg(name):
    try:
        # The path for the training folder is specified.
        path = "IMG_ACV_A3_2"
        fmask = cv2.createBackgroundSubtractorMOG2(216, 40, detectShadows=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Training the background subtractor model
        # Learning rate is kept at 1 which is full i.e. it will learn from all the previous images.
        cnt = 0
        for image_path in os.listdir(path):
            input_path = os.path.join(path, image_path)
            img_og = cv2.imread(input_path)
            img_og = cv2.resize(img_og, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
            img = img_og
            if path == "IMG_ACV_A3_2" or path == "Night":
                img = cv2.GaussianBlur(img_og, (3,3), 0)
                alpha = 1.6
                beta = 1.6
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                kernel_n = np.ones((5,5), np.uint8)
                img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel_n)
                filter = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
                img = cv2.filter2D(img, -1, filter)
            gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fmask.apply(gray1, learningRate=1)
            # cv2.imshow("1", temp)
            # cv2.waitKey(0)
            cnt = cnt + 1
            # if cnt == 50:
            #     break
        print("Training Complete!!")
        cnt = 0
        # The input image is read and the background separator model is applied on it
        # and morphological operations are done on the output of the background separator model
        # to give the final result
        frame_og = cv2.imread(name)
        frame_og = cv2.resize(frame_og, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        frame = frame_og
        if path == "IMG_ACV_A3_2" or path == "Night":
            frame = cv2.GaussianBlur(frame_og, (3, 3), 0)
            alpha = 1.6
            beta = 1.6
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            kernel_n = np.ones((5, 5), np.uint8)
            frame = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel_n)
            filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            frame = cv2.filter2D(frame, -1, filter)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if path == "Night":
            mask = fmask.apply(gray, learningRate=0.016)
        else:
            mask = fmask.apply(gray, learningRate=0.016)
        mask[mask == 127] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if path == "IMG_ACV_A3_2":
            new_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, new_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        # The original image and the output image is shown
        cv2.imshow('vid1', frame_og)
        cv2.waitKey(0)
        cv2.imshow('vid2', mask)
        cv2.waitKey(0)
        #cv2.imwrite("Deer_1178.jpg", mask)
    except Exception as err:
        print("There is an error!", err)
    except cv2.Error as err1:
        print("There is an error!", err1)


def main():
    seg("Deer_Testing\IMG_1178.jfif")


if __name__ == "__main__":
    main()
