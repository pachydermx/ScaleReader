from pyimagesearch import imutils
from skimage import exposure
import numpy as np
import argparse
import cv2

class ScaleReader:

    def onMouseClick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = [x, y]

        if event == cv2.EVENT_LBUTTONUP:
            self.end = [x, y]
            print self.start, self.end

    def __init__(self):
        image = cv2.imread("test.jpg")
        ratio = image.shape[0] / 600.0
        orig = image.copy()
        image = imutils.resize(image, height = 600)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)

        #cv2.imshow("Edged", edged)

        _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = None

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                screenCnt = approx
                break

        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
        #cv2.imshow("Gameboy screen", image)

        pts = screenCnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype = "float32")

        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        rect *= ratio

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        heightA = np.sqrt(((tr[0] - [br[0]]) ** 2) + ((tr[1] - bl[1]) ** 2))
        heightB = np.sqrt(((tl[0] - [bl[0]]) ** 2) + ((tl[1] - bl[1]) ** 2))

        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype = "float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

        #cv2.imshow("Warp", warp)

        gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 17, 17, 17)

        thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.namedWindow("Grayscale")
        cv2.setMouseCallback("Grayscale", self.onMouseClick)
        cv2.imshow("Grayscale", thresholded)

        print cv2.mean(thresholded)
        # wait
        cv2.waitKey(0)


sr = ScaleReader()
