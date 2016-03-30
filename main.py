from pyimagesearch import imutils
from skimage import exposure
import numpy as np
import cv2
from numberrec import NumberRecognizer

class ScaleReader:

    def getImageFromCamera(self, id):
        cap = cv2.VideoCapture(id)
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (800, 600))
            cv2.imshow("Camera", frame)

            self.init_image(frame)

            k = cv2.waitKey(1)

    def getImageFromVideo(self):
        cap = cv2.VideoCapture('test.avi')
        while cap.isOpened():
            ret, frame = cap.read()
            #frame = cv2.resize(frame, (720, 480))
            #cv2.imshow("Video Input", frame)

            self.init_image(frame)

            k = cv2.waitKey(self.wait)
            self.wait = 1

    def onMouseClick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = [x, y]

        if event == cv2.EVENT_LBUTTONUP:
            self.end = [x, y]
            test =self.recognize_block(self.start[0], self.end[0], self.start[1], self.end[1])
            print "[" , self.start[0] , "," , self.end[0], ",", self.start[1], ",", self.end[1] , "], "
            #print test

    def init_image(self, image):
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

        if screenCnt != None:
            cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
            cv2.imshow("Edged", image)

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


            if self.check_screen(maxWidth, maxHeight):
                print maxWidth, maxHeight

                M = cv2.getPerspectiveTransform(rect, dst)
                warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

                cv2.imshow("Warp", warp)

                gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 17, 17, 17)

                athresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                ret, thresholded = cv2.threshold(gray,60, 255, cv2.THRESH_BINARY)
                cv2.namedWindow("Grayscale")
                cv2.setMouseCallback("Grayscale", self.onMouseClick)

                self.athrimg = athresholded.copy()
                self.thrimg = thresholded.copy()
                cv2.imshow("Grayscale", self.athrimg)
                self.wait = 0
                #cv2.imshow("THRES", self.thrimg)

                # wait
                #cv2.waitKey(0)

    def recognize_block(self, args):
        x0 = args[0]
        x1 = args[1]
        y0 = args[2]
        y1 = args[3]
        cropped_img_ath = self.athrimg[y0:y1, x0:x1]
        cropped_img_th = self.thrimg[y0:y1, x0:x1]
        mean_ath, _, _, _ = cv2.mean(cropped_img_ath)
        mean_th, _, _, _ = cv2.mean(cropped_img_th)
        mean = (mean_ath + mean_th) / 2
        if mean < 200:
            return True
        else:
            return False
        #cv2.imshow("cropped", cropped_img_ath)

    def recognize_number(self, frames):
        # recognize blocks
        bool_block = []
        for index in range(len(frames)):
            bool_block.append([])
            for block_index in range(len(frames[index])):
                bool_block[index].append(self.recognize_block(frames[index][block_index]))
        # recognize numbers
        result = 0
        for index in range(len(bool_block)):
            nr =  NumberRecognizer(bool_block[index])
            recognized_number = nr.recoginze()
            if recognized_number < 0:
                print "Unkown number"
            else:
                result += recognized_number
                result *= 10
        result /= 10
        print result

    def check_screen(self, width, height):
        if width != 0 and height != 0:
            ratio = float(width) / height
            delta = ratio - self.screen_ratio
            delta_abs = abs(delta)
            print delta_abs
            if delta_abs < 0.1:
                return True
            else:
                return False
        else :
            return False

    def init_frame(self):
        self.screen_ratio = 1.32
        self.number_frame = [
            [
                [355, 414, 194, 221],
                [331, 354, 222, 296],
                [418, 438, 227, 301],
                [357, 411, 311, 329],
                [329, 349, 343, 421],
                [413, 432, 346, 418],
                [348, 412, 425, 447]
            ],[
                [488, 550, 198, 222],
                [469, 492, 227, 300],
                [553, 577, 227, 304],
                [494, 547, 315, 332],
                [467, 484, 343, 421],
                [550, 569, 348, 418],
                [548, 498, 431, 451]
            ]
        ]

    def __init__(self):
        self.wait = 1;
        image = cv2.imread("test.jpg")
        self.init_frame()
        self.getImageFromVideo()
        #self.getImageFromCamera(0)
        #self.init_image(image)
        #self.recognize_number(self.number_frame)


sr = ScaleReader()
