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

            if self.init_image(frame):
                self.recognize_number(self.number_frame)

            k = cv2.waitKey(self.wait)
            self.wait = 1

    def onMouseClick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = [x, y]

        if event == cv2.EVENT_LBUTTONUP:
            self.end = [x, y]
            #test =self.recognize_block(self.start[0], self.end[0], self.start[1], self.end[1])

            print "[" , float(self.start[0])/self.screen_width , "," , float(self.end[0])/self.screen_width, ",", float(self.start[1])/self.screen_height, ",", float(self.end[1])/self.screen_height , "], "
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
                #print maxWidth, maxHeight

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
                self.display = athresholded.copy()
                self.display = cv2.cvtColor(self.display, cv2.COLOR_GRAY2BGR)
                cv2.imshow("Grayscale", self.athrimg)
                self.wait = 0
                #cv2.imshow("THRES", self.thrimg)

                # wait
                #cv2.waitKey(0)
                return True
        return False

    def recognize_block(self, args, mark):
        print args
        x0 = int(args[0] * self.screen_width)
        x1 = int(args[1] * self.screen_width)
        y0 = int(args[2] * self.screen_height)
        y1 = int(args[3] * self.screen_height)


        cropped_img_ath = self.athrimg[y0:y1, x0:x1]
        cropped_img_th = self.thrimg[y0:y1, x0:x1]
        mean_ath, _, _, _ = cv2.mean(cropped_img_ath)
        mean_th, _, _, _ = cv2.mean(cropped_img_th)
        #mean = (mean_ath + mean_th) / 2
        mean = mean_ath
        if mean < 230:
            self.display = cv2.rectangle(self.display, (x0, y0), (x1, y1), (255, 0, 0), thickness=5)
            return True
        else:
            self.display = cv2.rectangle(self.display, (x0, y0), (x1, y1), (255, 0, 0))
            return False
        #cv2.imshow("cropped", cropped_img_ath)

    def recognize_number(self, frames):
        # recognize blocks
        bool_block = []
        for index in range(len(frames)):
            bool_block.append([])
            for block_index in range(len(frames[index])):
                bool_block[index].append(self.recognize_block(frames[index][block_index], True))
        cv2.imshow("display", self.display)
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
            #print delta_abs
            if delta_abs < 0.1:
                self.screen_width = width
                self.screen_height = height
                return True
            else:
                return False
        else :
            return False

    def init_frame(self):
        self.screen_ratio = 1.32
        self.number_frame = [
            [
                [0.296610169492, 0.353107344633, 0.357976653696, 0.4046692607],
                [0.251412429379, 0.282485875706, 0.451361867704, 0.529182879377],
                [0.378531073446, 0.409604519774, 0.447470817121, 0.529182879377],
                [0.302259887006, 0.358757062147, 0.599221789883, 0.63813229572],
                [0.262711864407, 0.282485875706, 0.719844357977, 0.79766536965],
                [0.375706214689, 0.406779661017, 0.688715953307, 0.801556420233],
                [0.293785310734, 0.361581920904, 0.828793774319, 0.859922178988],
            ],[
                [0.502824858757, 0.564971751412, 0.350194552529, 0.396887159533],
                [0.466101694915, 0.494350282486, 0.431906614786, 0.536964980545],
                [0.576271186441, 0.607344632768, 0.443579766537, 0.536964980545],
                [0.505649717514, 0.556497175141, 0.5953307393, 0.634241245136],
                [0.468926553672, 0.497175141243, 0.688715953307, 0.805447470817],
                [0.581920903955, 0.610169491525, 0.692607003891, 0.801556420233],
                [0.525423728814, 0.567796610169, 0.84046692607, 0.875486381323],
            ],[
                [0.717514124294, 0.765536723164, 0.350194552529, 0.389105058366],
                [0.655367231638, 0.69209039548, 0.428015564202, 0.536964980545],
                [0.779661016949, 0.816384180791, 0.431906614786, 0.533073929961],
                [0.723163841808, 0.768361581921, 0.5953307393, 0.630350194553],
                [0.666666666667, 0.697740112994, 0.708171206226, 0.79766536965],
                [0.785310734463, 0.816384180791, 0.669260700389, 0.805447470817],
                [0.703389830508, 0.768361581921, 0.828793774319, 0.871595330739],
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
