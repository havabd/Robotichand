#------------------------------------------------------
# Prototype de prothèse de main manoeuvrable par vidéo
#------------------------------------------------------

# Importation de modules
import cv2
import numpy as np
import copy
import math
from sklearn.metrics import pairwise
import serial
import time

# from appscript import app

# auteur : Abdulla Haval & Sofidis Alexandre
# python: 3.8
# opencv: 3.3.1
# credit : Mac OS EL Capitan
# https://github.com/lzane/Fingers-Detection-using-OpenCV-and-Python?fbclid=IwAR1FzTe7_tz9B0-PGJbQMcyn9BsZdAtG-h4I7DcWSwJA45i5uuFxttl6iEg


# Mise en place de paramètres

ser=serial.Serial('COM10', 115200)
cap_region_x_begin=0.5 # Dimension du cadre (x)
cap_region_y_end=0.8 # Dimension du cadre (y)
threshold = 60  #  threshold binaire
blurValue = 41  # paramètre GaussianBlur
bgSubThreshold = 50
learningRate = 0

#  Mise en place de variables
isBgCaptured = 0   # bool
triggerSwitch = False  # activation par keyboard operations

#-------------
# Définitions
#-------------

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    # Prise de l'opérateur bit à bit (bit-wise) AND entre la main "tresholded" utilisant la région d'intérêt (ROI) circulaire en tant que masque 
    # qui donne les coupes obtenues utilisant le masque dans l'image de la main "tresholded"
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

#------------------------------------------------------------------
# Comptage du nombre de doigts dans la région segmentée de la main
#------------------------------------------------------------------

def calculateFingers(res, roi):  # -> finished bool, cnt: compte des doigts
    # Recherche de défauts de l'enveloppe convexe (convex hull) des contours de la main
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # évite le crash.   (BUG not found)

            # Initilisation du comptage des doigts
            cnt = 0
            for i in range(defects.shape[0]):  # calcul d'angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                s = (a+b+c)/2
                ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
                # Distance entre les points et l'enveloppe convexe
                d=(2*ar)/a 
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57  # theorème des cosinus
                if angle <= 120 and d> 30:
                    cnt += 1
                    cv2.circle(roi, far, 8, [211, 84, 0], -1) # Dessin des cercles la région d'intérêt
            if cnt > 0:
                return True, cnt+1
            else:
                return True, 0
    return True, 5




#---------------------
# Fonction principale
#---------------------

# Camera
# Utilisation de la webcam
camera = cv2.VideoCapture(0)
camera.set(10,400)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

# Boucle
while camera.isOpened():
    # Obtention du cadre actuel
    #time.sleep(1.5)
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # Lissage du filtre
    frame = cv2.flip(frame, 1) # Retournement du cadre de tel sorte qu'il ne soit pas une image spéculaire
    # Redimensionnement du cadre
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    # Affichage de l'image originale
    cv2.imshow('original', frame)

    #  Opération principale
    if isBgCaptured == 1:  # Cette partie ne fonctionera pas tant que le background ne sera pas capturé
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # Séquençage du roi
        cv2.imshow('mask', img)

        # Convertion de l'image (roi) en image binaire ("grayscale")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Floutage de l'image binaire
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        # Affichage de l'image floutée
        cv2.imshow('blur', blur)
        # Seuillage (Tresholding) de l'image floutée pour avoir le premier plan
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        # Affichage de l'image "tresholded"
        cv2.imshow('ori', thresh)


        # clonage du cadre
        thresh1 = copy.deepcopy(thresh)
        # Obtention des contours dans l'image "tresholded"
        _,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        # Pas de retour si aucun contour n'est détecté
        if length > 0:
            for i in range(length):  # Recherche du plus gros contour conformément à l'Aire
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            # Recherche de l'enveloppe convexe (convex hull) des contours de la main
            hull = cv2.convexHull(res)
            # Prise de la région d'intérêt (roi) circulaire contenant la paume et les doigts
            roi = np.zeros(img.shape, np.uint8)
            # Dessin des contours et de l'enveloppe convexe de la main
            cv2.drawContours(roi, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(roi, [hull], 0, (0, 0, 255), 3)

            # Comptage du nombre de doigts
            # zeroetun, count = calculatecount(res, roi)
            isFinishCal,cnt = calculateFingers(res,roi)
            if triggerSwitch is True:
                # if zeroetun is True and count <= 5:
                    #print(2)
                if isFinishCal is True and cnt <= 5:
                    print(cnt)
                    # app('System Events').keystroke(' ')  # simulate pressing blank space
                    if cnt == 0:
                        ser.write(b'0')
                        #time.sleep(0.1)
                        int(elapsed_time) = time.time()-start_time
                        print(elapsed_time)
                        if elapsed_time == 20:
                         ser.write(b'5')
                         bgModel = None
                         triggerSwitch = False
                         isBgCaptured = 0
                         print ('RESET RESET')
                    elif cnt == 1:
                        ser.write(b'1')
                        #time.sleep(0.1)
                    elif cnt == 2:
                        ser.write(b'2')
                        #time.sleep(0.1)
                    elif cnt == 3:
                        ser.write(b'3')
                        #time.sleep(0.1)
                    elif cnt == 4:
                        ser.write(b'4')
                        #time.sleep(0.1)
                    elif cnt == 5:
                        ser.write(b'5')
                        #time.sleep(0.1)


        cv2.imshow('output', roi)

    # Keyboard operations
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'):  # press 'b' pour capturer le background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
        start_time = time.time()                  ## ça fait 0 il faut un start_time ici et un ailleur start_time2 et faire une soustraction
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print ('!!!Reset BackGround!!!')
    elif k == ord('n'):
        triggerSwitch = True
        print ('!!!Trigger On!!!')
        
    elif k == ord('o'):
        print(elapsed_time)