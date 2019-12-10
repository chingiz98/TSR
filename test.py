import cv2
import numpy as np
from math import sqrt
import imutils


def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized


def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3, 3), 0)  # paramter
    gray = cv2.cvtColor(LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian(gray, cv2.CV_8U, 3, 3, 2)  # parameter
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image


def binarization(image):



    thresh = cv2.threshold(image, 32, 255, cv2.THRESH_BINARY)[1]
    # thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return thresh


def preprocess_image(image):
    image = constrastLimit(image)
    cv2.imshow("CONTRAST", image)
    image = LaplacianOfGaussian(image)
    cv2.imshow("GAUSSIAN", image)
    image = binarization(image)
    cv2.imshow("BINARY", image)
    return image


# Find Signs
def removeSmallComponents(image, threshold):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    img2 = np.zeros((output.shape), dtype=np.uint8)
    # for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2


def findContour(image):
    # find contours in the thresholded image
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    return cnts


def contourIsSign(perimeter, centroid, threshold):
    #  perimeter, centroid, threshold
    # # Compute signature of contour
    result = []
    for p in perimeter:
        p = p[0]
        distance = sqrt((p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2)
        result.append(distance)
    max_value = max(result)

    signature = [float(dist) / max_value for dist in result]
    # Check signature of contour.
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold:  # is  the sign
        return True, max_value + 2
    else:  # is not the sign
        return False, max_value + 2

def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3,3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100,128,0])
    upper_blue = np.array([215,255,255])
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_white = np.array([0,0,128], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)
    # Threshold the HSV image to get only blue colors
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_black = np.array([0,0,0], dtype=np.uint8)
    upper_black = np.array([170,150,50], dtype=np.uint8)

    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask_1 = cv2.bitwise_or(mask_blue, mask_white)
    mask = cv2.bitwise_or(mask_1, mask_black)
    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame,frame, mask= mask)
    return mask

def cropSign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    #print(top,left,bottom,right)
    return image[top:bottom,left:right]

def findLargestSign(image, contours, threshold, distance_theshold):
    max_distance = 0
    coordinate = None
    sign = None
    i = 0
    signs = []
    coordinates = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = contourIsSign(c, [cX, cY], 1-threshold)
        if is_sign:
            print("SIGN_FOUND")


        if is_sign and distance > distance_theshold:
            coordinate = np.reshape(c, [-1, 2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis=0)
            coordinate = [(left - 2, top - 2), (right + 3, bottom + 1)]
            sign = cropSign(image, coordinate)
            #cv2.imshow("KEKPEP" + str(i), sign)
            signs.append(sign)
            coordinates.append(coordinate)
            #i += 1

        if is_sign and distance > max_distance and distance > distance_theshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1,2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinate = [(left-2,top-2),(right+3,bottom+1)]
            sign = cropSign(image,coordinate)
    return sign, coordinate, signs, coordinates

def localization(image, min_size_components, similitary_contour_with_circle, model, count, current_sign_type):
    original_image = image.copy()
    binary_image = preprocess_image(image)

    binary_image = removeSmallComponents(binary_image, min_size_components)

    binary_image = cv2.bitwise_and(binary_image,binary_image, mask=remove_other_color(image))

    #binary_image = remove_line(binary_image)

    cv2.imshow('BINARY IMAGE', binary_image)
    contours = findContour(binary_image)
    #signs, coordinates = findSigns(image, contours, similitary_contour_with_circle, 15)
    sign, coordinate = findLargestSign(original_image, contours, similitary_contour_with_circle, 15)
    return sign


vidcap = cv2.VideoCapture("test_video2.mp4")

status, frame = vidcap.read()

while True:
        print("NEXT FRAME")
        success,frame = vidcap.read()
        if not success:
            print("FINISHED")
            break
        frame = cv2.resize(frame, (640, 480))
        #planets = cv2.imread('123.jpg')
        planets = frame
        binary_image = preprocess_image(planets)
        binary_image = removeSmallComponents(binary_image, 200)
        binary_image = cv2.bitwise_and(binary_image, binary_image, mask=remove_other_color(planets))
        gray = cv2.cvtColor(planets, cv2.COLOR_BGR2GRAY)
        contours = findContour(binary_image)
        cv2.imshow("BINARY2", binary_image)
        sign, coordinate, signs, coordinates = findLargestSign(planets, contours, 0.8, 10)
        cv2.imshow("HoughCirlces1", frame)

        try:
            cv2.rectangle(planets, coordinate[0], coordinate[1], (255, 255, 255), 1)
        except Exception:
            5 + 5

        for c in coordinates:
            cv2.rectangle(planets, c[0], c[1], (255, 0, 0), 3)

        cv2.imshow("HoughCirlces", planets)
        cv2.waitKey(1)


        #edged = cv2.Canny(gray, 30, 200)



        #_, contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# cv2.imshow('Canny Edges After Contouring', binary_image)

# print("Number of Contours found = " + str(len(contours)))



# for i in range(0, 349):
#     cv2.drawContours(planets, contours[i], -1, (0, 255, 0), 3)
#     cv2.imshow('RESULT', planets)
#     cv2.waitKey(1)




cv2.waitKey()
cv2.destroyAllWindows()

"""
circles = cv2.HoughCircles(binary_image, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=0, maxRadius=0)
circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    #	draw	the	outer	circle
    cv2.circle(planets, (i[0], i[1]), i[2], (0, 255, 0), 6)
    #	draw	the	center	of	the	circle
    cv2.circle(planets, (i[0], i[1]), 2, (0, 0, 255), 3)

"""






