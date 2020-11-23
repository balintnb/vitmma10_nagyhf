import cv2
import numpy as np

def laneDetect(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    mask_yellow = cv2.inRange(hsv, (40, 0, 150),(80, 255, 255))
    mask_white = cv2.inRange(hsv, (0, 0, 100),(255, 9, 255))
    mask = cv2.bitwise_or(mask_yellow, mask_white)
    img2 = cv2.bitwise_and(img, img, mask=mask)

    stencil = np.zeros_like(img[:,:,0])
    polygon = np.array([[0,480], [80,250], [560,250], [640,480]])
    cv2.fillConvexPoly(stencil, polygon, 1)
    img3 = cv2.bitwise_and(img2, img2, mask=stencil)
    gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img_proc = gray
    ret, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    psrc = np.float32([[290, 210], [350, 210], [140, 350], [500, 350]])
    pdst = np.float32([[200, 0], [440, 0], [190, 355], [450, 355]])
    matrix = cv2.getPerspectiveTransform(psrc, pdst)
    minv = cv2.getPerspectiveTransform(pdst, psrc)
    birdseye = cv2.warpPerspective(thresh, matrix, (w, h))

    blur = cv2.GaussianBlur(birdseye,(3, 3), 0)
    canny = cv2.Canny(blur, 40, 60)

    # Detect lines
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, 30, maxLineGap=100)

    # create a copy of the original frame
    img4 = img[:,:,:].copy()

    # draw unwrapped Hough lines
    if lines is None:
        return img4
        
    for line in lines:
        x1, y1, x2, y2 = line[0]
        src = np.zeros((1, 1, 2))
        src[:, 0] = [x1, y1]
        dst = cv2.perspectiveTransform(src, minv)
        ox1, oy1 = dst[:, 0, 0], dst[:, 0, 1]
        src[:, 0] = [x2, y2]
        dst = cv2.perspectiveTransform(src, minv)
        ox2, oy2 = dst[:, 0, 0], dst[:, 0, 1]
        cv2.line(img4, (int(ox1), int(oy1)), (int(ox2), int(oy2)), (255, 0, 0), 3)

    return img_proc, img4
