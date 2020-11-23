import cv2
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

# Compute scaled absolute image gradients
def create_sobels(image, sobel_kernel=3):
    # Convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute x and y derivatives using sobel
    sobelx = cv2.Sobel (gray, cv2.CV_32F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, sobel_kernel)

    # Get absolute value
    sobelx_abs = np.absolute(sobelx)
    sobely_abs = np.absolute(sobely)

    # Compute scaling factor
    scale_x = 255 / np.max(sobelx_abs)
    scale_y = 255 / np.max(sobely_abs)

    # Scale gradients
    sobelx_scaled = scale_x*sobelx_abs
    sobely_scaled = scale_y*sobely_abs

    return sobelx_abs,sobely_abs,sobelx_scaled,sobely_scaled
    
# Threshold gradients based on magintude
def mag_threshold(scaled_sobelx, scaled_sobely, mag_thresh=(30, 100)):
    # Compute gradient magnitude
    mag = scaled_sobelx + scaled_sobely

    # Threshold using inRange
    mag_binary = cv2.inRange(mag, mag_thresh[0] * 2, mag_thresh[1] * 2)

    return mag_binary
    
# Threshold gradients based on direction
def dir_threshold(abs_sobelx, abs_sobely, thresh=(0, np.pi/3)):
    # Compute gradient direction
    dir = np.arctan2 (abs_sobely, abs_sobelx)

    # Threshold using inRange
    dir_binary = cv2.inRange(dir, thresh[0], thresh[1])

    return dir_binary
    
# Apply color threshold
def color_threshold(image):
    thresh=(200,255)
    # Convert to hls
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # Get saturation
    image_s = image_hls[:, :, 2]

    # Threshold using inRange
    color_binary = cv2.inRange(image_s, thresh[0], thresh[1])

    return color_binary
    
# Combine all kinds of thresholds
def apply_thresholds(image, ksize=3):
    # Get derivatives
    abs_sobelx, abs_sobely, scaled_sobelx, scaled_sobely = create_sobels(image)
    
    # Compute magnitude and direction threshold
    mag_thresh = mag_threshold(scaled_sobelx, scaled_sobely)
    dir_thresh = dir_threshold(abs_sobelx, abs_sobely)
    
    # Compute color threshold
    color_thresh = color_threshold(image)

    # Combine all thresholded images
    combined = np.zeros_like(abs_sobelx)
    combined[((mag_thresh == 255) & (dir_thresh == 255)) | (color_thresh == 255)] = 1

    return combined
    
# Read video file
clip = cv2.VideoCapture("original.mp4")

# Get ROI
#while(True):
# Read frame from video
success, img = clip.read()
#if not success:
#    break

img = img[450:710, 220:1150]
thresholded = apply_thresholds(img)
#Run lane detection
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(25,15))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(thresholded, cmap='gray')

def warp(img):
    img_size = (img.shape[1], img.shape[0])

    # Source and destination points
    src = np.float32(
        [[380, 0],
          [875, 235],
          [60, 235],
          [470, 0]])

    dst = np.float32(
        [[150, 0],
         [800, 260],
         [150, 260],
         [800, 0]])

    # Get perspective transforms
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp image
    binary_warped = cv2.warpPerspective(img, M, img_size)#, flags=cv2.INTER_LINEAR)
    
    return binary_warped, Minv
    

def get_histogram(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)

    return histogram
    

class LineDetector(object):

    def draw_lane_lines(self, original_image, warped_image, Minv, draw_info):
        leftx = draw_info['leftx']
        rightx = draw_info['rightx']
        left_fitx = draw_info['left_fitx']
        right_fitx = draw_info['right_fitx']
        ploty = draw_info['ploty']

        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
        result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

        return result

    def slide_window(self, binary_warped, histogram):

        # Get line bases
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Define sliding window
        nwindows = 9
        window_height = np.int(binary_warped.shape[0]/nwindows)

        # Get nonzero pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Initialize sliding window
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            # Compute window coordinates
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Gather points that fit into the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append to list
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If found enough pixels, then shift the line midpoint
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Reshape index array
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Gather point coordinates
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit second orde polynoms
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Get points that fit on the polynoms
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
        left_fit[1]*nonzeroy + left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
        right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Refit polynom
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate polynom points
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Return value
        ret = {}
        ret['leftx'] = leftx
        ret['rightx'] = rightx
        ret['left_fitx'] = left_fitx
        ret['right_fitx'] = right_fitx
        ret['ploty'] = ploty

        return ret

    def measure_curvature(self, lines_info):
        # Approximate meters per pixels
        ym_per_pix = 30/(260)
        xm_per_pix = 3.7/(650)

        # Get polynom points
        leftx = lines_info['left_fitx']
        rightx = lines_info['right_fitx']

        # Reverse order
        leftx = leftx[::-1]
        rightx = rightx[::-1]

        ploty = lines_info['ploty']

        # Compute corvature
        y_eval = np.max(ploty)
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        return left_curverad, right_curverad

    def sanity(self, ret, left_curverad, right_curverad):
        # Sanity check: whether the lines are roughly parallel and have similar curvature
        slope_left = ret['left_fitx'][0] - ret['left_fitx'][-1]
        slope_right = ret['right_fitx'][0] - ret['right_fitx'][-1]
        slope_diff = abs(slope_left - slope_right)
        slope_threshold = 150
        curve_diff = abs(left_curverad - right_curverad)
        curve_threshold = 10000

        return (slope_diff < slope_threshold and curve_diff < curve_threshold)


    def process_image(self, image):

        roi = image[450:710, 220:1150]

        # Thresholding
        binary = apply_thresholds(roi)

        # Transforming Perspective
        binary_warped, Minv = warp(binary)

        # Getting Histogram
        histogram = get_histogram(binary_warped)

        # Sliding Window to detect lane lines
        ret = self.slide_window(binary_warped, histogram)

        # Measuring Curvature
        left_curverad, right_curverad = self.measure_curvature(ret)

        # Sanity check
        if not self.sanity(ret,left_curverad,right_curverad):
            # Use last good images if sanity check fails
            binary_warped = self.used_warped
            ret = self.used_ret

        result = image

        # Visualizing Lane Lines Info
        result[450:710, 220:1150] = self.draw_lane_lines(roi, binary_warped, Minv, ret)

        # Compute deviation
        deviation_pixels = image.shape[1]/2 - abs(ret['right_fitx'][-1] - ret['left_fitx'][-1])
        xm_per_pix = 3.7/(650)
        deviation = deviation_pixels * xm_per_pix

        # update last good images
        self.used_warped = binary_warped
        self.used_ret = ret

        return result, deviation, left_curverad

class Estimator(object):
    def __init__(self):
        super(Estimator,self).__init__()

        # Number of measurementes to use
        self.numEstim = 25

        # Number of estimates to use for slope estimation
        self.slopeEstimNum = 5

        # Iteration counter
        self.iter = 0

        # Arrays for input variables
        self.curvatures = np.zeros(self.numEstim)
        self.deviations = np.zeros(self.numEstim)

        # Store filtered deviations for slope estimation
        self.filteredDeviations = np.zeros(self.slopeEstimNum)

        # Create coefficient matrix for LS estimate
        '''
        [ 0 1 ]
        [ 1 1 ]
        [ 2 1 ]
        [ 3 1 ]
        [ 4 1 ]
        '''
        self.X = np.transpose(np.array([np.linspace(0,self.slopeEstimNum-1, num=self.slopeEstimNum),np.ones(self.slopeEstimNum)]))

    def update(self, curv, dev):

        # Shift the arrays
        self.curvatures = np.roll(self.curvatures, 1)
        self.curvatures[0] = curv

        self.deviations = np.roll(self.deviations, 1)
        self.deviations[0] = dev

        # Write new elements to replace the last
        

        # Compute means
        curvEst = np.mean(self.curvatures)
        devEst = np.mean(self.deviations)

        # Append latest deviation estimate and remove oldest
        self.filteredDeviations = np.roll(self.filteredDeviations, 1)
        self.filteredDeviations[0] = devEst

        self.iter += 1

        self.iter += 1

        if self.iter < self.slopeEstimNum:
            slopeEst = 0
        else:
            # Perform LS estimation to fit a line on the deviations
            lineEst = np.linalg.lstsq(self.X, self.filteredDeviations, rcond=None)[0]
            slopeEst = -lineEst[0]

            # Get the slope of the line - change of the deviation
            

        return curvEst,slopeEst,devEst



def drawImage(img, dev, cur):

    # Annotating curvature
    fontType = cv2.FONT_HERSHEY_SIMPLEX
    curvature_text = 'The radius of curvature = ' + str(round(cur, 3)) + 'm'
    cv2.putText(img, curvature_text, (30, 60), fontType, 0.5, (255, 255, 255), 1)

    # Annotating deviation
    direction = "left" if dev < 0 else "right"
    deviation_text = 'Vehicle is ' + str(round(abs(dev), 3)) + 'm ' + direction + ' of center'
    cv2.putText(img, deviation_text, (30, 110), fontType, 0.5, (255, 255, 255), 1)

    return img


