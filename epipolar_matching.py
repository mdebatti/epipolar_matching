import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

class EpipolarGeometry:
    def __init__(self, showImages):
        # Load images
        root = os.getcwd()
        imgLeftPath = os.path.join(root, 'demoImages/image_l.jpg')
        imgRightPath = os.path.join(root, 'demoImages/image_r.jpg')
        #imgLeftPath = os.path.join(root, 'demoImages/image1.png')
        #imgRightPath = os.path.join(root, 'demoImages/image2.png')

        # Convert to grey-scale as most feature detection algorithms 
        # work on intensity values
        self.imgLeft = cv.imread(imgLeftPath, cv.IMREAD_GRAYSCALE)
        self.imgRight = cv.imread(imgRightPath, cv.IMREAD_GRAYSCALE)

        if showImages:
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.imgLeft, cmap='gray')
            plt.subplot(122)
            plt.imshow(self.imgRight, cmap='gray')
            plt.show()

    def drawStereoEpilines(self, method='SIFT'):
        """
        Draws the stereo epilines using the specified feature detection method.

        Args:
            method: The feature detection method to use. Options are 'SIFT' or 'ORB'.
                    Default is 'SIFT'.
        """
    
        # Feature Matching
        if method == 'SIFT':
            #SIFT (Scale-Invariant Feature Transform)
            sift = cv.SIFT_create()

            # Optional binary mask to specify the region of interest (ROI)
            # for feature detection (255: detect in this region, 0: don't)
            mask = None

            # Detect N keypoints (KpLeft/KpRight) 
            # and and computes N x 128 descriptor (desLeft/desRight) for SIFT (floats)
            kpLeft, desLeft = sift.detectAndCompute(self.imgLeft, mask)
            kpRight, desRight = sift.detectAndCompute(self.imgRight, mask)

            # Use FLANN-based matcher for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desLeft, desRight, k=2)

            # If using SIFT, we need to apply the ratio test as per Lowe's paper
            good_matches = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)

        elif method == 'ORB':
            # ORB PARAMETERS
            nfeatures =      500                    #The maximum number of features to retain. The default is 500.
            scaleFactor =    1.2                    #Pyramid decimation ratio, greater than 1. The default is 1.2.
            nlevels =        8                      #The number of pyramid levels. The default is 8.
            edgeThreshold =  31                     #This is the size of the border where features are not detected. The default is 31.
            firstLevel =     0                      #The level of pyramid to put the source image to. The default is 0.
            WTA_K =          2                      #The number of points that produce each element of the oriented BRIEF descriptor. The default is 2.
            scoreType =      cv.ORB_HARRIS_SCORE    #The type of score to use for ranking features. The default is cv.ORB_HARRIS_SCORE.
            patchSize =      31                     #The size of the patch used by the BRIEF descriptor. The default is 31.
            fastThreshold =  20                     #The FAST threshold for the keypoint detector. The default is 20.

            # Feature Matching ORB (Oriented FAST and Rotated BRIEF)
            # Binary Robust Independent Elementary Features (BRIEF)
            # Features from Accelerated Segment Test (FAST) 
            # feature matching can be done via the Hamming distance 
            # (number of different bits - very fast with bitwise operations)
            orb = cv.ORB_create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold)

            # Detect N keypoints (KpLeft/KpRight) 
            # and and computes N x 32 descriptor (desLeft/desRight) for ORB (binary)
            kpLeft, desLeft = orb.detectAndCompute(self.imgLeft, None)
            kpRight, desRight = orb.detectAndCompute(self.imgRight, None)

            # Use BFMatcher with Hamming distance for ORB
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desLeft, desRight)

            # Sort matches by distance (best matches first)
            good_matches = sorted(matches, key=lambda x: x.distance)

        else:

            raise ValueError(f"Unknown method '{method}'. Please use 'SIFT' or 'ORB'.")
        
        # Extract points based on good matches
        ptsLeft = np.float32([kpLeft[m.queryIdx].pt for m in good_matches])
        ptsRight = np.float32([kpRight[m.trainIdx].pt for m in good_matches])

        # Calculate the fundamental matrix
        F, mask = cv.findFundamentalMat(ptsLeft, ptsRight, cv.FM_LMEDS)

        # Extract inliers (good points)
        ptsLeft = ptsLeft[mask.ravel() == 1]
        ptsRight = ptsRight[mask.ravel() == 1]
        step = 10
        ptsLeft = ptsLeft[::step, :]
        ptsRight = ptsRight[::step, :]

        # Draw the epipolar lines on both images
        imgLeftLines, imgRightLines = EpipolarGeometry.drawEpilines(self.imgLeft, self.imgRight, ptsLeft, ptsRight, F)

        plt.subplot(121)
        plt.imshow(imgLeftLines)
        plt.subplot(122)
        plt.imshow(imgRightLines)
        plt.show()

    @staticmethod
    def drawEpilines(img1, img2, pts1, pts2, F):
        """ Draw the epipolar lines on both images.
        
        Args:
            img1: The first image on which epilines from the other image will be drawn.
            img2: The second image corresponding to img1.
            pts1: Points from the first image.
            pts2: Corresponding points from the second image.
            F: The fundamental matrix.
            
        Returns:
            img1_with_lines: The first image with epilines drawn.
            img2_with_lines: The second image with matching feature points.
        """
    
        # Convert images to color (if they are grayscale) to draw colored lines
        img1_color = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
        img2_color = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

        # Compute epilines corresponding to points in the second image (pts2)
        # and drawing the lines on the first image (img1)
        lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)

        # Compute epilines corresponding to points in the first image (pts1)
        # and drawing the lines on the second image (img2)
        lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)

        # Draw the lines and points on both images with matching colors
        for line1, line2, pt1, pt2 in zip(lines1, lines2, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())

            # Convert points to integers
            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2))

            # Draw the epiline on the first image
            x0, y0 = map(int, [0, -line1[2] / line1[1]])
            x1, y1 = map(int, [img1_color.shape[1], -(line1[2] + line1[0] * img1_color.shape[1]) / line1[1]])
            cv.line(img1_color, (x0, y0), (x1, y1), color, 1)
            cv.circle(img1_color, pt1, 5, color, -1)

            # Draw the epiline on the second image
            x0, y0 = map(int, [0, -line2[2] / line2[1]])
            x1, y1 = map(int, [img2_color.shape[1], -(line2[2] + line2[0] * img2_color.shape[1]) / line2[1]])
            cv.line(img2_color, (x0, y0), (x1, y1), color, 1)
            cv.circle(img2_color, pt2, 5, color, -1)

        return img1_color, img2_color

def demoViewPics():
    # See pictures
    eg = EpipolarGeometry(showImages=True)

def demoDrawEpilines():
    # Draw epilines
    eg = EpipolarGeometry(showImages=False)
    eg.drawStereoEpilines(method='ORB')  # method='SIFT' (default) or method='ORB'

if __name__ == "__main__":
    #demoViewPics()
    demoDrawEpilines()
