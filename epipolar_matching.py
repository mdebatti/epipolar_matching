import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

class EpipolarGeometry:
    def __init__(self, showImages):
        # Load images
        root = os.getcwd()
        imgLeftPath = os.path.join(root, 'demoImages/image_left.png')
        imgRightPath = os.path.join(root, 'demoImages/image_right.png')
        #imgLeftPath = os.path.join(root, 'demoImages/image_l.jpg')
        #imgRightPath = os.path.join(root, 'demoImages/image_r.jpg')
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

            # Use FLANN-based (Fast Library for Approximate Nearest Neighbors) matcher for SIFT
            FLANN_INDEX_KDTREE = 1  # use a  KD-Tree algorithm
            num_trees = 5           # Number of trees used in the KD-tree (more increases precision but slower)
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees = num_trees)
            search_params = dict(checks=50) # number of checks, increase for accuracy at the cost of speed
            flann = cv.FlannBasedMatcher(index_params, search_params)

            # FInding matches between descriptor (for each descriptor in desLeft, find 2 nearest matches from desRight)
            matches = flann.knnMatch(desLeft, desRight, k=2)

            # If using SIFT, we need to apply the ratio test as per Lowe's paper
            # (a good match should have a significantly closer match compared to the second-best match)
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

            # Use BFMatcher (Brute Force) with Hamming distance for ORB
            # check that if left matches right, right also matches left
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

            # list of matches, each contains (1) queryIdx (index of desLeft, i.e query)
            # (2) trainIdx (index of desRight, i.e train), (3) distance (hamming distance)
            matches = bf.match(desLeft, desRight)

            # Sort matches by distance (best matches first)
            good_matches = sorted(matches, key=lambda x: x.distance)

        else:
            raise ValueError(f"Unknown method '{method}'. Please use 'SIFT' or 'ORB'.")
        
        # Extract coordinates of keypoints based on good matches
        # kpLeft/kpRights are tuples of openCV KeyPoint objects. 
        # each keypoint object kp has an attribute pt which is a (x,y) tuple 
        # representing coordinate of that keypoint in the image
        # 2D NumPy arrays of shape (N, 2), where N is the number of good matches, 
        # and each row contains the coordinates (x, y) of a keypoint
        ptsLeft = np.float32([kpLeft[m.queryIdx].pt for m in good_matches])
        ptsRight = np.float32([kpRight[m.trainIdx].pt for m in good_matches])

        # 1. The fundamental matrix (F) encodes the relationship between corresponding points in two views (images).
        #    For points (x1, y1) in the left image and (x2, y2) in the right, the relation is: p2.T * F * p1 = 0.
        #    p1 and p2 are the homogeneous coordinates of the points in the left and right images, respectively.

        # 2. Purpose of F:
        #    - Maps points to epipolar lines between images.
        #    - Used in stereo calibration, image rectification, and 3D scene reconstruction.

        # 3. Code:
        #    F, mask = cv.findFundamentalMat(ptsLeft, ptsRight, cv.FM_LMEDS)
        #    - Inputs: 
        #      ptsLeft, ptsRight: 2D arrays with keypoint coordinates in left and right images.
        #      cv.FM_LMEDS: Least Median of Squares, robust to outliers.
        #    - Outputs: 
        #      F: 3x3 fundamental matrix encoding epipolar geometry.
        #      mask: Indicates inliers (1) and outliers (0) in the point correspondences.

        # 4. Inside cv.findFundamentalMat():
        #    - It estimates F by minimizing errors for inliers, ignoring outliers.
        #    - cv.FM_LMEDS focuses on minimizing median squared residuals.

        # 5. Usage of F:
        #    - Epipolar lines: Compute corresponding lines in the other image.
        #    - Image rectification: Align stereo images to make corresponding points lie on the same horizontal line.
        #    - Depth estimation: Reconstruct 3D structure via stereo triangulation.

        # 6. Estimation methods:
        #    - cv.FM_LMEDS: Robust to outliers using Least Median of Squares.
        #    - cv.FM_RANSAC: Randomly samples points to estimate F, also robust to outliers.
        #    - cv.FM_8POINT: Requires 8 points, works well in noise-free data but less robust to outliers.
        F, mask = cv.findFundamentalMat(ptsLeft, ptsRight, cv.FM_LMEDS)

        # Evalutate quality of Fundamental matrix
        self.check_epipolar_constraint(F, ptsLeft, ptsRight)
        self.check_inlier_ratio(mask)
        self.check_residuals(F, ptsLeft, ptsRight)

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
    
    def check_epipolar_constraint(self, F, ptsLeft, ptsRight):
        """
        The fundamental matrix (F) encodes the epipolar constraint between two images.
        The constraint is tested by checking if: p2.T * F * p1 ≈ 0, where:
            - p1 = [x1, y1, 1].T is a point in the left image (homogeneous coordinates).
            - p2 = [x2, y2, 1].T is the corresponding point in the right image (homogeneous coordinates).
            - F is the 3x3 fundamental matrix.

        The error for each point pair is calculated as the absolute value of p2.T * F * p1.
        If F is correct, the errors should be small (close to 0).

        Example code:
            - ptsLeft_h and ptsRight_h: Points converted to homogeneous coordinates.
            - For each pair, the error is computed and stored.
            - The average error gives an indication of how well the fundamental matrix satisfies the epipolar constraint.
        """
        ptsLeft_h = np.hstack((ptsLeft, np.ones((ptsLeft.shape[0], 1))))
        ptsRight_h = np.hstack((ptsRight, np.ones((ptsRight.shape[0], 1))))
        
        errors = []
        for i in range(len(ptsLeft)):
            error = np.abs(np.dot(ptsRight_h[i], np.dot(F, ptsLeft_h[i].T)))
            errors.append(error)
        
        avg_error = np.mean(errors)
        print("Average epipolar constraint error:", avg_error)
        return avg_error


    def check_inlier_ratio(self, mask):
        """
        The mask returned by cv.findFundamentalMat() indicates which point correspondences are inliers (valid points).
        - Points with a mask value of 1 are considered inliers, and 0 are outliers.

        To verify the quality of the fundamental matrix (F):
        - Check the ratio of inliers to total points.
        - A high inlier ratio (e.g., > 0.5) suggests that F is a good fit for the majority of points.

        Example code:
        - inlier_ratio: The proportion of valid points (inliers) to total points.
        - A higher inlier ratio indicates that the fundamental matrix is reliable.
        """

        inlier_ratio = np.sum(mask) / len(mask)
        print(f"Inlier ratio: {inlier_ratio:.2f}")
        return inlier_ratio

    def point_line_distance(self, pt, line):
        """Compute the distance between a point and a line."""
        a, b, c = line
        x, y = pt
        return abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

    def check_residuals(self, F, ptsLeft, ptsRight):
        """
        To quantify how well the points satisfy the epipolar constraint, compute the distance from each point 
        to its corresponding epipolar line. A small residual error (distance) suggests that the fundamental matrix (F) is accurate.

        Function:
        - point_line_distance: Computes the distance from a point to a line using the line equation ax + by + c = 0.
        
        Example usage:
        - Residuals are the distances from points to epipolar lines in both images.
        - Low average residuals indicate that the fundamental matrix fits the data well.

        Threshold:
        - There is no fixed universal threshold for residuals, but typically, values below 1 pixel suggest a good fit.
        This can vary depending on the noise level in the data and the precision of the point correspondences.
        """
        
        # Compute epipolar lines in the right image
        linesRight = cv.computeCorrespondEpilines(ptsLeft.reshape(-1, 1, 2), 1, F)
        linesRight = linesRight.reshape(-1, 3)
        
        # Compute epipolar lines in the left image
        linesLeft = cv.computeCorrespondEpilines(ptsRight.reshape(-1, 1, 2), 2, F)
        linesLeft = linesLeft.reshape(-1, 3)
        
        # Compute residuals (point-to-line distance)
        residuals = []
        for i in range(len(ptsLeft)):
            residuals.append(self.point_line_distance(ptsRight[i], linesRight[i]))
            residuals.append(self.point_line_distance(ptsLeft[i], linesLeft[i]))
        
        avg_residual = np.mean(residuals)
        print("Average point-to-epipolar-line distance:", avg_residual)
        return avg_residual


    def demoViewPics():
        # See pictures
        eg = EpipolarGeometry(showImages=True) 

def demoDrawEpilines():
    # Draw epilines
    eg = EpipolarGeometry(showImages=False)
    eg.drawStereoEpilines(method='SIFT')  # method='SIFT' (default) or method='ORB'

if __name__ == "__main__":
    #demoViewPics()
    demoDrawEpilines()
