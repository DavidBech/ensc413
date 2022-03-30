import math
from tkinter import W
import cv2
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as cluster
from collections import defaultdict
from statistics import mean
import string
import random 
import glob
import os

def setupDirs(testOutPath, trainOutPath):
    # Make Target Directories
    dirNames=["BB","BK","BN","BP","BQ","BR","Empty","WB","WK","WN","WP","WQ","WR"]
    
    ClearOldData = False
    gotInput = False
    for root in [testOutPath, trainOutPath]:
        if gotInput:
            break
        for dir in dirNames:
            dir_path = "./" + root + "/" + dir
            if len(glob.glob(dir_path + "/*")) > 0:
                print("Old images exist, Clear them(Y/N)?")
                x = input()
                if x in ["Y", "y", "yes", "Yes"]:
                    ClearOldData = True
                    gotInput = True
                break
     
    # Make Directories to store cropped images
    for root in [testOutPath, trainOutPath]:
        if not os.path.isdir(root):
            print(f"{root} --  directory Not found make one here(Y/N)? {os.getcwd()}")
            x = input()
            if x in ["Y", "y", "yes", "Yes"]:
                os.mkdir("./" + root)
            else:
                exit(1)

        for dir in dirNames:
            dir_path = "./" + root + "/" + dir
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)
            elif ClearOldData:
                for file in glob.glob(dir_path + "/*"):
                    os.remove(file)

def displayImage(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)

def addLinesToImage(img, lines):
    # display lines on image
    img0 = img.copy()
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a*rho
        y0 = b*rho
        pt1  = (int(x0 + 1000*(-b)), int(y0 + 1000*a))
        pt2  = (int(x0 - 1000*(-b)), int(y0 - 1000*a))
        cv2.line(img0, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    return img0

def addPointsToImage(img, points):
    img0 = img.copy()
    for i, point in enumerate(points):
        img0 = cv2.circle(img0, (int(point[0]), int(point[1])), radius = 3, color=(0,255,0), thickness=-1)
        #cv2.putText(img0, str(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (209,80,0,255), 3)
    return img0

def find_midpoints(points):
    rows = [[]]*9
    for i in range(9):
        rows[i] = points[9*i:9*i+9]
        rows[i] = sorted(rows[i], key = lambda k: [k[0], k[1]])

    tileMidPoints = []
    for i in range(8):
        tileMidPoints.append([])
        for j in range(8):
            tileMidPoints[i].append((0.25*(rows[i][j][0] + rows[i+1][j][0] + rows[i][j+1][0] + rows[i+1][j+1][0]), 0.25*(rows[i][j][1] + rows[i+1][j][1] + rows[i][j+1][1] + rows[i+1][j+1][1])))
            
    return tileMidPoints, rows




# Read image and do lite image processing
def read_img(file):
    img = cv2.imread(str(file), 1)

    W = 1000
    height, width, depth = img.shape
    imgScale = W / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
    img = cv2.resize(img, (int(newX), int(newY)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (5, 5))
    return img, gray_blur


# Canny edge detection
def canny_edge(img, sigma=0.0):
    v = np.median(img)/3
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper)
    #cv2.imshow("edge", edges)
    #cv2.waitKey(0)
    return edges


# Hough line detection
def hough_line(edges, min_line_length=100, max_line_gap=10):
    lines1 = cv2.HoughLines(edges, 1, np.pi / 180, 125, min_line_length, max_line_gap)
    lines = np.reshape(lines1, (-1, 2))
    return lines1, lines


# Separate line into horizontal and vertical
def h_v_lines(lines):
    h_lines, v_lines = [], []
    for rho, theta in lines:
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            v_lines.append([rho, theta])
        else:
            h_lines.append([rho, theta])
    return h_lines, v_lines


# Find the intersections of the lines
def line_intersections(h_lines, v_lines):
    points = []
    for r_h, t_h in h_lines:
        for r_v, t_v in v_lines:
            a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
            b = np.array([r_h, r_v])
            inter_point = np.linalg.solve(a, b)
            points.append(inter_point)
    return np.array(points)


# Hierarchical cluster (by euclidean distance) intersection points
def cluster_points(points):
    dists = spatial.distance.pdist(points)
    single_linkage = cluster.hierarchy.single(dists)
    flat_clusters = cluster.hierarchy.fcluster(single_linkage, 15, 'distance')
    cluster_dict = defaultdict(list)
    for i in range(len(flat_clusters)):
        cluster_dict[flat_clusters[i]].append(points[i])
    cluster_values = cluster_dict.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), cluster_values)
    return sorted(list(clusters), key=lambda k: [k[1], k[0]])


# Average the y value in each row and augment original point
def augment_points(points):
    points_shape = list(np.shape(points))
    augmented_points = []
    for row in range(int(points_shape[0] / 11)):
        start = row * 11
        end = (row * 11) + 10
        rw_points = points[start:end + 1]
        rw_y = []
        rw_x = []
        for point in rw_points:
            x, y = point
            rw_y.append(y)
            rw_x.append(x)
        y_mean = mean(rw_y)
        for i in range(len(rw_x)):
            point = (rw_x[i], y_mean)
            augmented_points.append(point)
    augmented_points = sorted(augmented_points, key=lambda k: [k[1], k[0]])
    return augmented_points


# Crop board into separate images
def write_crop_images(img, points, img_count, folder_path='./raw_data/'):
    num_list = []
    shape = list(np.shape(points))
    start_point = shape[0] - 14

    if int(shape[0] / 11) >= 8:
        range_num = 8
    else:
        range_num = int((shape[0] / 11) - 2)

    for row in range(range_num):
        start = start_point - (row * 11)
        end = (start_point - 8) - (row * 11)
        num_list.append(range(start, end, -1))


    for row in num_list:
        for s in row:
            # ratio_h = 2
            # ratio_w = 1
            #base_len = math.dist(points[s], points[s + 1])
            base_len = math.sqrt(sum((px -qx)**2.0 for px, qx in zip(points[s], points[s + 1])))
            bot_left, bot_right = points[s], points[s + 1]
            start_x, start_y = int(bot_left[0]), int(bot_left[1] - (base_len * 2))
            end_x, end_y = int(bot_right[0]), int(bot_right[1])
            if start_y < 0:
                start_y = 0
            cropped = img[start_y: end_y, start_x: end_x]
            img_count += 1
            if np.shape(cropped)[0] is 0 or np.shape(cropped)[1] is 0:
                continue
            cv2.imwrite('./raw_data/alpha_data_image' + str(img_count) + '.jpeg', cropped)
            print(folder_path + 'data' + str(img_count) + '.jpeg')
    return img_count

def writeSquaresToFile(img, midPoints, cornerPoints, imgName, json, trainOutPath, testOutPath, test_percent):
    topMultiplier = 4
    otherDirMultiplier = 1.25
    pieceToDir = {
        "bishop_b": 'BB',
        "king_b": 'BK',
        "knight_b": 'BN',
        "pawn_b": 'BP',
        "queen_b": 'BQ',
        "rook_b": 'BR',
        #"": 'Empty',
        "bishop_w": 'WB',
        "king_w": 'WK',
        "knight_w": 'WN',
        "pawn_w": 'WP',
        "queen_w": 'WQ',
        "rook_w": 'WR',
    }
    for row in range(8):
        for column in range(8):
            folder = trainOutPath if random.random() > test_percent else testOutPath
            folder = "./" + folder + "/"
            locationString = string.ascii_uppercase[row] + str(column+1)
            try:
                #print(json["config"][locationString])
                classificationFolder = pieceToDir[json["config"][locationString]]
            except KeyError:
                classificationFolder = "Empty"

            midpoint = midPoints[row][column]
            topLeft = cornerPoints[row][column]
            topRight = cornerPoints[row][column+1]
            bottomLeft = cornerPoints[row+1][column]
            bottomRight = cornerPoints[row+1][column+1]
            halfWidth = max(abs(midpoint[0]-bottomRight[0]), abs(midpoint[0]-bottomLeft[0]), abs(midpoint[0]-topLeft[0]), abs(midpoint[0]-topRight[0]))
            halfHeight = max(abs(midpoint[1]-bottomRight[1]), abs(midpoint[1]-bottomLeft[1]), abs(midpoint[1]-topLeft[1]), abs(midpoint[1]-topRight[1]))
            
            startX = max(int(midpoint[0] - halfWidth*otherDirMultiplier), 0)
            endX = max(int(midpoint[0] + halfWidth*otherDirMultiplier), 0)
            startY = max(int(midpoint[1] - halfHeight*topMultiplier), 0)
            endY = max(int(midpoint[1] + halfHeight*otherDirMultiplier), 0)
            croppedImg = img [startY: endY, startX: endX]
            cv2.imwrite(folder + classificationFolder + "/" + imgName + "_" + locationString + ".jpeg", croppedImg)


if __name__ == "__main__":
    print(f"This file contains methods used in the other create_data_* files use one of those to create data")
    exit(1)
