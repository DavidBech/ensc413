import create_data_common 
import glob
import json
import math

# The location of train cropped images
trainOutPath = "train_blender"

# The location of test cropped images
testOutPath = "test_blender"

# Expected percentage of test images
test_percent = 0.4

def filter_lines(lines, tolerance=50):
    filtered_lines = []
    for line in lines:
        rho = line[0]
        theta = line[1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a*rho
        y0 = b*rho
        pt1  = (int(x0 + 1000*(-b)), int(y0 + 1000*a))
        pt2  = (int(x0 - 1000*(-b)), int(y0 - 1000*a))
        if abs(pt1[0] - pt2[0]) > tolerance and abs(pt1[1] -pt2[1]) > tolerance:
            pass
        else:
            filtered_lines.append(line)
    return filtered_lines

if __name__ == "__main__":
    create_data_common.setupDirs(testOutPath, trainOutPath)
    
    # Create a list of image file names
    img_filename_list = []
    folder_name = "./full_data/synthetic_blender/imag/100*.jpg"
    successCount = 0
    failCount = 0

    # use static corner points because all unity generated images are identical other than position
       # Get All Images from data directory
    for path_name in glob.glob(folder_name):
        img_filename_list.append(path_name)

    # For each file crop the image and save to category depending on json file
    for file_name in img_filename_list:
        imageName = file_name.split("\\")[-1].split(".")[0]

        # get grayscale image
        img, gray_blur = create_data_common.read_img(file_name)

        # get edges in image
        edges = create_data_common.canny_edge(gray_blur, 1)

        # get hough lines from edges
        raw, lines = create_data_common.hough_line(edges)

        # remove diagonal lines
        lines = filter_lines(lines)

        # separate horizontal and vertical lines from hough lines
        h_lines, v_lines = create_data_common.h_v_lines(lines)

        # get intersection points of horizontal and vertical lines
        intersection_points = create_data_common.line_intersections(h_lines, v_lines)

        # make points close together into one point
        points = create_data_common.cluster_points(intersection_points)

        # sperate the points into rows and find the midpoints
        #tileMidPoints, rows = create_data_common.find_midpoints(points)

        # open the json file for the image
        correspondingJson = "./full_data/synthetic_blender/json/" + imageName + ".json"
        jsonFile = open(correspondingJson, "r")
        data = json.load(jsonFile)
        jsonFile.close()

        # write the cropped image to a a file
        #create_data_common.writeSquaresToFile(img, tileMidPoints, rows, imageName, data, trainOutPath, testOutPath, test_percent)

        print(file_name)
        successCount += 1

    print(f"Finished Cropped:{successCount} Images")
