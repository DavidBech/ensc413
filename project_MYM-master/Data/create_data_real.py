import create_data_common 
import glob
import json

# The location of train cropped images
trainOutPath = "train_real"

# The location of test cropped images
testOutPath = "test_real"

# Expected percentage of test images
test_percent = 0.6

if __name__ == "__main__":
    create_data_common.setupDirs(testOutPath, trainOutPath)

    # Create a list of image file names
    img_filename_list = []
    folder_name = "./full_data/real/imag/*"
    successCount = 0
    failCount = 0

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

        # separate horizontal and vertical lines from hough lines
        h_lines, v_lines = create_data_common.h_v_lines(lines)

        # get intersection points of horizontal and vertical lines
        intersection_points = create_data_common.line_intersections(h_lines, v_lines)

        # make points close together into one point
        points = create_data_common.cluster_points(intersection_points)

        # make sure there are the correct quantity of points
        if len(points) > 81:
            points = points[:len(points)-9]
        if len(points) != 81:
            print(f"Skipping -- {imageName}")
            failCount +=1 
            continue

        # sperate the points into rows and find the midpoints
        tileMidPoints, rows = create_data_common.find_midpoints(points)

        # open the json file for the image
        correspondingJson = "./full_data/real/json/" + imageName + ".json"
        jsonFile = open(correspondingJson, "r")
        data = json.load(jsonFile)
        jsonFile.close()

        # write the cropped image to a a file
        create_data_common.writeSquaresToFile(img, tileMidPoints, rows, imageName, data, trainOutPath, testOutPath, test_percent)

        print(file_name)
        successCount += 1

    print(f"Finished\n\tsuccess:{successCount}\n\tfail:{failCount}")
