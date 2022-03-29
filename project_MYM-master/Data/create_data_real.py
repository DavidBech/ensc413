import create_data_common 
import os
import glob
import json

# The location of train cropped images
trainOutPath = "train_real"

# The location of test cropped images
testOutPath = "test_real"

# Expected percentage of test images
test_percent = 0.6

if __name__ == "__main__":
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
                os.mkdir("./" + testOutPath)
            else:
                exit(1)

        for dir in dirNames:
            dir_path = "./" + root + "/" + dir
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)
            elif ClearOldData:
                for file in glob.glob(dir_path + "/*"):
                    os.remove(file)

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

        # display lines on image
        #img0 = img.copy()
        #for i in range(0, len(raw)):
        #    rho = raw[i][0][0]
        #    theta = raw[i][0][1]
        #    a = math.cos(theta)
        #    b = math.sin(theta)
        #    x0 = a*rho
        #    y0 = b*rho
        #    pt1  = (int(x0 + 1000*(-b)), int(y0 + 1000*a))
        #    pt2  = (int(x0 - 1000*(-b)), int(y0 - 1000*a))
        #    cv2.line(img0, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        #cv2.imshow("lines", img0)

        # separate horizontal and vertical lines from hough lines
        h_lines, v_lines = create_data_common.h_v_lines(lines)

        # get intersection points of horizontal and vertical lines
        intersection_points = create_data_common.line_intersections(h_lines, v_lines)

        # make points close together into one point
        points = create_data_common.cluster_points(intersection_points)

        if len(points) > 81:
            points = points[:len(points)-9]
        if len(points) != 81:
            #img2 = img.copy()
            #for i, point in enumerate(points):
            #    #img2 = cv2.circle(img2, (int(point[0]), int(point[1])), radius = 1, color=(255,0,0), thickness=-1)
            #    cv2.putText(img2, str(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (209,80,0,255), 3)
            #cv2.imshow("point", img2)
            #cv2.waitKey(0)
            print(f"Skipping -- {imageName}")
            failCount +=1 
            continue

        rows = [[]]*9
        for i in range(9):
            rows[i] = points[9*i:9*i+9]
            rows[i] = sorted(rows[i], key = lambda k: [k[0], k[1]])

        #img3 = img.copy()
        #for r, row in enumerate(rows):
        #    for i, point in enumerate(row):
        #        cv2.putText(img3, str(r) + str(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (209,80,0,255), 3)
        #cv2.imshow("row point", img3)

        tileMidPoints = []
        for i in range(8):
            tileMidPoints.append([])
            for j in range(8):
                # x coord
                tileMidPoints[i].append((0.25*(rows[i][j][0] + rows[i+1][j][0] + rows[i][j+1][0] + rows[i+1][j+1][0]), 0.25*(rows[i][j][1] + rows[i+1][j][1] + rows[i][j+1][1] + rows[i+1][j+1][1])))

        #img4 = img.copy()
        #for r, row in enumerate(tileMidPoints):
        #    for i, point in enumerate(row):
        #        img4 = cv2.circle(img4, (int(point[0]), int(point[1])), radius = 1, color=(255,0,0), thickness=-1)
        #        cv2.putText(img4, string.ascii_lowercase[r] + str(i+1), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (209,80,0,255), 3)
        #cv2.imshow("mid point", img4)
        #cv2.waitKey(0)

        correspondingJson = "./full_data/real/json/" + imageName + ".json"
        jsonFile = open(correspondingJson, "r")
        data = json.load(jsonFile)
        jsonFile.close()

        create_data_common.writeSquaresToFile(img, tileMidPoints, rows, imageName, data, trainOutPath, testOutPath, test_percent)

        print(file_name)
        successCount += 1

    print(f"Finished\n\tsuccess:{successCount}\n\tfail:{failCount}")
