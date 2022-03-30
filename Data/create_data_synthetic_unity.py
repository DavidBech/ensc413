import create_data_common 
import glob
import json

# The location of train cropped images
trainOutPath = "train_unity"

# The location of test cropped images
testOutPath = "test_unity"

# Expected percentage of test images
test_percent = 0.4

if __name__ == "__main__":
    create_data_common.setupDirs(testOutPath, trainOutPath)
    
    # Create a list of image file names
    img_filename_list = []
    folder_name = "./full_data/synthetic_unity/imag/*.jpg"
    successCount = 0
    failCount = 0

    # use static corner points because all unity generated images are identical other than position
    cornerPoints = (
        # Row 0
        (332, 114), #0
        (372, 114), #1
        (414, 114), #2
        (457, 114), #3 
        (499, 114), #4
        (540, 114), #5
        (584, 114), #6
        (626, 114), #7
        (668, 114), #8

        # Row 1
        (326, 147), #9
        (367, 147), #10
        (410, 147), #11
        (455, 147), #12
        (499, 147), #13
        (542, 147), #14
        (588, 147), #15
        (632, 147), #16
        (675, 147), #17

        # Row 2
        (319, 181), #18
        (362, 181), #19
        (407, 181), #20
        (454, 181), #21
        (499, 181), #22
        (544, 181), #23
        (591, 181), #24
        (637, 181), #25
        (681, 181), #26

        # Row 3
        (311, 218), #27 
        (357, 218), #28
        (404, 218), #29
        (452, 218), #30
        (499, 218), #31
        (546, 218), #32
        (594, 218), #33
        (641, 218), #34
        (689, 218), #35


        (304, 259), #36
        (351, 259), #37
        (400, 259), #38
        (450, 259), #39
        (499, 259), #40
        (548, 259), #41
        (597, 259), #42
        (646, 259), #43
        (696, 259), #44

        (294, 302), #45
        (345, 302), #46
        (397, 302), #47
        (447, 302), #48
        (499, 302), #49
        (550, 302), #50
        (601, 302), #51
        (653, 302), #52
        (704, 302), #53


        (287, 349), #54
        (338, 349), #55
        (393, 349), #56
        (445, 349), #57
        (499, 349), #58
        (553, 349), #59
        (605, 349), #60
        (662, 349), #61
        (713, 349), #62
                    
        (277, 401), #63
        (331, 401), #64
        (388, 401), #65
        (442, 401), #66
        (499, 401), #67
        (555, 401), #68
        (610, 401), #69
        (668, 401), #70
        (724, 401), #71
                    
        (267, 456), #72
        (323, 456), #73
        (383, 456), #74
        (439, 456), #75
        (499, 456), #76
        (558, 456), #77
        (615, 456), #78
        (676, 456), #79
        (735, 456), #80
    )

    # Get All Images from data directory
    for path_name in glob.glob(folder_name):
        img_filename_list.append(path_name)

    # For each file crop the image and save to category depending on json file
    for file_name in img_filename_list:
        imageName = file_name.split("\\")[-1].split(".")[0]

        # get grayscale image
        img, gray_blur = create_data_common.read_img(file_name)

        points = cornerPoints

        # sperate the points into rows and find the midpoints
        tileMidPoints, rows = create_data_common.find_midpoints(points)

        # open the json file for the image
        correspondingJson = "./full_data/synthetic_unity/json/" + imageName + ".json"
        jsonFile = open(correspondingJson, "r")
        data = json.load(jsonFile)
        jsonFile.close()

        # write the cropped image to a a file
        create_data_common.writeSquaresToFile(img, tileMidPoints, rows, imageName, data, trainOutPath, testOutPath, test_percent)

        print(file_name)
        successCount += 1

    print(f"Finished Cropped:{successCount} Images")
