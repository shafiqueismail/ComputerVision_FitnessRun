import os                   # navigating the folders on my computer 
import json                 # reading the json files that were made using labelme
import numpy as np          # storing the final data set


LABELME_DIR = './CustomModels/landMark_DetectionModel/imagePoses_JSONForm/'     # path to labeled folder

# landmark order inside each polygon (hip > knee > ankle)
KEYPOINT_NAMES = ["hip", "knee", "ankle"]
SIDES = ["right", "left"]

data = []               # store the points (x, y) and u have 6 points so 12 in total data numbers for one image
filenames = []          # store the filenames of the 200 json files

# loops through each JSON file
for file in os.listdir(LABELME_DIR): 
    if file.endswith(".json"):              # will only look at the files that are .json in the folder since there is png files in there too       
        with open(os.path.join(LABELME_DIR, file), 'r') as f:
            label_data = json.load(f)       # loads the content of the json file into a python dictionary
        
        keypoints = {}
        for shape in label_data["shapes"]:
            label = shape["label"]          # either right or left leg
            points = shape["points"]        # 0 for hip, 1 for knee, 2 for ankle
            
            if len(points) != 3:
                continue  # skip malformed ones, meaning it needs to see 3 points, and if it only detects 2 or one it wont work
            
            for i, (x, y) in enumerate(points):
                part_name = f"{label.split('_')[0]}_{KEYPOINT_NAMES[i]}"    # label.split (either right or left) and keypoint_name is the hip, knee, or ankle 
                keypoints[part_name] = [x, y]           # example like this: keypoints["right_hip"] = [203, 106]
        
        # sort the keypoints in a fixed order
        row = []
        for side in SIDES:
            for joint in KEYPOINT_NAMES:
                coord = keypoints.get(f"{side}_{joint}", [np.nan, np.nan])
                row.extend(coord)
        
        data.append(row)
        filenames.append(label_data["imagePath"])

# convert to NumPy array
data_np = np.array(data)
print("Shape of dataset:", data_np.shape)

# save it
np.save("pose_keypoints.npy", data_np)
np.save("pose_filenames.npy", np.array(filenames))






# Bottom part is only for visualizing the data that we just extracted form the json files, in to the numpy arrays (so that we can use it later for CNN)


# Load your keypoints and filenames
keypoints = np.load("pose_keypoints.npy")
filenames = np.load("pose_filenames.npy")

# Print as a nice headered table
KEYPOINTS_ORDER = [
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle"
]

print("Filename,", ", ".join([kp + "_x, " + kp + "_y" for kp in KEYPOINTS_ORDER]))

for name, row in zip(filenames, keypoints):
    flat_coords = ", ".join([f"{x:.1f}" for x in row])
    print(f"{name}, {flat_coords}")
