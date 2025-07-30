# Computer Vision Fitness Runner
Project Run is a real-time pose-controlled endless runner game built in Python using Pygame and OpenCV. Instead of using keyboard inputs, the game detects real squats via webcam and uses them to control the character. This gamifies fitness through computer vision and AI! 


## Experiencing the Game first-hand! --- Setup:
Follow these steps to set up and run the project locally!

### 1. Set Up a Virtual Environment:
Make sure you have **Python 3.10.x** installed.  

Install `virtualenv` if you haven’t already:
pip install virtualenv

Then create a new virtual environment:
python -m venv .venv

After that activate the environment (check the table with "Command to activate virtual environment"):
https://docs.python.org/3/library/venv.html

Then install the dependencies:
pip install -r requirements.txt 

Finally, you can than run the game on your local computer by running this file **"main.py"**

## Run the custom landmark detection model
python3 ./CustomModels/landMark_DetectionModel/testModel.py




## Custom Pose Landmark Detection (CNN Model)

This project includes a **custom-trained Convolutional Neural Network** that detects 6 keypoints on the body (right/left: hip, knee, ankle) from input images.

### Input Format

Each image is labeled with 6 keypoints (12 values: x1, y1, ..., x6, y6). Some points may be missing (`NaN`) and are automatically masked during training.

---

### Training Pipeline

1. **Input**
   - JPEG images of people in squatting poses
   - `.json` files (LabelMe) with annotated keypoints
   - 6 keypoints: right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle

2. **Preprocessing**
   - Images resized to **128x128**
   - Keypoints normalized to `[0, 1]` using image width/height
   - Missing keypoints (`NaN`) masked during loss computation

3. **Model Architecture**

| Layer Type     | Details                              |
|----------------|--------------------------------------|
| Conv2d + ReLU  | 3×3 kernel, 32 channels               |
| MaxPool2d      | Reduces spatial size by 2×           |
| Conv2d + ReLU  | 3×3 kernel, 64 channels               |
| MaxPool2d      | Reduces again                        |
| Conv2d + ReLU  | 3×3 kernel, 128 channels              |
| MaxPool2d      | Final spatial reduction              |
| Flatten        | Converts to 1D                       |
| Dense (FC)     | 128 units + ReLU                     |
| Output Layer   | 12 outputs (6 keypoints × 2)         |

- **Activation:** ReLU
- **Loss Function:** Mean Squared Error (MSE), masked to ignore missing keypoints
- **Optimizer:** Adam (learning rate = 1e-4)

4. **Training**
   - 20 epochs
   - Batches of 16
   - Prints masked average loss per epoch

5. **Evaluation**
   - Tested on 10 custom images (5 diverse, 5 normal)
   - Accuracy = % of keypoints within **25 pixels**
   - Images are displayed with:
     - Ground Truth → Green Dots
     - Estimates → Red Xs
   - Average accuracy + distance printed

# Example of game-play!
<img width="752" height="422" alt="image" src="https://github.com/user-attachments/assets/cd743cc0-b653-4d1f-ad4e-c644815d1427" />

# Research Paper:
You can look at this file in the repository for the research paper: ConferencePaperIEEE_IsmailShafique_AdamAtamnia (4).pdf

# Some Resources Used:

https://www.youtube.com/watch?v=DHgj5jhMJKg&list=PLjcN1EyupaQm20hlUE11y9y8EY2aXLpnv&index=4

https://www.youtube.com/watch?v=M6e3_8LHc7A

https://www.youtube.com/watch?v=DCqV1ARz-Yw

Chatgpt

Some of the art assets used are paid so do not distribute them.

We also generally used youtube and google searches for looking up how to do different things.

Custom model images taken from H&M, Forever21, Simons, and general google search.

