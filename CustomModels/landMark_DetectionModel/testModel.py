

'''
Steps to complete the code for landmark detcetion, training a model ourselves

1. Input Images & Keypoints
You load image paths and 12 keypoints per image (6 points → x,y for: right_hip, knee, ankle + left_hip, knee, ankle).

Missing keypoints are marked as NaN.

2. Preprocessing
Images are resized to 128x128.

Keypoints are normalized (x divided by width, y by height → range [0, 1]).

3. Model
You use a Simple Convolutional Neural Network (CNN).

It ends with a fully connected layer that predicts 12 values = 6 keypoints (x, y).

This is a regression model (not classification).

4. Loss Function
You use Mean Squared Error (MSE).

You handle NaNs in keypoints by masking them out, so they don't affect training.

5. Training
You train the model for 20 epochs, printing the average loss per epoch.

6. Evaluation
You test on 5 + 5 custom test images.

For each estimated vs ground-truth keypoint, you:

Compute distance in pixels.

Count how many are within 25 pixels (your accuracy metric).

Display the image with ground truth (green) and prediction (red Xs).

'''

import os
import numpy as np
import json
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import math

# === DATASET ===
class PoseDataset(Dataset):
    def __init__(self, image_dir, image_paths, keypoints, transform=None):
        self.image_dir = image_dir
        self.image_paths = image_paths
        self.keypoints = keypoints
        self.transform = transform
        self.valid_lengths = [self._count_valid_points(kp) for kp in keypoints]

    def _count_valid_points(self, kp):
        return np.count_nonzero(~np.isnan(kp))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        kp = self.keypoints[idx].copy()
        kp[::2] = kp[::2] / width
        kp[1::2] = kp[1::2] / height
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(kp, dtype=torch.float32), self.valid_lengths[idx]

# === LOADING ===
IMG_SIZE = 128
image_dir = "./CustomModels/landMark_DetectionModel"
image_dir_jpeg = "./CustomModels/landMark_DetectionModel/imagePoses_jpegForm"
image_dir_json = "./CustomModels/landMark_DetectionModel/imagePoses_JSONForm"
image_dir_test = "./CustomModels/landMark_DetectionModel/TestDiversePoseImages"
image_paths = np.load(os.path.join(image_dir, 'NumPy_DataArraySet', "pose_filenames_official.npy"), allow_pickle=True)
keypoints = np.load(os.path.join(image_dir, 'NumPy_DataArraySet', "pose_keypoints_official.npy"))

train_paths, test_paths, train_kps, test_kps = train_test_split(
    image_paths, keypoints, test_size=0.2, random_state=42)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_dataset = PoseDataset(image_dir_jpeg, train_paths, train_kps, transform)
# test_dataset = PoseDataset(image_dir_test, test_paths, test_kps, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16)

# === MODEL ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 128),
            nn.ReLU(),
            nn.Linear(128, 12)
        )

    def forward(self, x):
        return self.net(x)

# === TRAINING ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss(reduction='none')

EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, targets, valid_len in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        masks = ~torch.isnan(targets)
        safe_targets = torch.nan_to_num(targets, nan=0.0)
        loss = criterion(outputs, safe_targets)
        masked_loss = loss * masks
        loss_per_sample = masked_loss.sum(dim=1) / masks.sum(dim=1).clamp(min=1)
        loss = loss_per_sample.mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss / len(train_loader):.4f}")

# === JSON LOADER ===
def load_labelme_keypoints(json_path, dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    width, height = Image.open(os.path.join(dir, data["imagePath"])).size
    kp_dict = {}
    for shape in data["shapes"]:
        label = shape["label"]
        points = shape["points"]
        for i, (x, y) in enumerate(points):
            part = f"{label.split('_')[0]}_{['hip', 'knee', 'ankle'][i]}"
            kp_dict[part] = [x, y]
    ordered = []
    for side in ['right', 'left']:
        for joint in ['hip', 'knee', 'ankle']:
            key = f"{side}_{joint}"
            ordered.extend(kp_dict.get(key, [np.nan, np.nan]))
    return np.array(ordered), (width, height)

# === UPDATED ACCURACY TESTER ===
def test_custom_images(model, test_items, threshold=25, is_test_diverse=True):
    model.eval()
    total_distance = 0
    total_valid = 0
    correct_within_thresh = 0

    print(f"\n{'Image':<20} | {'Accuracy (%)':<15} | {'Avg Distance (px)':<20}")
    print("-" * 60)

    image_dir_jpeg_test = "./CustomModels/landMark_DetectionModel/TestNormalPoseImages"
    image_dir_json_test = "./CustomModels/landMark_DetectionModel/TestNormalPoseJSON"

    if is_test_diverse:
        image_dir_jpeg_test = "./CustomModels/landMark_DetectionModel/TestDiversePoseImages"
        image_dir_json_test = "./CustomModels/landMark_DetectionModel/TestDiversePoseJSON"

    for img_name, json_name in test_items:
        img_path = os.path.join(image_dir_jpeg_test, img_name)
        json_path = os.path.join(image_dir_json_test, json_name)
        image = Image.open(img_path).convert('RGB')
        orig_size = image.size
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(input_tensor)[0].cpu().numpy()

        pred_x = pred[::2] * orig_size[0]
        pred_y = pred[1::2] * orig_size[1]

        gt_kp, _ = load_labelme_keypoints(json_path, image_dir_jpeg_test)
        gt_x = gt_kp[::2]
        gt_y = gt_kp[1::2]

        img_total_dist = 0
        img_total_valid = 0
        img_correct = 0

        for px, py, gx, gy in zip(pred_x, pred_y, gt_x, gt_y):
            if np.isnan(gx) or np.isnan(gy):
                continue
            dist = np.sqrt((px - gx)**2 + (py - gy)**2)
            img_total_dist += dist
            total_distance += dist
            img_total_valid += 1
            total_valid += 1
            if dist <= threshold:
                img_correct += 1
                correct_within_thresh += 1

        img_accuracy = 100 * img_correct / img_total_valid if img_total_valid else 0
        img_avg_dist = img_total_dist / img_total_valid if img_total_valid else float('inf')

        print(f"{img_name:<20} | {img_accuracy:<15.2f} | {img_avg_dist:<20.2f}")

        # Visualization
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.scatter(gt_x, gt_y, c='g', label='Ground Truth', s=40)
        plt.scatter(pred_x, pred_y, c='r', label='Estimation', s=40, marker='x')
        plt.legend()
        plt.title(f"Comparison on {img_name}")
        plt.axis('off')
        plt.show()

    # Overall stats
    avg_dist = total_distance / total_valid if total_valid else float('inf')
    accuracy = 100 * correct_within_thresh / total_valid if total_valid else 0

    print("-" * 60)
    print(f"{'OVERALL':<20} | {accuracy:<15.2f} | {avg_dist:<20.2f}")

# === FIRST SET ===
original_test_items = [
    ("test1.jpg", "test1.json"),
    ("test2.jpg", "test2.json"),
    ("test3.jpg", "test3.json"),
    ("test4.jpg", "test4.json"),
    ("test5.jpg", "test5.json"),
]
print("\n--- Running Accuracy Test on New 5 Test Images ---")
test_custom_images(model, original_test_items, threshold=25, is_test_diverse=True)

# === SECOND SET ===
extra_test_items = [
    ("test1.1.jpg", "test1.1.json"),
    ("test1.2.jpg", "test1.2.json"),
    ("test1.3.jpg", "test1.3.json"),
    ("test1.4.jpg", "test1.4.json"),
    ("test1.5.jpg", "test1.5.json"),
]
print("\n--- Running Accuracy Test on Extra 5 Test Images ---")
test_custom_images(model, extra_test_items, threshold=25, is_test_diverse=False)


'''
MORE DETAILS ON ARCHECTURE, this is how it works in simple words

3 Convolutional Layers:

Conv2d with increasing channels: 32 -> 64 -> 128: spatial features

ReLU activations: Adds non-linearity, allowing your model to learn more complex patterns

MaxPooling after each convolution (factor of 2), max pooling keeps the strongest activation in each area.

Flatten the feature map

Fully Connected Layers:

Dense layer -> 128 units -> ReLU

Output layer -> 12 units (for 6 keypoints x 2 coordinates)

Optimizer Adam - balances speed and adaptiveness
'''