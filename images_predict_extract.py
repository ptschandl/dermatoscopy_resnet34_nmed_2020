#!/usr/bin/env python3

import argparse
import json
import warnings
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms, models
from PIL import Image
import glob
import os
from tqdm import tqdm
import pandas as pd
import itertools
import numpy as np


parser = argparse.ArgumentParser()
arg = parser.add_argument
arg(
    "--model_path",
    default="./model_last_epoch_34_torchvision0_3_state.ptw",
    type=str,
)
arg("--image_folder", default="./images/", type=str)
args = parser.parse_args()

assert os.path.isfile(args.model_path), "You have to provide a valid .pth model path!"
assert os.path.isdir(args.image_folder)

# Check whether normalization has to be performed
params = json.load(
    open(os.path.join(os.path.dirname(args.model_path), "params.json"), "r")
)
dxlabels = params["CLASSES"]
if params["MEAN_NORM"]:
    raise NotImplementedError()
else:

    def normtransform(x):
        return x


def resnet_extraction_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    feature_vector = torch.flatten(x, 1)
    x = self.fc(feature_vector)
    return x, feature_vector


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torchvision.models.ResNet.forward.__code__ = resnet_extraction_forward.__code__
model = torchvision.models.resnet34()
model.fc = torch.nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load(args.model_path))
model.eval()
model.to(DEVICE)
torch.set_grad_enabled(False)


class PandasDataset(Dataset):
    def __init__(self, dataframe, x_col, y_col, transform=None):
        self.dataframe = dataframe
        self.x_col = x_col
        self.y_col = y_col
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def print_classes(self):
        print(self.dataframe[self.y_col].value_counts())

    def __getitem__(self, idx):
        sample = {}
        img_name = self.dataframe.iloc[idx][self.x_col]
        img = Image.open(img_name)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        sample['image'] = img
        sample['dx'] = self.dataframe.iloc[idx][self.y_col]

        return sample["image"], sample["dx"]

# Get images
testdf = pd.DataFrame(
    sorted(
        glob.glob(os.path.join(args.image_folder, "*.jpg"))
        + glob.glob(os.path.join(args.image_folder, "*.jpeg"))
        + glob.glob(os.path.join(args.image_folder, "**", "*.jpg"))
        + glob.glob(os.path.join(args.image_folder, "**", "*.jpeg"))
    ),
    columns=["image"],
).assign(dx=lambda x: 0)
testdf = testdf.loc[~testdf.image.str.endswith(".txt")]
testdf = testdf.loc[~testdf.image.str.endswith(".json")]

prediction_tensor = torch.zeros([len(testdf), len(dxlabels)]).to(DEVICE)
feature_tensor = torch.zeros(
    [len(testdf), torchvision.models.resnet34().fc.in_features]
).to(DEVICE)

available_sizes = [params["IMGSIZE_CROP"]]
target_sizes, hflips, rotations, crops = available_sizes, [0, 1], [0, 90], [0.8]
aug_combos = [x for x in itertools.product(target_sizes, hflips, rotations, crops)]

for target_size, hflip, rotation, crop in tqdm(aug_combos, leave=True):
    tfm = transforms.Compose(
        [
            transforms.Resize(int(target_size // crop)),
            transforms.CenterCrop(target_size),
            transforms.RandomHorizontalFlip(hflip),
            transforms.RandomRotation([rotation, rotation]),
            transforms.ToTensor(),
            normtransform,
        ]
    )
    test_data = PandasDataset(testdf, "image", "dx", tfm)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    running_preds = torch.FloatTensor().to(DEVICE)
    running_features = torch.FloatTensor().to(DEVICE)
    for data in tqdm(test_loader, leave=False):
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        outputs, features = model(inputs)
        running_preds = torch.cat([running_preds, outputs])
        running_features = torch.cat([running_features, features])

    prediction_tensor += running_preds
    feature_tensor += running_features

prediction_tensor /= len(aug_combos)
feature_tensor /= len(feature_tensor)

testdf.image = testdf.image.apply(lambda x: os.path.basename(x).replace(".jpg", ""))
testdf[dxlabels] = pd.DataFrame(
    F.softmax(prediction_tensor, dim=1).cpu().numpy(), columns=dxlabels
)
testdf = testdf.set_index("image").drop("dx", axis=1)

# Store predictions as CSV, and features as both .pth and .npy
testdf.to_csv(f"images_predictions.csv", encoding="utf-8")
torch.save(feature_tensor.cpu(), f"images_features.pth")
np.save(f"images_features.npy", feature_tensor.cpu().numpy())
