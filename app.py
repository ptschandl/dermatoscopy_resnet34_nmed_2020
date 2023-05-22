#!/usr/bin/env python3
import gradio as gr
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import itertools
import numpy as np
import os
import logging


args = {
    "model_path": "model_last_epoch_34_torchvision0_3_state.ptw",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dxlabels": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
}
logging.warning(args)


# No specific normalization was performed during training
def normtransform(x):
    return x


# Load model
logging.warning("Loading model...")
model = torchvision.models.resnet34()
model.fc = torch.nn.Linear(model.fc.in_features, len(args["dxlabels"]))
model.load_state_dict(torch.load(args["model_path"]))
model.eval()
model.to(device=args["device"])
torch.set_grad_enabled(False)
logging.warning("Model loaded.")


def predict(image: str) -> dict:
    global model, args

    logging.warning(f"Starting predict('{image}') ...")
    prediction_tensor = torch.zeros([1, len(args["dxlabels"])]).to(
        device=args["device"]
    )

    # Test-time augmentations
    available_sizes = [224]
    target_sizes, hflips, rotations, crops = available_sizes, [0, 1], [0, 90], [0.8]
    aug_combos = [x for x in itertools.product(target_sizes, hflips, rotations, crops)]

    # Load image
    img = Image.open(image)
    img = img.convert("RGB")

    # Predict with Test-time augmentation
    for target_size, hflip, rotation, crop in aug_combos:
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
        test_data = tfm(img).unsqueeze(0).to(device=args["device"])
        with torch.no_grad():
            outputs = model(test_data)
        prediction_tensor += outputs

    prediction_tensor /= len(aug_combos)
    predictions = F.softmax(prediction_tensor, dim=1)[0].detach().cpu().tolist()
    logging.warning(f"Returning {predictions=}")
    return {args["dxlabels"][enu]: p for enu, p in enumerate(predictions)}


description = """
Research image classification model for multi-class predictions of common dermatologic tumors, the model was trained on the [HAM10000 dataset](https://www.nature.com/articles/sdata2018161).

This is the model used in the publication [Tschandl P. et al. Nature Medicine 2020](https://www.nature.com/articles/s41591-020-0942-0) where human-computer interaction of such a system was analyzed.

Instructions for uploading: Ensure the image is not blurred, the lesion is centered and in focus, and no black/white vignette is in the surrounding. The image should depict the whole lesion, and not a zoomed-in part.

For education and research use only. **DO NOT use this to obtain medical advice!**
If you have a skin change in question, seek contact to a health care professional."""

logging.warning("Starting Gradio interface...")
gr.Interface(
    predict,
    inputs=gr.Image(label="Upload a dermatoscopic image", type="filepath"),
    outputs=gr.Label(num_top_classes=len(args["dxlabels"])),
    title="Dermatoscopic classification",
    description=description,
    allow_flagging="never",
    examples=[
        os.path.join(os.path.dirname(__file__), "images", x)
        for x in ["ISIC_0024306.jpg", "ISIC_0024315.jpg", "ISIC_0024318.jpg"]
    ],
).launch()
