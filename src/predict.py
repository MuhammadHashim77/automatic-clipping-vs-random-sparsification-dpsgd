import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from networks import get_model
from torchvision import transforms


def load_image(path, size=224):
    img = np.load(path)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)


def init_model(path, name="unet", in_channels=1, classes=9):
    model = get_model(name, in_channels=in_channels, num_classes=classes)
    state = torch.load(path, map_location="cpu")
    if list(state.keys())[0].startswith("_module"):
        state = {k.replace("_module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def infer(model, image):
    with torch.no_grad():
        return model(image)


def save_plot(image, mask, prediction, out_path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Input Image')
    axs[1].imshow(mask, cmap='plasma')
    axs[1].set_title('True Segmentation')
    axs[2].imshow(prediction, cmap='viridis')
    axs[2].set_title('Model Output')
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="code/unet.pt")
    parser.add_argument("--data_path", type=str, default="data/dicom-images-test")
    args = parser.parse_args()

    model = init_model(args.model_path).to("cpu")
    img_dir = os.path.join(args.data_path, "images")
    mask_dir = os.path.join(args.data_path, "masks")
    os.makedirs("predictions", exist_ok=True)

    for fname in os.listdir(img_dir):
        img = load_image(os.path.join(img_dir, fname)).to("cpu")
        output = infer(model, img)
        pred = torch.argmax(output, dim=1).squeeze().numpy()
        mask = np.load(os.path.join(mask_dir, fname))
        save_plot(img.squeeze(), mask, pred, f"predictions/{fname}_prediction.png")


if __name__ == "__main__":
    run()
