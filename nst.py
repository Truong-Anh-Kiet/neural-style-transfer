import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

IMG_SIZE = 512
ALPHA = 1
BETA = 10000
STEPS = 300

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STYLE_PATH = os.path.join(BASE_DIR, "style.jpg")
CONTENT_DIR = os.path.join(BASE_DIR, "contents")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

postprocess = transforms.Compose([
    transforms.Lambda(lambda t: t * torch.tensor(IMAGENET_STD, device=t.device)[:, None, None]
                      + torch.tensor(IMAGENET_MEAN, device=t.device)[:, None, None]),
    transforms.Lambda(lambda t: t.clamp(0, 1)),
    transforms.ToPILImage()
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0).to(DEVICE)

CONTENT_LAYERS = {'conv4_2': 21}
STYLE_LAYERS = {'conv1_1': 0, 'conv2_1': 5, 'conv3_1': 10, 'conv4_1': 19, 'conv5_1': 28}

vgg = models.vgg19(pretrained=True).features.to(DEVICE).eval()
for p in vgg.parameters():
    p.requires_grad_(False)

def get_features(x, model):
    feats = {}
    for name, layer in model._modules.items():
        x = layer(x)
        for lname, lidx in {**CONTENT_LAYERS, **STYLE_LAYERS}.items():
            if int(name) == lidx:
                feats[lname] = x
    return feats

def gram_matrix(f):
    n, c, h, w = f.size()
    F = f.view(c, h * w)
    G = torch.mm(F, F.t())
    return G

def compute_content_loss(target_feats, content_feats):
    return torch.mean((target_feats['conv4_2'] - content_feats['conv4_2']) ** 2)

def compute_style_loss(target_feats, style_grams):
    style_loss = 0.0
    for layer_name in style_grams.keys():
        target_feature = target_feats[layer_name]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer_name].to(target_feature.device)
        layer_loss = torch.mean((target_gram - style_gram) ** 2) / 1e6

        style_loss += layer_loss
    return style_loss

def compute_total_loss(content_loss, style_loss, alpha, beta):
    return alpha * content_loss + beta * style_loss

def run_nst(content_img, style_img, num_steps=STEPS, alpha=ALPHA, beta=BETA):
    content_feats = get_features(content_img, vgg)
    style_feats = get_features(style_img, vgg)
    style_grams = {l: gram_matrix(style_feats[l]).to(DEVICE) for l in STYLE_LAYERS}

    target = content_img.clone().requires_grad_(True).to(DEVICE)
    optimizer = optim.LBFGS([target])
    run = [0]

    print("Optimizing...")
    while run[0] <= num_steps:
        def closure():
            optimizer.zero_grad()
            target_feats = get_features(target, vgg)

            c_loss = compute_content_loss(target_feats, content_feats)
            s_loss = compute_style_loss(target_feats, style_grams)
            total_loss = compute_total_loss(c_loss, s_loss, alpha, beta)

            total_loss.backward()
            if run[0] % 100 == 0:
                print(f"Step {run[0]} | Total: {total_loss.item():.4f} | "
                      f"Content: {c_loss.item():.6f} | Style: {s_loss.item():.6f}")
            run[0] += 1
            return total_loss
        optimizer.step(closure)
    return target.detach()

if __name__ == "__main__":
    style_img = load_image(STYLE_PATH)
    content_files = sorted([f for f in os.listdir(CONTENT_DIR) if f.lower().endswith(('.jpg', '.png'))])

    results = []
    for i, fname in enumerate(content_files, 1):
        content_path = os.path.join(CONTENT_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, f"out_{i}.png")

        print(f"\nProcessing {fname}  →  {os.path.basename(output_path)}")
        content_img = load_image(content_path)
        output = run_nst(content_img, style_img, num_steps=STEPS, alpha=ALPHA, beta=BETA)
        result_img = postprocess(output.squeeze(0).cpu())
        result_img.save(output_path)
        results.append(result_img)

    plt.figure(figsize=(20, 10))
    for i, img in enumerate(results):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Generated {i + 1}")
    plt.tight_layout()
    plt.show()

    print("\nAll results saved to:", OUTPUT_DIR)