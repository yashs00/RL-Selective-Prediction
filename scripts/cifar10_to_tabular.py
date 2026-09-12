import os
import sys
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

# Ensure data_cache directory exists to trick the existing pipeline loader
CACHE_DIR = Path(__file__).resolve().parents[1] / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Download CIFAR-10 Test Set 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

# 2. Load Pre-Trained ResNet-20
print("Downloading Pre-trained ResNet-20...")
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
model = model.to(device)
model.eval()

# 3. Extract Embeddings (Penultimate Layer) and Labels
embeddings = []
labels = []

def hook(module, input, output):
    # input is a tuple. The first element is the tensor flowing into the fc layer
    embeddings.append(input[0].detach().cpu().numpy())

# Attach the hook to the final fully-connected layer
model.fc.register_forward_hook(hook)

print("Extracting ResNet embeddings for 10,000 CIFAR-10 test images...")
with torch.no_grad():
    for inputs, targets in testloader:
        inputs = inputs.to(device)
        _ = model(inputs)
        labels.append(targets.numpy())

X_emb = np.vstack(embeddings)
y = np.concatenate(labels)

# 4. Create Tabular DataFrame
print(f"Extracted feature shape: {X_emb.shape} (10000 images, {X_emb.shape[1]} features)")
df = pd.DataFrame(X_emb, columns=[f"feat_{i}" for i in range(X_emb.shape[1])])
df["target"] = y

# 5. Save as fake OpenML dataset (ID 99999) to bypass API fetching
out_path = CACHE_DIR / "openml_99999.parquet"
df.to_parquet(out_path)

print(f"\n✅ Success! Saved to {out_path}")
print("Your existing pipeline will now treat CIFAR-10 as a tabular dataset with ResNet features!")
