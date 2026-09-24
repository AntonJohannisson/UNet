"""
Pixel-level prediction / semantisk segmentering med U-Net
Oxford-IIIT Pet dataset (katt/hund vs bakgrund vs kontur)
"""

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2 as T

# ---------- INSTÄLLNINGAR ----------
IMG_SIZE = 128  # bilder/masker skalas ner till 128x128 för snabb träning
N_TRAIN_SAMPLES = 200  # delmängd av träningsdata, räcker för en demo
N_VAL_SAMPLES = 40
EPOCHS = 8
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- TRANSFORMER ----------
# Bild: skalas ner och görs om till en tensor med värden 0-1
image_transform = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]
)


# Mask: samma storlek, men vi vill ha heltalsklasser (0,1,2), inte 0-1-flyttal
def mask_transform(mask):
    mask = mask.resize((IMG_SIZE, IMG_SIZE))
    mask = torch.as_tensor(list(mask.getdata()), dtype=torch.long)
    mask = mask.reshape(IMG_SIZE, IMG_SIZE)
    # Originalmaskens klasser är 1=pet, 2=background, 3=border -> gör om till 0,1,2
    return mask - 1


# ---------- LADDA DATA ----------
print("Laddar dataset (laddas ner första gången, kan ta en stund)...")
full_dataset = OxfordIIITPet(
    root="data",
    split="trainval",
    target_types="segmentation",
    download=True,
    transform=image_transform,
    target_transform=mask_transform,
)

train_dataset = Subset(full_dataset, range(N_TRAIN_SAMPLES))
val_dataset = Subset(
    full_dataset, range(N_TRAIN_SAMPLES, N_TRAIN_SAMPLES + N_VAL_SAMPLES)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------- BYGG MODELLEN ----------
# Encoder: ResNet18, förtränad på ImageNet (krymper bilden, känner igen kantar/former/texturer)
# Decoder: bygger upp en pixel-karta igen, skip connections kopplar encoder->decoder direkt
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3,  # pet / background / border
).to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- TRÄNA ----------
print("Startar träning...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(images)  # (batch, 3, H, W), en poäng per klass och pixel
        loss = loss_fn(predictions, masks)  # jämför mot facit-masken pixel för pixel
        loss.backward()  # backpropagation
        optimizer.step()  # uppdatera vikterna

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, loss: {avg_loss:.4f}")

# ---------- SPARA MODELLEN ----------
torch.save(model.state_dict(), "unet_pet_segmentation.pth")
print("Modell sparad som unet_pet_segmentation.pth")

# ---------- VISUALISERA PÅ OSEDDA BILDER ----------
model.eval()
images, masks = next(iter(val_loader))
images, masks = images.to(DEVICE), masks.to(DEVICE)

with torch.no_grad():
    predictions = model(images)
    predicted_classes = predictions.argmax(
        dim=1
    )  # välj klassen med högst poäng per pixel

n_show = 3
fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
for i in range(n_show):
    axes[i, 0].imshow(images[i].cpu().permute(1, 2, 0))
    axes[i, 0].set_title("Original")
    axes[i, 1].imshow(masks[i].cpu(), cmap="viridis", vmin=0, vmax=2)
    axes[i, 1].set_title("Facit-mask")
    axes[i, 2].imshow(predicted_classes[i].cpu(), cmap="viridis", vmin=0, vmax=2)
    axes[i, 2].set_title("Modellens gissning")
    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()
plt.savefig("segmentation_results.png")
plt.show()
print("Resultat sparat som segmentation_results.png")

