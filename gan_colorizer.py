import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
import clip

# ------------- Setup --------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-B/32", device=device)

# ------------- Helpers -------------
def get_text_embedding(text: str):
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        return clip_model.encode_text(tokens).float()  # [1, 512]

def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((128, 128))
    gray = img.convert("L")
    color = T.ToTensor()(img)      # [3,128,128]
    gray  = T.ToTensor()(gray)     # [1,128,128]
    return gray.unsqueeze(0).to(device), color.unsqueeze(0).to(device)

# ------------- Generator -------------
class Generator(nn.Module):
    def __init__(self, text_dim=512):
        super().__init__()
        # project CLIP embedding → 128×8×8
        self.fc = nn.Linear(text_dim, 128 * 8 * 8)
        # a simple conv stack that keeps 128×128 resolution
        self.conv = nn.Sequential(
            nn.Conv2d(2,  64, 3, padding=1), nn.ReLU(),  # (gray + 1 text) → 64 ch
            nn.Conv2d(64,128, 3, padding=1), nn.ReLU(),  # 128×128 → 128×128
            nn.Conv2d(128,64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32,  3, 3, padding=1), nn.Tanh()   # → 3 ch color
        )

    def forward(self, gray, text_embed):
        b, _, h, w = gray.shape
        txt_feat = self.fc(text_embed).view(b, 128, 8, 8)
        txt_feat = F.interpolate(txt_feat, size=(h, w), mode='bilinear', align_corners=False)
        # now txt_feat is (b,128,128,128)
        x = torch.cat([gray, txt_feat[:, :1]], dim=1)  # (b,2,128,128)
        return self.conv(x)                           # (b,3,128,128)

# ------------- Discriminator -------------
class Discriminator(nn.Module):
    def __init__(self, text_dim=512):
        super().__init__()
        self.fc = nn.Linear(text_dim, 128 * 8 * 8)
        self.conv = nn.Sequential(
            nn.Conv2d(5,  64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64,128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 1),
            nn.Sigmoid()
        )

    def forward(self, gray, color, text_embed):
        b, _, h, w = gray.shape
        txt_feat = self.fc(text_embed).view(b, 128, 8, 8)
        txt_feat = F.interpolate(txt_feat, size=(h, w), mode='bilinear', align_corners=False)
        # concat: 1 gray + 3 rgb + 1 text = 5 channels
        x = torch.cat([gray, color, txt_feat[:, :1]], dim=1)
        return self.conv(x)  # → scalar

# ------------- Training/Test Routine -------------
def train_test_pass(image_path, description):
    gray, real_color = preprocess_image(image_path)
    text_embed      = get_text_embedding(description)

    G = Generator().to(device)
    D = Discriminator().to(device)

    # 1) Forward through G
    fake_color = G(gray, text_embed)
    print("→ generator output shape:", fake_color.shape)  # should be (1, 3, 128, 128)

    # 2) Forward through D
    d_real = D(gray, real_color, text_embed)
    d_fake = D(gray, fake_color.detach(), text_embed)
    print("→ D(real) =", d_real.item(), " D(fake) =", d_fake.item())

    # 3) One dummy train step
    criterion = nn.BCELoss()
    g_opt = torch.optim.Adam(G.parameters(), lr=2e-4)
    d_opt = torch.optim.Adam(D.parameters(), lr=2e-4)
    real_label = torch.ones_like(d_real)
    fake_label = torch.zeros_like(d_fake)

    # D step
    d_loss = criterion(d_real, real_label) + criterion(d_fake, fake_label)
    d_opt.zero_grad(); d_loss.backward(); d_opt.step()

    # G step
    d_fake2 = D(gray, fake_color, text_embed)
    g_loss  = criterion(d_fake2, real_label) + nn.L1Loss()(fake_color, real_color)
    g_opt.zero_grad(); g_loss.backward(); g_opt.step()

    # 4) Save generated
    os.makedirs("outputs", exist_ok=True)
    save_image(fake_color, "outputs/generated_final.png")
    print("✅ Saved `outputs/generated_final.png`")

if __name__ == "__main__":
    img_path    = r"C:\b tech\semesters\6th sem\topics in deep learning\project\GAN\dataset\flower.jpg"
    description = "a pink flower with green leaves"
    train_test_pass(img_path, description)
