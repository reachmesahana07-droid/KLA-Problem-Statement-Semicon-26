import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn

# ==========================================
# 1. Model Architecture (RestorationNet)
# ==========================================
# Defined directly in the evaluation script as requested by the README
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c * 2, kernel_size=3, padding=1)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, c)

    def forward(self, x):
        return x + self.conv2(self.sg(self.conv1(self.norm(x))))

class RestorationNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=32, scale_factor=2):
        super().__init__()
        self.intro = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        # Lightweight convolutional trunk
        self.body = nn.Sequential(*[NAFBlock(num_features) for _ in range(4)])
        # 2x Spatial Upscaling
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        feat = self.intro(x)
        return self.upsampler(feat + self.body(feat))

# ==========================================
# 2. Evaluation Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="KLA Image Restoration Evaluation Script")
    parser.add_argument("--input_dir", type=str, default="./test_images", help="Path to input .npy images")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Path to save restored .npy images")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth", help="Path to model weights")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup Device (Automatic CUDA selection)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")

    # Initialize Model
    print("[*] Initializing RestorationNet...")
    model = RestorationNet(scale_factor=2).to(device)

    # Load Checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
    
    print(f"[*] Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle the requested dictionary format vs plain state dict
    output_min = 0.0
    output_max = 1.0
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        output_min = checkpoint.get("output_min", 0.0)
        output_max = checkpoint.get("output_max", 1.0)
    else:
        # Fallback if checkpoint is just the plain PyTorch state dict
        model.load_state_dict(checkpoint)
        
    model.eval()
    print(f"[*] Output clipping range set to: [{output_min}, {output_max}]")

    # Find all .npy files
    input_files = glob.glob(os.path.join(args.input_dir, "*.npy"))
    if len(input_files) == 0:
        print(f"[!] No .npy files found in {args.input_dir}")
        return

    print(f"[*] Found {len(input_files)} images. Starting inference...")

    # Run Inference
    with torch.inference_mode():
        for file_path in input_files:
            filename = os.path.basename(file_path)
            
            # 1. Load the 2D floating-point array (H, W)
            input_img = np.load(file_path).astype(np.float32)
            
            # Note: Explicitly NOT clipping the input here per README instructions
            
            # 2. Add batch and channel dimensions -> (1, 1, H, W)
            input_tensor = torch.from_numpy(input_img).unsqueeze(0).unsqueeze(0).to(device)
            
            # 3. Model forward pass
            output_tensor = model(input_tensor)
            
            # 4. Remove extra dimensions -> (H_new, W_new)
            restored_img = output_tensor.squeeze().cpu().numpy()
            
            # 5. Clip ONLY the final restored prediction to the ground-truth range
            restored_img = np.clip(restored_img, output_min, output_max)
            
            # 6. Save the output as a 2D float32 .npy array
            save_path = os.path.join(args.output_dir, filename)
            np.save(save_path, restored_img.astype(np.float32))
            
            print(f"    -> Processed {filename} | Input: {input_img.shape} -> Output: {restored_img.shape}")

    print("[*] Evaluation complete!")

if __name__ == "__main__":
    main()