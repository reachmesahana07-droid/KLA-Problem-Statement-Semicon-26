# KLA-Problem-Statement-Semicon-26
Repository created by Sahana.S,Shri Bargavi.S,Shivani,Shri Vatsan from SSNCE for KLA problem statement in Semicon 2026
KLA Image Restoration — Evaluation Package
This repository contains the evaluation component for the KLA image-restoration challenge.
The model is expected to restore a degraded single-channel grayscale image that may contain:
Speckle noise
Gaussian noise
2× spatial resolution reduction
Combinations of the above degradations
The input is a low-resolution/noisy image and the expected output is the corresponding clean, full-resolution image.
1. Repository contents
```text
.
├── README.md
├── evaluate.py
├── checkpoints/
│   └── best_model.pth
├── test_images/
└── outputs/
```
`evaluate.py` is a standalone Python script. It accepts an input directory and an output directory, loads the trained model, runs inference on every `.npy` image, and writes the restored images to the requested output directory.
2. Requirements
Recommended environment:
Python 3.10+
PyTorch 2.x
NumPy
Install dependencies:
```bash
pip install torch numpy
```
For GPU inference, install the appropriate PyTorch build for the CUDA version available on the evaluation machine.
3. Model checkpoint
The evaluation script uses the lightweight `RestorationNet` architecture defined inside `evaluate.py`.
The training code must save the best checkpoint as:
```text
checkpoints/best_model.pth
```
The preferred checkpoint format is:
```python
{
    "model_state_dict": model.state_dict(),
    "scale_factor": 2,
    "output_min": 0.0,
    "output_max": 1.0
}
```
A plain PyTorch state dictionary is also accepted.
The architecture is deliberately kept inside the evaluation script so the reviewer does not need to edit the script or import a notebook.
4. Input format
The challenge images are grayscale NumPy arrays.
Example:
```text
test_images/
├── 000001.npy
├── 000002.npy
├── 000003.npy
└── ...
```
Each file should contain a 2-D floating-point array:
```text
(H, W)
```
The model performs 2× spatial upscaling:
```text
128 × 128  →  256 × 256
256 × 256  →  512 × 512
```
The input intensity is not clipped before inference. This is important because the challenge explicitly states that speckle noise can push degraded-image values outside the ground-truth intensity range.
5. Running evaluation
From the repository root:
```bash
python evaluate.py --input_dir ./test_images --output_dir ./outputs
```
To specify another checkpoint:
```bash
python evaluate.py \
    --input_dir ./test_images \
    --output_dir ./outputs \
    --checkpoint ./checkpoints/best_model.pth
```
The script automatically:
Finds every `.npy` file in the input directory.
Loads each image as a float32 grayscale array.
Adds the channel dimension required by the neural network.
Runs inference using CPU or CUDA automatically.
Restores the image to 2× its input resolution.
Clips the prediction to the configured ground-truth intensity range.
Saves the result using the original filename.
Example:
```text
test_images/000298.npy
        ↓
    RestorationNet
        ↓
outputs/000298.npy
```
6. Output format
Restored images are written as NumPy `.npy` files containing a 2-D `float32` array.
For example:
```text
Input:  (128, 128)
Output: (256, 256)
```
or:
```text
Input:  (256, 256)
Output: (512, 512)
```
No RGB conversion is performed.
7. Important implementation details
Intensity handling
The degraded image may contain values below 0 or above 1 because of the specified noise process. Therefore, the evaluator does not perform:
```python
np.clip(input, 0, 1)
```
before inference.
Only the final restored prediction is clipped to the ground-truth range stored in the checkpoint (default `0.0–1.0`).
Speed
The evaluator uses:
`torch.inference_mode()`
automatic CUDA selection when available
batch processing
a lightweight convolutional restoration network
The model is designed to provide a practical accuracy/speed trade-off rather than using an unnecessarily large architecture.
Out-of-distribution robustness
The network is fully convolutional and does not depend on a fixed image content layout. Training should therefore use augmentation and a diverse mixture of the supplied semiconductor structures rather than memorizing individual images.
8. Expected training setup
The paired training data should be used as:
```text
degraded/noisy low-resolution image
                    ↓
                 MODEL
                    ↓
clean full-resolution ground truth
```
The training objective should strongly penalize pixel reconstruction error while preserving edges and fine structures.
A recommended starting point is a combination of:
L1 reconstruction loss
SSIM/perceptual structural loss
random flips/rotations for augmentation
validation-based checkpoint selection
The final checkpoint used by `evaluate.py` must correspond to the architecture defined in that file.
9. Example reviewer workflow
A reviewer should be able to clone the repository and run:
```bash
pip install -r requirements.txt
python evaluate.py --input_dir ./test_images --output_dir ./outputs
```
No source-code edits should be required.
10. Notes about the supplied test archive
The supplied `Test_NoisyLR.zip` contains `.npy` grayscale arrays under:
```text
NoisyLR/
```
The current archive contains 400 NumPy test images, and the inspected samples are `128 × 128` `float32` arrays. Some values are outside `[0, 1]`, which is consistent with the challenge's warning about noise-induced intensity excursions.
Before evaluation, extract the archive so that the `.npy` files are available under a local test directory, for example:
```text
test_images/
├── 000001.npy
├── ...
└── 000400.npy
```
11. Reproducibility
The evaluation script does not depend on a Jupyter notebook or manual notebook state. All inference-time model definitions and preprocessing required by the checkpoint are contained in 'evaluate.py'.
