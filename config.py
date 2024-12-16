import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "dataset"
TEST_DIR = "reference"
OUTPUT_DIR = "output"
BATCH_SIZE = 1
LEARNING_RATE = 0.0002
LAMBDA_CYCLE = 10
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_BW = "checkpoints/gen_BW.pth.tar"
CHECKPOINT_GEN_C = "checkpoints/gen_C.pth.tar"
CHECKPOINT_CRITIC_BW = "checkpoints/critic_BW.pth.tar"
CHECKPOINT_CRITIC_C = "checkpoints/critic_C.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256, interpolation=3),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
    is_check_shapes=False
)
