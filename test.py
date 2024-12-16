import torch
from dataset import MangaDataset
import sys
from utils import load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from generator_model import Generator


def test_fn(gen_C, gen_BW, loader):
    loop = tqdm(loader, leave=True)

    for idx, (color, blackwhite) in enumerate(loop):
        color = color.to(config.DEVICE)
        blackwhite = blackwhite.to(config.DEVICE)

        with torch.amp.autocast('cuda'):
            fake_blackwhite = gen_BW(color)
            fake_color = gen_C(blackwhite)
            
        save_image(fake_blackwhite * 0.5 + 0.5, config.OUTPUT_DIR + f"/blackwhite_out_{idx}.png")
        save_image(fake_color * 0.5 + 0.5, config.OUTPUT_DIR + f"/color_out_{idx}.png")
        save_image(blackwhite * 0.5 + 0.5, config.OUTPUT_DIR + f"/blackwhite_ref_{idx}.png")
        save_image(color * 0.5 + 0.5, config.OUTPUT_DIR + f"/color_ref_{idx}.png")



def main():
    gen_C = Generator(in_channels=1, out_channels=3, num_residuals=9).to(config.DEVICE)
    gen_BW = Generator(in_channels=3, out_channels=1, num_residuals=9).to(config.DEVICE)

    opt_gen = optim.Adam(
        list(gen_C.parameters()) + list(gen_BW.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    load_checkpoint(
        config.CHECKPOINT_GEN_BW,
        gen_BW,
        opt_gen,
        config.LEARNING_RATE,
    )
    load_checkpoint(
        config.CHECKPOINT_GEN_C,
        gen_C,
        opt_gen,
        config.LEARNING_RATE,
    )

    dataset = MangaDataset(
        root_blackwhite=config.TEST_DIR + "/bw",
        root_color=config.TEST_DIR + "/c",
        transform=config.transforms,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    test_fn(gen_C, gen_BW, loader)

if __name__ == "__main__":
    main()
