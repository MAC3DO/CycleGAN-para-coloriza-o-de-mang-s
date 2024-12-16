import torch
from dataset import MangaDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(
    disc_BW, disc_C, gen_C, gen_BW, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    BW_reals = 0
    BW_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (color, blackwhite) in enumerate(loop):
        color = color.to(config.DEVICE)
        blackwhite = blackwhite.to(config.DEVICE)

        # Treinamento dos Discriminators
        with torch.amp.autocast('cuda'):
            fake_blackwhite = gen_BW(color)
            D_BW_real = disc_BW(blackwhite)
            D_BW_fake = disc_BW(fake_blackwhite.detach())
            BW_reals += D_BW_real.mean().item()
            BW_fakes += D_BW_fake.mean().item()
            D_BW_real_loss = mse(D_BW_real, torch.ones_like(D_BW_real))
            D_BW_fake_loss = mse(D_BW_fake, torch.zeros_like(D_BW_fake))
            D_BW_loss = D_BW_real_loss + D_BW_fake_loss

            fake_color = gen_C(blackwhite)
            D_C_real = disc_C(color)
            D_C_fake = disc_C(fake_color.detach())
            D_C_real_loss = mse(D_C_real, torch.ones_like(D_C_real))
            D_C_fake_loss = mse(D_C_fake, torch.zeros_like(D_C_fake))
            D_C_loss = D_C_real_loss + D_C_fake_loss

            D_loss = (D_BW_loss + D_C_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Treinamento dos Generators 
        with torch.amp.autocast('cuda'):
            # loss adversaria para geradorees
            D_BW_fake = disc_BW(fake_blackwhite)
            D_C_fake = disc_C(fake_color)
            loss_G_BW = mse(D_BW_fake, torch.ones_like(D_BW_fake))
            loss_G_C = mse(D_C_fake, torch.ones_like(D_C_fake))

            # loss ciclica
            cycle_color = gen_C(fake_blackwhite)
            cycle_blackwhite = gen_BW(fake_color)
            cycle_color_loss = l1(color, cycle_color)
            cycle_blackwhite_loss = l1(blackwhite, cycle_blackwhite)

            # loss total
            G_loss = (
                loss_G_C
                + loss_G_BW
                + cycle_color_loss * config.LAMBDA_CYCLE
                + cycle_blackwhite_loss * config.LAMBDA_CYCLE
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_blackwhite * 0.5 + 0.5, f"saved_images/fake_blackwhite_{idx}.png")
            save_image(fake_color * 0.5 + 0.5, f"saved_images/fake_color_{idx}.png")
            save_image(color * 0.5 + 0.5, f"saved_images/real_color_{idx}.png")
            save_image(blackwhite * 0.5 + 0.5, f"saved_images/real_blackwhite_{idx}.png")

        loop.set_postfix(BW_real=BW_reals / (idx + 1), BW_fake=BW_fakes / (idx + 1))


def main():
    disc_BW = Discriminator(in_channels=1).to(config.DEVICE)
    disc_C = Discriminator(in_channels=3).to(config.DEVICE)
    gen_C = Generator(in_channels=1, out_channels=3, num_residuals=9).to(config.DEVICE)
    gen_BW = Generator(in_channels=3, out_channels=1, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_BW.parameters()) + list(disc_C.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_C.parameters()) + list(gen_BW.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
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
        load_checkpoint(
            config.CHECKPOINT_CRITIC_BW,
            disc_BW,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_C,
            disc_C,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = MangaDataset(
        root_blackwhite=config.TRAIN_DIR + "/bw",
        root_color=config.TRAIN_DIR + "/c",
        transform=config.transforms,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    g_scaler = torch.amp.GradScaler('cuda')
    d_scaler = torch.amp.GradScaler('cuda')
    
    lr_scheduler_disc = optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=(config.NUM_EPOCHS//2), eta_min=0)
    lr_scheduler_gen = optim.lr_scheduler.CosineAnnealingLR(opt_gen, T_max=(config.NUM_EPOCHS//2), eta_min=0)

    for epoch in range(config.NUM_EPOCHS):
        print(f"EPOCH - {epoch}")
        train_fn(
            disc_BW,
            disc_C,
            gen_C,
            gen_BW,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_BW, opt_gen, filename=config.CHECKPOINT_GEN_BW)
            save_checkpoint(gen_C, opt_gen, filename=config.CHECKPOINT_GEN_C)
            save_checkpoint(disc_BW, opt_disc, filename=config.CHECKPOINT_CRITIC_BW)
            save_checkpoint(disc_C, opt_disc, filename=config.CHECKPOINT_CRITIC_C)
            
        # learning rate decay
        if epoch >= (config.NUM_EPOCHS // 2):
            lr_scheduler_disc.step()
            lr_scheduler_gen.step()
            print(lr_scheduler_disc.get_last_lr()[0], lr_scheduler_gen.get_last_lr()[0])


if __name__ == "__main__":
    main()
