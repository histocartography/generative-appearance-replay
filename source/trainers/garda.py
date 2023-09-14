import sys
import os
import logging
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from source.models.cond_eql_style_large_generator import CondEqlGenerator
from source.models.cond_eql_large_discriminator import CondEqlDiscriminator
from source.models.vgg19 import Encoder
from source.models.layers import AdaIN
from source.models.unet import U_Net
from source.utils.logger import DiceLogger
from source.utils.saver import BaseSaver, EarlyStoppingSaver
from source.utils.loss import content_loss, seg_loss_kd
from source.utils.io import (
    get_device,
    load_checkpoint,
    save_checkpoint,
    mnms_collate_fn,
    ps_collate_fn,
)
from source.utils.misc import get_n_trainable_params


log = logging.getLogger("GarDATrainer")
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
h1.setFormatter(formatter)
log.addHandler(h1)


class GarDATrainer:
    """GarDA trainer class."""

    def __init__(self, args, task, datasets):
        """Initialize GarDA.

        Args:
            args: configuration arguments
            task: segmentation task
            datasets: datasets
        """
        self.device = get_device()
        self.config = args
        self.task = task
        self.save_path = args.save_path
        self.seg_model_path = args.seg_model_path
        self.gen_model_path = args.gen_model_path
        self.resume_training = args.resume_training
        self.tb_logger = SummaryWriter(log_dir=self.save_path)

        self.domain_id = args.domain_id
        self.latent_dim = args.latent_dim
        self.n_domains = args.n_domains
        self.n_gen_steps = args.n_gen_steps
        self.n_seg_steps = args.n_seg_steps
        self.batch_size = args.batch_size
        self.gan_batch_size = args.gan_batch_size
        self.seg_lr = args.seg_lr
        self.disc_lr = args.disc_lr
        self.gen_lr = args.gen_lr
        self.image_size = args.image_size
        self.n_seg_classes = args.n_seg_classes
        self.background_label = args.background_label
        self.step = 0

        # early stopping config
        self.early_stopping = args.early_stopping
        self.patience = args.patience
        self.stop_epoch = args.stop_epoch
        self.stop_metric = args.stop_metric

        # loss config
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=self.background_label)
        self.kd_temp = args.kd_temp
        self.optim_name = args.optimizer
        self.softplus = torch.nn.Softplus()

        # Initialize dataloaders
        self.gan_loader = DataLoader(
            dataset=datasets[0],
            batch_size=args.gan_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )
        self.gan_loader_iter = enumerate(self.gan_loader)
        self.train_loader = DataLoader(
            dataset=datasets[1],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )
        self.train_loader_iter = enumerate(self.train_loader)
        self.val_loader = DataLoader(dataset=datasets[2], batch_size=args.batch_size, num_workers=args.num_workers)

        if self.task == "ods":
            self.test_loaders = [
                DataLoader(dataset=datasets[3], batch_size=args.batch_size, num_workers=args.num_workers)
            ]
            if args.prev_dataset is not None:
                self.test_loaders.append(
                    DataLoader(
                        dataset=datasets[4],
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    )
                )
            if args.src_dataset is not None:
                self.test_loaders.append(
                    DataLoader(
                        dataset=datasets[5],
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    )
                )
        elif self.task == "cs":
            self.test_loaders = [
                DataLoader(
                    dataset=datasets[3],
                    batch_size=None,
                    num_workers=args.num_workers,
                    collate_fn=mnms_collate_fn,
                )
            ]
            if args.prev_dataset is not None:
                self.test_loaders.append(
                    DataLoader(
                        dataset=datasets[4],
                        batch_size=None,
                        num_workers=args.num_workers,
                        collate_fn=mnms_collate_fn,
                    )
                )
            if args.src_dataset is not None:
                self.test_loaders.append(
                    DataLoader(
                        dataset=datasets[5],
                        batch_size=None,
                        num_workers=args.num_workers,
                        collate_fn=mnms_collate_fn,
                    )
                )
        elif self.task == "ps":
            self.test_loaders = [
                DataLoader(
                    dataset=datasets[3],
                    batch_size=None,
                    num_workers=args.num_workers,
                    collate_fn=ps_collate_fn,
                )
            ]
            if args.prev_dataset is not None:
                for prev_ds in datasets[4]:
                    self.test_loaders.append(
                        DataLoader(
                            dataset=prev_ds,
                            batch_size=None,
                            num_workers=args.num_workers,
                            collate_fn=ps_collate_fn,
                        )
                    )
            if args.src_dataset is not None:
                self.test_loaders.append(
                    DataLoader(
                        dataset=datasets[5],
                        batch_size=None,
                        num_workers=args.num_workers,
                        collate_fn=ps_collate_fn,
                    )
                )
        else:
            raise ValueError(f"Task '{self.task}' is not implemented.")

        # Initialize segmentation model
        self.model = None
        self.prev_model = None

        # Initialize generator
        self.generator = None
        self.generator_ema = None
        self.prev_generator = None

        # Initialize discriminator
        self.discriminator = None

        # Initialize vgg19 encoder
        self.encoder = Encoder()

        # Initialize AdaIN layer
        self.adain = AdaIN(self.encoder.noutchannels)

        # Feature matching config
        self.feat_matching_layer = args.feat_matching_layer

        # GAN training config
        # EMA half-life in 1k imgs (500k imgs proposed in paper)
        self.ema_nk_img = args.ema_nk_img
        self.ema_rampup_div = 4
        # linearly ramp up EMA horizon from 0 to ema_nk_nkimg in x steps
        self.ema_rampup = self.ema_nk_img * 1000 / (self.gan_batch_size * self.n_gen_steps / self.ema_rampup_div)

        # R1 regularization weight (gradient penalty on D(real))
        self.r1_gamma = args.r1_gamma
        if self.r1_gamma is None:
            gamma_0 = 0.0002
            gen_image_size = 128
            self.r1_gamma = gamma_0 * (gen_image_size**2 / self.gan_batch_size)

        # Lazy R1 regularization
        self.r1_interval = args.r1_interval
        self.lazy_c = 1
        self.d_b1 = 0.0
        self.d_b2 = 0.99
        if self.r1_interval > 1:
            self.lazy_c = self.r1_interval / (self.r1_interval + 1)
            self.disc_lr *= self.lazy_c
            self.d_b1 = self.d_b1**self.lazy_c
            self.d_b2 = self.d_b2**self.lazy_c

        # Regularization weights
        self.cont_coefficient = args.cont_coefficient
        self.img_coefficient = args.img_coefficient
        self.img_dist_loss_type = args.img_dist_loss_type
        if self.img_dist_loss_type == "L1":
            self.img_dist_loss_fn = torch.nn.L1Loss(reduction="mean")
        else:
            self.img_dist_loss_fn = torch.nn.MSELoss(reduction="mean")

        # Tensorboard summary config
        self.n_fixed_z = 16
        self.tb_fixed_z = torch.randn(self.n_fixed_z, self.latent_dim, device=self.device)
        self.palette_z = torch.randn(5, self.latent_dim, device=self.device)
        self.ckpt = args.ckpt

    def _init_model(self):
        return U_Net(classes=self.n_seg_classes, in_channels=3 if self.task == "ods" else 1)

    def _init_discriminator(self):
        return CondEqlDiscriminator(
            feat_matching_layer=self.feat_matching_layer,
            device=self.device,
            n_cls=self.n_domains,
            mbdis=True,
        )

    def _init_generator(self):
        return CondEqlGenerator(
            latent_dim=self.latent_dim,
            device=self.device,
            n_cls=self.n_domains,
            out_channels=3 if self.task == "ods" else 1,
        )

    def train_gen(self):
        """Train GAN (G and D)."""
        self.model = self._init_model()
        self.generator = self._init_generator()
        self.discriminator = self._init_discriminator()
        log.info("Number of trainable params in G: {}".format(get_n_trainable_params(self.generator)))
        log.info("Number of trainable params in D: {}".format(get_n_trainable_params(self.discriminator)))

        # initialize the current generator/discriminator to the previous saved ema generator/discriminator checkpoints
        if self.resume_training:
            log.info("Resume training...")
            log.info("Load checkpoints to resume training from {}/***.pt".format(self.gen_model_path))
            self.generator_ema = deepcopy(self.generator)
            self.generator_ema = load_checkpoint(
                self.generator_ema, os.path.join(self.gen_model_path, f"{self.ckpt}K_generator.pt")
            )
            self.generator = load_checkpoint(
                self.generator,
                os.path.join(self.gen_model_path, f"{self.ckpt}K_curr_generator.pt"),
            )
            self.discriminator = load_checkpoint(
                self.discriminator,
                os.path.join(self.gen_model_path, f"{self.ckpt}K_discriminator.pt"),
            )
        else:
            if self.gen_model_path is not None:
                log.info("Load generator checkpoint from {}/***.pt".format(self.gen_model_path))
                self.generator = load_checkpoint(
                    self.generator, os.path.join(self.gen_model_path, f"{self.ckpt}K_generator.pt")
                )
                self.prev_generator = deepcopy(self.generator).eval()
                self.prev_generator.requires_grad(False)
            self.generator_ema = deepcopy(self.generator)

        self.generator_ema.requires_grad(False)
        self.generator_ema.eval()

        self.adain.to(self.device)
        self.encoder.to(self.device)

        # initialize optimizers for generator/discriminator
        optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=self.disc_lr, betas=(self.d_b1, self.d_b2))
        optimizer_gen = optim.Adam(self.generator.parameters(), lr=self.gen_lr, betas=(0.0, 0.99))

        if self.resume_training:
            # load optimizer checkpoint
            log.info("Load optimizer checkpoints to resume training from {}_***.pt".format(self.gen_model_path))
            optimizer_ckpt = torch.load(os.path.join(self.gen_model_path, f"{self.ckpt}K_optimizers.pt"))
            optimizer_disc.load_state_dict(optimizer_ckpt["optimizer_disc"])
            optimizer_gen.load_state_dict(optimizer_ckpt["optimizer_gen"])
            self.step = int(self.ckpt) * 1000

        d_steps = self.step
        g_steps = self.step

        for _ in range(self.n_gen_steps):
            try:
                _, real_curr_images = self.gan_loader_iter.__next__()
            except:
                self.gan_loader_iter = enumerate(self.gan_loader)
                _, real_curr_images = self.gan_loader_iter.__next__()

            real_curr_images = (
                self._resize(real_curr_images, size=self.image_size)
                if real_curr_images.shape[-1] != self.image_size
                else real_curr_images
            )
            real_curr_images = real_curr_images.to(self.device)

            # ratio of previous vs current task samples per batch
            ratio = 0.5
            n_prev = int(ratio * self.gan_batch_size)
            n_curr = self.gan_batch_size - n_prev

            if self.domain_id == 0:
                ######################
                # Update discriminator
                ######################
                # train D
                self.discriminator.requires_grad(True)
                self.discriminator.zero_grad()
                # freeze G
                self.generator.requires_grad(False)

                # sample D inputs (z used later for G update)
                # real_curr_labels --> current domain
                real_curr_labels = self._to_one_hot(
                    torch.randint(low=self.domain_id, high=self.domain_id + 1, size=(self.gan_batch_size,))
                )
                imgs_r, labels_r, imgs_f, labels_f, z = self.sample_d_inputs(
                    real_curr_images, real_curr_labels, n_prev, n_curr
                )

                # D(real)
                d_real = self.discriminator(imgs_r, labels_r)
                # D(fake)
                d_fake = self.discriminator(imgs_f, labels_f)

                # non-saturating D loss
                d_ns_loss = torch.mean(self.softplus(d_fake) + self.softplus(-d_real))
                # (lazy) R1 regularization (gradient penalty on reals)
                d_r1_loss = self._get_r1(imgs_r, labels_r)
                d_loss = d_ns_loss + d_r1_loss

                # backprop
                d_loss.backward()
                optimizer_disc.step()

                ##################
                # Update generator
                ##################
                # train G
                self.generator.requires_grad(True)
                self.generator.zero_grad()
                # freeze D
                self.discriminator.requires_grad(False)

                # generate fakes
                fake_images = self.generator(z)
                # upsample generated images
                fake_images = (
                    self._resize(fake_images, size=self.image_size)
                    if fake_images.shape[-1] != self.image_size
                    else fake_images
                )
                imgs_f = fake_images

                # D(fake)
                d_fake = self.discriminator(imgs_f, labels_f)

                # non-saturating G loss
                g_gan_loss = torch.mean(self.softplus(-d_fake))
                loss_gen = g_gan_loss

                # Backprop
                loss_gen.backward()
                optimizer_gen.step()

                # update ema copy
                with torch.autograd.profiler.record_function("G_ema"):
                    # ramp up linearly from 0 to ema_nk_img*1000
                    ema_nimg = min(self.ema_nk_img * 1000, self.gan_batch_size * g_steps * self.ema_rampup)
                    ema_beta = 0.5 ** (self.gan_batch_size / max(ema_nimg, 1e-8))
                    # lin. interpolate and update
                    for p_ema, p in zip(self.generator_ema.parameters(), self.generator.parameters()):
                        p_ema.copy_(p.lerp(p_ema, ema_beta))
                    # copy buffers
                    for b_ema, b in zip(self.generator_ema.buffers(), self.generator.buffers()):
                        b_ema.copy_(b)

                # log scalars
                if self.step % 100 == 0:
                    metric_dict = {
                        "L_G_GAN": g_gan_loss.item(),
                        "L_D_GAN": d_ns_loss.item(),
                        "L_G": loss_gen.item(),
                        "L_D": d_loss.item(),
                        "D_real": torch.mean(d_real).item(),
                        "D_fake": torch.mean(d_fake).item(),
                    }
                    self.tb_logger.add_scalars("GAN", metric_dict, self.step)

                # palette summary of generated images
                if (self.step % 1000 == 0) & (self.step != 0):
                    log.info(f"Step {self.step}")
                    for i in range(self.domain_id + 1):
                        self._log_ema_samples(
                            z=torch.cat(
                                [
                                    self.tb_fixed_z,
                                    self._to_one_hot(torch.randint(low=i, high=i + 1, size=(self.n_fixed_z,))),
                                ],
                                dim=1,
                            ),
                            T=i,
                        )

                self.step += 1
                g_steps += 1

                # save G chpt
                if self.step % 50000 == 0:
                    log.info("Save checkpoint for step {}".format(self.step))
                    save_checkpoint(
                        self.generator,
                        os.path.join(self.save_path, f"{int(self.step/1000)}K_curr_generator.pt"),
                    )
                    save_checkpoint(
                        self.generator_ema,
                        os.path.join(self.save_path, f"{int(self.step/1000)}K_generator.pt"),
                    )
                    save_checkpoint(
                        self.discriminator,
                        os.path.join(self.save_path, f"{int(self.step/1000)}K_discriminator.pt"),
                    )
                    torch.save(
                        {
                            "optimizer_gen": optimizer_gen.state_dict(),
                            "optimizer_disc": optimizer_disc.state_dict(),
                        },
                        os.path.join(self.save_path, f"{int(self.step/1000)}K_optimizers.pt"),
                    )

                d_steps += 1

            else:
                ######################
                # Update discriminator
                ######################
                # train D
                self.discriminator.requires_grad(True)
                self.discriminator.zero_grad()
                # freeze G
                self.generator.requires_grad(False)

                # sample D inputs (z used later for G update)
                z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
                src_labels = torch.cat(
                    [
                        self._to_one_hot(torch.randint(low=0, high=self.domain_id, size=(n_prev,))),
                        self._to_one_hot(torch.randint(low=0, high=1, size=(self.batch_size - n_prev,))),
                    ],
                    dim=0,
                )
                tgt_labels = self._to_one_hot(
                    torch.randint(low=self.domain_id, high=self.domain_id + 1, size=(self.batch_size,))
                )

                z_prev = torch.cat([z, src_labels], dim=1)
                z_curr = torch.cat([z, torch.cat([src_labels[:n_prev], tgt_labels[n_prev:]], dim=0)], dim=1)

                # noise map
                noise_maps = self.sample_noise_maps(self.gan_batch_size)

                img_real = self.prev_generator(z_prev, noise=noise_maps)
                img_real = torch.clip(img_real, 0.0, 1.0)
                img_fake = self.generator(z_curr, noise=noise_maps)

                # concat along channel dim to make it compatible for vgg19
                real_curr_images = torch.cat([real_curr_images, real_curr_images, real_curr_images], dim=1)
                img_real = torch.cat([img_real, img_real, img_real], dim=1)
                img_fake = torch.cat([img_fake, img_fake, img_fake], dim=1)

                # matching vgg feature for style transfer
                f_src_real = self.encoder(img_real).detach()
                f_tgt_real = self.encoder(real_curr_images[n_prev:]).detach()
                f_fake = self.encoder(img_fake).detach()
                f_src_fake, f_tgt_fake = f_fake[:n_prev], f_fake[n_prev:]

                # renormalize with AdaIN
                f_adain = self.adain(f_src_real[n_prev:], f_tgt_real)

                # D(real)
                d_real = self.discriminator(
                    torch.cat([f_src_real[:n_prev], f_adain], dim=0),
                    torch.cat([src_labels[:n_prev], tgt_labels[n_prev:]], dim=0),
                )
                # D(fake)
                d_fake = self.discriminator(
                    torch.cat([f_src_fake, f_tgt_fake], dim=0),
                    torch.cat([src_labels[:n_prev], tgt_labels[n_prev:]], dim=0),
                )
                # (lazy) R1 regularization (gradient penalty on reals)
                d_r1_loss = self._get_r1(
                    torch.cat([f_src_real[:n_prev], f_adain], dim=0),
                    torch.cat([src_labels[:n_prev], tgt_labels[n_prev:]], dim=0),
                )

                d_ns_loss = torch.mean(self.softplus(d_fake) + self.softplus(-d_real))
                d_loss = d_ns_loss + d_r1_loss

                # backprop
                d_loss.backward()
                optimizer_disc.step()

                ##################
                # Update generator
                ##################
                # train G
                self.generator.requires_grad(True)
                self.generator.zero_grad()
                # freeze D
                self.discriminator.requires_grad(False)

                img_real = self.prev_generator(z_prev, noise=noise_maps)
                img_fake = self.generator(z_curr, noise=noise_maps)

                # concat along channel dim to make it compatible for vgg19
                img_real = torch.cat([img_real, img_real, img_real], dim=1)
                img_fake = torch.cat([img_fake, img_fake, img_fake], dim=1)

                f_src_real = self.encoder(torch.clip(img_real, 0.0, 1.0))
                f_fake = self.encoder(img_fake)

                f_src_fake, f_tgt_fake = f_fake[:n_prev], f_fake[n_prev:]

                d_fake = self.discriminator(
                    torch.cat([f_src_fake, f_tgt_fake], dim=0),
                    torch.cat([src_labels[:n_prev], tgt_labels[n_prev:]], dim=0),
                )

                # content loss
                f_adain_bw = self.adain(f_tgt_fake, f_src_real[n_prev:])
                loss_c = content_loss(f_adain_bw, f_src_real[n_prev:])
                loss_c *= self.cont_coefficient

                # image distillation
                img_dist_loss = 0
                # send half the batch through previous G and compute image distillation on it
                dist_imgs_gt = img_real[:n_prev]
                # upsample generated images
                dist_imgs_gt = (
                    self._resize(dist_imgs_gt, size=self.image_size)
                    if dist_imgs_gt.shape[-1] != self.image_size
                    else dist_imgs_gt
                )
                # fakes for image distillation
                dist_imgs_f = img_fake[:n_prev]

                # image distillation loss
                dist_imgs_gt = dist_imgs_gt.view(n_prev, -1)
                dist_imgs_f = dist_imgs_f.view(n_prev, -1)
                img_dist_loss = self.img_dist_loss_fn(dist_imgs_gt, dist_imgs_f)
                img_dist_loss *= self.img_coefficient

                # non-saturating G loss
                g_gan_loss = torch.mean(self.softplus(-d_fake))

                # total generator loss
                loss_gen = g_gan_loss + img_dist_loss + loss_c

                # backprop
                loss_gen.backward()
                optimizer_gen.step()

                # update ema copy
                with torch.autograd.profiler.record_function("G_ema"):
                    # ramp up linearly from 0 to ema_nk_img*1000
                    ema_nimg = min(self.ema_nk_img * 1000, self.gan_batch_size * g_steps * self.ema_rampup)
                    ema_beta = 0.5 ** (self.gan_batch_size / max(ema_nimg, 1e-8))
                    # linearly interpolate and update
                    for p_ema, p in zip(self.generator_ema.parameters(), self.generator.parameters()):
                        p_ema.copy_(p.lerp(p_ema, ema_beta))
                    # copy buffers
                    for b_ema, b in zip(self.generator_ema.buffers(), self.generator.buffers()):
                        b_ema.copy_(b)

                # log scalars
                if self.step % 100 == 0:
                    metric_dict = {
                        "L_G_GAN": g_gan_loss.item(),
                        "L_D_GAN": d_ns_loss.item(),
                        "L_G": loss_gen.item(),
                        "L_D": d_loss.item(),
                        "L_G_dist": img_dist_loss.item(),
                        "L_C": loss_c.item(),
                        "D_real": torch.mean(d_real).item(),
                        "D_fake": torch.mean(d_fake).item(),
                    }
                    self.tb_logger.add_scalars("GAN", metric_dict, self.step)

                # palette summary of generated images
                if (self.step % 1000 == 0) & (self.step != 0):
                    for i in range(self.domain_id + 1):
                        self._log_ema_samples(
                            z=torch.cat(
                                [
                                    self.tb_fixed_z,
                                    self._to_one_hot(torch.randint(low=i, high=i + 1, size=(self.n_fixed_z,))),
                                ],
                                dim=1,
                            ),
                            T=i,
                        )

                self.step += 1
                g_steps += 1

                # save G chpt
                if self.step % 10000 == 0:
                    log.info("Save checkpoint for step {}".format(self.step))
                    save_checkpoint(
                        self.generator_ema,
                        os.path.join(self.save_path, f"{int(self.step/1000)}K_generator.pt"),
                    )
                    save_checkpoint(
                        self.discriminator,
                        os.path.join(self.save_path, f"{int(self.step/1000)}K_discriminator.pt"),
                    )

                d_steps += 1

    def train_seg(self):
        """Train segmentation model."""
        self.model = self._init_model()
        self.model.train()
        self.model.to(self.device)

        # set optimizer
        if self.optim_name == "Adam":
            optimizer_seg = optim.Adam(self.model.parameters(), lr=self.seg_lr, betas=(0.0, 0.99), weight_decay=0.0005)
        elif self.optim_name == "SGD":
            optimizer_seg = optim.SGD(self.model.parameters(), lr=self.seg_lr, momentum=0.9, weight_decay=0.0005)
        else:
            raise ValueError(f"Optimizer '{self.optim_name}' not available.")

        if self.domain_id == 0:
            # train source segmentation model
            dice_logger = DiceLogger(n_classes=self.n_seg_classes, background_label=self.background_label)
            if self.early_stopping is True:
                model_saver = EarlyStoppingSaver(
                    logger=log,
                    patience=self.patience,
                    stop_epoch=self.stop_epoch,
                    metric=self.stop_metric,
                    save_path=self.save_path,
                )
            else:
                model_saver = BaseSaver(save_path=self.save_path, logger=log)

            log.info("Training source segmentation model...")
            for epoch in range(self.n_seg_steps):
                optimizer_seg.zero_grad()
                train_src_loss = 0.0
                val_src_loss = 0.0
                for _, (img_src, label_src, _) in enumerate(self.train_loader):
                    img_src, label_src = img_src.to(self.device), label_src.to(self.device)
                    logits_src = self.model(img_src)

                    loss_src = self.seg_loss(logits_src, label_src)
                    train_src_loss += loss_src
                    loss_src.backward()

                    optimizer_seg.step()
                    optimizer_seg.zero_grad()

                train_src_loss /= len(self.train_loader)

                with torch.no_grad():
                    for _, (img_src, label_src, mask_src) in enumerate(self.val_loader):
                        img_src, label_src = img_src.to(self.device), label_src.to(self.device)
                        logits_src = self.model(img_src)

                        loss_src = self.seg_loss(logits_src, label_src)
                        val_src_loss += loss_src

                        y_prob = torch.softmax(logits_src, dim=1)
                        y_hat = torch.argmax(y_prob, axis=1)

                        dice_logger.log(y_hat.detach().cpu(), label_src.detach().cpu(), mask_src.bool())

                    _, val_mean_dice_score = dice_logger.get_dice_score()
                    val_src_loss /= len(self.val_loader)
                    val_summary = {"val_loss": val_src_loss, "val_dice": val_mean_dice_score}
                    model_saver(epoch, self.model, val_summary)
                    if model_saver.early_stop:
                        break

                    if epoch % 5 == 0:
                        train_summary_dict = {
                            "train_loss": train_src_loss.item(),
                            "val_loss": val_src_loss.item(),
                            "val_dice": val_mean_dice_score,
                        }
                        self.tb_logger.add_scalars("seg_training", train_summary_dict, epoch)

            log.info("Finished training source segmentation model.")

        else:
            log.info("Training target segmentation model...")
            assert self.seg_model_path is not None, "Previous segmentation model needed"
            assert self.gen_model_path is not None, "Previous generator and discriminator needed"

            # initialize current segmentation model as previous segmentation model
            self.model = load_checkpoint(self.model, os.path.join(self.seg_model_path, "model_best_dice.pt"))
            total_params_unet = get_n_trainable_params(self.model)
            log.info("Number of trainable params in U-Net: {}".format(total_params_unet))
            self.prev_model = deepcopy(self.model)
            self.prev_model = load_checkpoint(self.prev_model, os.path.join(self.seg_model_path, "model_best_dice.pt"))
            self.prev_model.eval()
            self.prev_model.requires_grad(False)

            # load previous generator
            self.prev_generator = self._init_generator()
            self.prev_generator = load_checkpoint(
                self.prev_generator,
                os.path.join(self.gen_model_path, f"{self.ckpt}K_generator.pt"),
            )
            self.prev_generator.eval()
            self.prev_generator.requires_grad(False)

            for i_iter in range(self.n_seg_steps):
                z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
                src_labels = self._to_one_hot(torch.randint(low=0, high=1, size=(self.batch_size,)))
                tgt_labels = self._to_one_hot(torch.randint(low=0, high=self.domain_id + 1, size=(self.batch_size,)))

                # generate source images
                img_src = self.prev_generator(torch.cat([z, src_labels], dim=1))
                img_src = torch.clip(img_src, 0.0, 1.0)

                # generate target images
                img_tgt = self.prev_generator(torch.cat([z, tgt_labels], dim=1))
                img_tgt = torch.clip(img_tgt, 0.0, 1.0)

                # generate ground truth
                logits = self.prev_model(img_src)
                label = torch.argmax(torch.softmax(logits, dim=1), dim=1)

                # pass the target images (from all seen domains) to train the model
                logits_tgt = self.model(img_tgt)
                loss_tgt = seg_loss_kd(pred_logits=logits_tgt, true_logits=logits, temp=self.kd_temp)

                loss = loss_tgt
                loss.backward()

                optimizer_seg.step()
                optimizer_seg.zero_grad()

                if i_iter % 10 == 0:
                    with torch.no_grad():
                        logits = self.model(img_src)
                    n_imgs_per_side = int(np.sqrt(self.batch_size))
                    img_src = make_grid(img_src, n_imgs_per_side, n_imgs_per_side)
                    img_tgt = make_grid(img_tgt, n_imgs_per_side, n_imgs_per_side)

                    gt_src_map = torch.zeros(label.shape)
                    gt_src_map[label != 0] = 1
                    gt_src_map = make_grid(gt_src_map.unsqueeze(dim=1), n_imgs_per_side, n_imgs_per_side)

                    pred_src_map = torch.zeros(label.shape)
                    pred_src_map[torch.argmax(torch.softmax(logits, dim=1), dim=1) != 0] = 1
                    pred_src_map = make_grid(pred_src_map.unsqueeze(dim=1), n_imgs_per_side, n_imgs_per_side)

                    pred_tgt_map = torch.zeros(label.shape)
                    pred_tgt_map[torch.argmax(torch.softmax(logits_tgt, dim=1), dim=1) != 0] = 1
                    pred_tgt_map = make_grid(pred_tgt_map.unsqueeze(dim=1), n_imgs_per_side, n_imgs_per_side)
                    self.tb_logger.add_image("source_images", img_src, int(i_iter // 20))
                    self.tb_logger.add_image("target_images", img_tgt, int(i_iter // 20))
                    self.tb_logger.add_image("source_pred", pred_src_map, int(i_iter // 20))
                    self.tb_logger.add_image("source_gt", gt_src_map, int(i_iter // 20))
                    self.tb_logger.add_image("target_pred", pred_tgt_map, int(i_iter // 20))
                    self.tb_logger.add_scalar("seg_loss", loss_tgt.item(), int(i_iter // 20))

                save_checkpoint(self.model, os.path.join(self.save_path, f"model_best_dice.pt"))

            log.info("Finished training target segmentation model.")

    def test_seg(self, during_training=False, i_iter=0):
        """Test segmentation model.

        Args:
            during_training: whether to test with the model currently being trained. Defaults to False.
            i_iter: iteration. Defaults to 0.
        """
        if not during_training:
            self.model = self._init_model()
            self.model.to(self.device)
            self.model = load_checkpoint(self.model, os.path.join(self.seg_model_path, "model_best_dice.pt"))
            self.model.eval()

        for test_loader in self.test_loaders:
            dataset = test_loader.dataset.dataset
            split = test_loader.dataset.split

            if self.task == "ods":
                dice_logger = DiceLogger(n_classes=self.n_seg_classes, background_label=self.background_label)
                with torch.no_grad():
                    for _, (img_tgt, label_tgt, mask_tgt) in enumerate(test_loader):
                        img_tgt, label_tgt = img_tgt.to(self.device), label_tgt.to(self.device)
                        logits_tgt = self.model(img_tgt)

                        y_prob = torch.softmax(logits_tgt, dim=1)
                        y_hat = torch.argmax(y_prob, axis=1)

                        dice_logger.log(y_hat.detach().cpu(), label_tgt.detach().cpu(), mask_tgt.bool())

                    _, test_mean_dice_score = dice_logger.get_dice_score()
                    log.info(f"dice_{dataset}_{split}: {test_mean_dice_score}")
                    self.tb_logger.add_scalar(f"dice_{dataset}_{split}", test_mean_dice_score, int(i_iter // 20))

            if self.task == "cs":
                with torch.no_grad():
                    for phase in ["ED", "ES"]:
                        overall_test_mean_dice_score = 0
                        overall_test_MYO_dice_score = 0
                        overall_test_LV_dice_score = 0
                        overall_test_RV_dice_score = 0

                        i_patient = 0
                        for _, (pat_ed, pat_es) in enumerate(test_loader):
                            if phase == "ED":
                                pat = pat_ed
                            else:
                                pat = pat_es
                            img_tgt, label_tgt, mask_tgt = pat
                            patient_dice_logger = DiceLogger(
                                n_classes=self.n_seg_classes,
                                background_label=self.background_label,
                            )
                            img_tgt, label_tgt = img_tgt.to(self.device), label_tgt.to(self.device)
                            logits_tgt = self.model(img_tgt)

                            y_prob = torch.softmax(logits_tgt, dim=1)
                            y_hat = torch.argmax(y_prob, axis=1)

                            for i in range(y_hat.shape[0]):
                                patient_dice_logger.log(
                                    y_hat[i].detach().cpu(),
                                    label_tgt[i].detach().cpu(),
                                    np.array(mask_tgt[i], dtype=bool),
                                )

                            # dice score per patient
                            (
                                patient_test_dice_score,
                                patient_test_mean_dice_score,
                            ) = patient_dice_logger.get_dice_score()
                            patient_test_MYO_dice_score = patient_test_dice_score[1]
                            patient_test_LV_dice_score = patient_test_dice_score[2]
                            patient_test_RV_dice_score = patient_test_dice_score[3]

                            i_patient += 1

                            # dice score accumulated
                            overall_test_mean_dice_score += patient_test_mean_dice_score
                            overall_test_MYO_dice_score += patient_test_MYO_dice_score
                            overall_test_LV_dice_score += patient_test_LV_dice_score
                            overall_test_RV_dice_score += patient_test_RV_dice_score

                        # mean dice score over all patients
                        overall_test_mean_dice_score /= i_patient
                        overall_test_MYO_dice_score /= i_patient
                        overall_test_LV_dice_score /= i_patient
                        overall_test_RV_dice_score /= i_patient

                        metric_dict = {
                            f"dice_{dataset}_{split}_mean_{phase}": overall_test_mean_dice_score,
                            f"dice_{dataset}_{split}_MYO_{phase}": overall_test_MYO_dice_score,
                            f"dice_{dataset}_{split}_LV_{phase}": overall_test_LV_dice_score,
                            f"dice_{dataset}_{split}_RV_{phase}": overall_test_RV_dice_score,
                        }
                        log.info(f"dice_{dataset}_{split}_mean_{phase}: {overall_test_mean_dice_score}")
                        self.tb_logger.add_scalars("test_seg", metric_dict, int(i_iter // 20))

            elif self.task == "ps":
                with torch.no_grad():
                    test_mean_dice_score = 0
                    i_patient = 0
                    for _, pat in enumerate(test_loader):
                        img_tgt, label_tgt, mask_tgt = pat
                        patient_dice_logger = DiceLogger(
                            n_classes=self.n_seg_classes, background_label=self.background_label
                        )
                        img_tgt, label_tgt = img_tgt.to(self.device), label_tgt.to(self.device)
                        logits_tgt = self.model(img_tgt)

                        y_prob = torch.softmax(logits_tgt, dim=1)
                        y_hat = torch.argmax(y_prob, axis=1)

                        for i in range(y_hat.shape[0]):
                            patient_dice_logger.log(
                                y_hat[i].detach().cpu(),
                                label_tgt[i].detach().cpu(),
                                np.array(mask_tgt[i], dtype=bool),
                            )

                        # dice score per patient
                        _, patient_test_mean_dice_score = patient_dice_logger.get_dice_score()
                        i_patient += 1

                        # dice score accumulated
                        test_mean_dice_score += patient_test_mean_dice_score

                    # mean dice score over all patients
                    test_mean_dice_score /= i_patient
                    log.info(f"dice_{dataset}: {test_mean_dice_score}")
                    self.tb_logger.add_scalar(f"dice_{dataset}", test_mean_dice_score, int(i_iter // 20))
            else:
                raise ValueError(f"Task '{self.task}' is not implemented.")

    def sample_d_inputs(self, curr_images, curr_labels, n_prev, n_curr):
        """Sample discriminator inputs for a batch.

        Args:
            curr_images: current images
            curr_labels: labels of current domain
            n_prev: number of previous domain images in a batch
            n_curr: number of current domain images in a batch

        Returns:
            real images and labels, fake images and labels, latent noise vectors
        """
        # real D input
        if self.prev_generator is None:
            imgs_r = curr_images
            labels_r = curr_labels
            z = self.sample_z(self.gan_batch_size, mode="current")
        else:
            # divide batch in previous (pseudo-real) and current (real) task samples
            z_prev = self.sample_z(n_prev, mode="previous")
            prev_imgs = self.prev_generator(z_prev).detach()
            # upsample generated images
            prev_imgs = (
                self._resize(prev_imgs, size=self.image_size) if prev_imgs.shape[-1] != self.image_size else prev_imgs
            )
            # combine previous (pseudo-real) and current (real) to form real batch for D
            imgs_r = torch.cat([prev_imgs, curr_images[:n_curr]], dim=0)
            labels_r = torch.cat([z_prev[:, self.latent_dim :], curr_labels[:n_curr]], dim=0)
            z_curr = self.sample_z(n_curr, mode="current")
            z = torch.cat([z_prev, z_curr], dim=0)

        # fake D input
        imgs_f = self.generator(z).detach()
        # upsample generated images
        imgs_f = self._resize(imgs_f, size=self.image_size) if imgs_f.shape[-1] != self.image_size else imgs_f
        labels_f = z[:, self.latent_dim :]

        return imgs_r, labels_r, imgs_f, labels_f, z

    def sample_z(self, n_samples, mode):
        """Sample a batch of latent noise vectors.

        Args:
            n_samples: number of samples
            mode: previous (for previous domains) or current (for current domain)

        Returns:
            latent noise vectors concatenated with domain labels
        """
        # sample latent
        z = torch.randn(n_samples, self.latent_dim, device=self.device)

        if mode == "previous":
            # randomly sampled labels from previous domains
            sampled_prev_labels = torch.randint(low=0, high=self.domain_id, size=(n_samples,))
            one_hot_labels = self._to_one_hot(sampled_prev_labels)
        elif mode == "current":
            # labels only from current domain
            sampled_curr_labels = torch.randint(low=self.domain_id, high=self.domain_id + 1, size=(n_samples,))
            one_hot_labels = self._to_one_hot(sampled_curr_labels)
        else:
            raise ValueError(f"Unknown sampling mode '{mode}'.")

        # concat latent and onehot
        return torch.cat([z, one_hot_labels], dim=1)

    def sample_noise_maps(self, n_samples):
        """Sample noise maps for the generator.

        Args:
            n_samples: number of samples

        Returns:
            dictionary of noise maps
        """
        noise_maps = {
            1: torch.randn(n_samples, 1, 4, 4, device=self.device),
            2: torch.randn(n_samples, 1, 4, 4, device=self.device),
            3: torch.randn(n_samples, 1, 8, 8, device=self.device),
            4: torch.randn(n_samples, 1, 16, 16, device=self.device),
            5: torch.randn(n_samples, 1, 32, 32, device=self.device),
            6: torch.randn(n_samples, 1, 64, 64, device=self.device),
            7: torch.randn(n_samples, 1, 128, 128, device=self.device),
            8: torch.randn(n_samples, 1, 128, 128, device=self.device),
            9: torch.randn(n_samples, 1, 256, 256, device=self.device),
        }
        return noise_maps

    def _get_r1(self, real_img, real_labels):
        """Compute (lazy) R1 regularization.

        Args:
            real_img: real images
            real_labels: domain labels of real images

        Returns:
            R1 regularization loss
        """
        # lazy R1 regularization: don't compute for every mini-batch
        if self.step % self.r1_interval == 0:
            # gradients w.r.t. image
            if self.feat_matching_layer == "multiple":
                out1_r, out2_r, out3_r, out4_r = real_img
                out1_r = out1_r.detach().requires_grad_(True)
                out2_r = out2_r.detach().requires_grad_(True)
                out3_r = out3_r.detach().requires_grad_(True)
                out4_r = out4_r.detach().requires_grad_(True)
                real_img = (out1_r, out2_r, out3_r, out4_r)
                real_logits = self.discriminator(real_img, real_labels)
                r1_grad = torch.autograd.grad(
                    outputs=[real_logits.sum()],
                    inputs=[real_img[0], real_img[1], real_img[2], real_img[3]],
                    create_graph=True,
                    only_inputs=True,
                )[0]
            else:
                real_img = real_img.detach().requires_grad_(True)
                real_logits = self.discriminator(real_img, real_labels)
                r1_grad = torch.autograd.grad(
                    outputs=[real_logits.sum()],
                    inputs=[real_img],
                    create_graph=True,
                    only_inputs=True,
                )[0]
            r1_penalty = r1_grad.square().sum(dim=[1, 2, 3])
            r1_loss = r1_penalty.mean() * (self.r1_gamma / 2) * self.lazy_c

            # log r1
            if self.r1_interval > 1 or (self.r1_interval == 1 and self.step % 100 == 0):
                self.tb_logger.add_scalar("d_r1", r1_loss.item(), self.step)
        else:
            r1_loss = 0.0

        return r1_loss

    def _resize(self, images, size):
        images = transforms.Resize(size)(images)
        return images

    def _to_one_hot(self, labels):
        n_samples = len(labels)
        one_hot_encoding = np.zeros((n_samples, self.n_domains), dtype=np.float32)
        for i_label, label in enumerate(labels):
            one_hot_encoding[i_label, label] = 1.0

        return torch.tensor(one_hot_encoding, dtype=torch.float32, device=self.device)

    def _log_ema_samples(self, z, T):
        imgs = self.generator_ema(z)
        imgs = self._prepare_for_plotting(imgs)
        n_imgs_per_side = int(np.sqrt(self.n_fixed_z))
        imgs = make_grid(imgs, n_imgs_per_side, n_imgs_per_side)
        self.tb_logger.add_image("fake_imgs_T{}".format(T), imgs, self.step)

    def _prepare_for_plotting(self, imgs):
        imgs = torch.clip(imgs, 0.0, 1.0)
        return imgs
