import os
import copy
import random
import numpy as np
import torch
import tqdm
import wandb
from torch.utils.data import DataLoader
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

from uwlab.policy.cfm_pcd_policy import CFMPCDPolicy


class TrainCFMWorkspace:
    def __init__(self, cfg, dataset, policy: CFMPCDPolicy):
        # seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.cfg = cfg
        self.dataset = dataset
        self.model = policy

        self.ema_model = copy.deepcopy(policy)
        self.optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=cfg.training.lr,
            betas=tuple(cfg.training.get("betas", [0.9, 0.999])),
            weight_decay=cfg.training.weight_decay,
        )

        self.global_step = 0
        self.epoch = 0
        self.best_val_mse = float("inf")

        # wandb
        self.use_wandb = cfg.training.get("use_wandb", False)
        if self.use_wandb:
            wandb.init(
                project=cfg.training.wandb_project,
                name=cfg.training.get("wandb_run_name", None),
                config=dict(cfg),
            )

    def run(self):
        cfg = self.cfg
        device = torch.device(cfg.training.device)

        self.model.to(device)
        self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # normalizer
        normalizer = self.dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        self.ema_model.set_normalizer(normalizer)
        self.model.to(device)
        self.ema_model.to(device)

        train_dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.dataloader.batch_size,
            num_workers=cfg.dataloader.num_workers,
            shuffle=True,
            pin_memory=cfg.dataloader.get("pin_memory", True),
            persistent_workers=cfg.dataloader.get("persistent_workers", False),
        )
        val_dataset = self.dataset.get_validation_dataset()
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.dataloader.batch_size,
            num_workers=cfg.dataloader.num_workers,
            shuffle=False,
            pin_memory=cfg.dataloader.get("pin_memory", True),
            persistent_workers=cfg.dataloader.get("persistent_workers", False),
        )

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=len(train_dataloader) * cfg.training.num_epochs,
            last_epoch=self.global_step - 1,
        )

        ema_cfg = cfg.get("ema", {})
        ema: EMAModel = EMAModel(
            model=self.ema_model,
            update_after_step=ema_cfg.get("update_after_step", 0),
            inv_gamma=ema_cfg.get("inv_gamma", 1.0),
            power=ema_cfg.get("power", 0.75),
            min_value=ema_cfg.get("min_value", 0.0),
            max_value=ema_cfg.get("max_value", 0.9999),
        )

        ckpt_dir = cfg.training.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        # resume
        if cfg.training.get("resume", False):
            ckpt_path = os.path.join(ckpt_dir, "latest.ckpt")
            if os.path.isfile(ckpt_path):
                print(f"Resuming from {ckpt_path}")
                self._load_checkpoint(ckpt_path)

        for _ in range(cfg.training.num_epochs):
            # ---- train ----
            self.model.train()
            train_losses = []
            with tqdm.tqdm(train_dataloader, desc=f"Epoch {self.epoch}", leave=False) as pbar:
                for batch in pbar:
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    loss = self.model.compute_loss(batch)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()
                    ema.step(self.model)

                    loss_val = loss.item()
                    train_losses.append(loss_val)
                    pbar.set_postfix(loss=loss_val)
                    self.global_step += 1

            train_loss = np.mean(train_losses)
            log = {"train_loss": train_loss, "epoch": self.epoch, "lr": lr_scheduler.get_last_lr()[0]}

            # ---- val ----
            if self.epoch % cfg.training.val_every == 0:
                self.ema_model.eval()
                val_losses, val_mses = [], []
                with torch.no_grad():
                    for batch in val_dataloader:
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        loss = self.model.compute_loss(batch)
                        val_losses.append(loss.item())

                        result = self.ema_model.predict_action(batch["obs"])
                        mse = torch.nn.functional.mse_loss(
                            result["action_pred"], batch["action"].to(device)
                        )
                        val_mses.append(mse.item())

                val_loss = np.mean(val_losses)
                val_mse = np.mean(val_mses)
                log["val_loss"] = val_loss
                log["val_action_mse"] = val_mse

                if val_mse < self.best_val_mse:
                    self.best_val_mse = val_mse
                    self._save_checkpoint(os.path.join(ckpt_dir, "best.ckpt"))

            # ---- checkpoint ----
            if self.epoch % cfg.training.checkpoint_every == 0:
                self._save_checkpoint(os.path.join(ckpt_dir, "latest.ckpt"))
                self._save_checkpoint(os.path.join(ckpt_dir, f"epoch_{self.epoch:04d}.ckpt"))

            if self.use_wandb:
                wandb.log(log, step=self.global_step)

            self.epoch += 1

        if self.use_wandb:
            wandb.finish()

    def _save_checkpoint(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_mse": self.best_val_mse,
        }, path)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.ema_model.load_state_dict(ckpt["ema_model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["global_step"]
        self.epoch = ckpt["epoch"]
        self.best_val_mse = ckpt.get("best_val_mse", float("inf"))
