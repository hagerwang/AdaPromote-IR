import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import AdaIRTrainDataset
from net.model import DGIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
# import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class DGIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = DGIR(is_train=True)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch



        d2c, c2d, l_d2c, l_c2d = self.net(degrad_patch, clean_patch)

        loss_d2c = self.loss_fn(d2c, clean_patch) * 25
        loss_c2d = self.loss_fn(c2d, degrad_patch) * 10
        loss_sim = 0
        for ind, (dcs, cds) in enumerate(zip(l_d2c, l_c2d)):
            sim_current = torch.cosine_similarity(dcs.squeeze(-1).squeeze(-1), cds.squeeze(-1).squeeze(-1).detach(),
                                                  dim=-1).mean()

            # sim_current = torch.cosine_similarity(dcs.squeeze(-1).squeeze(-1), cds.squeeze(-1).squeeze(-1),
            #                                       dim=-1).mean()
            sim_current = 1 - sim_current
            sim_current = sim_current * (0.25*(ind+1))
            self.log("5_semantic_loss_{}".format(str(ind)), sim_current)
            loss_sim += sim_current

        # loss_sim = torch.cosine_similarity(l_d2c.squeeze(-1).squeeze(-1), l_c2d.squeeze(-1).squeeze(-1).detach(), dim=-1)
        loss_semantic = loss_sim / 50

        loss = loss_d2c + loss_c2d + loss_semantic
        # loss = loss_d2c + loss_c2d
        self.log("1_loss", loss)
        self.log("2_d2c_loss", loss_d2c)
        self.log("3_c2d_loss", loss_c2d)

        # d2c_psnr, d2c_ssim = 0, 0
        # c2d_psnr, c2d_ssim = 0, 0
        #
        # d2c_numpy = d2c.cpu().detach().permute(0, 2, 3, 1).numpy()
        # c2d_numpy = c2d.cpu().detach().permute(0, 2, 3, 1).numpy()
        # clean_numpy = clean_patch.cpu().detach().permute(0, 2, 3, 1).numpy()
        # degrad_numpy = degrad_patch.cpu().detach().permute(0, 2, 3, 1).numpy()
        #
        # for d2c_n, c2d_n, clean_n, degrad_n in zip(d2c_numpy, c2d_numpy, clean_numpy, degrad_numpy):
        #     d2c_psnr += peak_signal_noise_ratio(d2c_n, clean_n, data_range=1)
        #     d2c_ssim += structural_similarity(d2c_n, clean_n, data_range=1, channel_axis=-1)
        #     c2d_psnr += peak_signal_noise_ratio(c2d_n, degrad_n, data_range=1)
        #     c2d_ssim += structural_similarity(c2d_n, degrad_n, data_range=1, channel_axis=-1)
        # self.log("d2c_psnr", d2c_psnr / len(d2c_numpy))
        # self.log("d2c_ssim", d2c_ssim / len(d2c_numpy))
        # self.log("c2d_psnr", c2d_psnr / len(d2c_numpy))
        # self.log("c2d_ssim", c2d_ssim / len(d2c_numpy))
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=180)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=96)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=120)

        return [optimizer],[scheduler]


def main():
    print("Options")
    print(opt)
    # if opt.wblogger is not None:
    #     logger  = WandbLogger(project=opt.wblogger,name="AdaIR-Train")
    # else:
    logger = TensorBoardLogger(save_dir = "logs/")

    trainset = AdaIRTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    model = DGIRModel()
    
    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)
    # trainer.fit(model=model, train_dataloaders=trainloader, ckpt_path="1.ckpt")


if __name__ == '__main__':
    main()