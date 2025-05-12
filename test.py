import os
import sys
from email.policy import strict

import numpy as np
import argparse
import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import DGIR
import matplotlib
import matplotlib.pyplot as plt

def save_mask(masks, save_dir):
    matplotlib.use('TkAgg')
    # for i, mask in enumerate(masks):
    #     fig, ax = plt.subplots()
    #     vis_prob = mask[0].cpu().detach().numpy()
    #     _, h, w = vis_prob.shape
    #     ax.imshow(vis_prob[0, :, :], cmap="bwr")
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.axis("off")
    #     # plt.show()
    #     plt.savefig(save_dir+"_"+str(i+1)+".png", dpi=600, bbox_inches='tight', pad_inches=0)
    #     plt.close()

    for i, mask in enumerate(masks):


        fig, ax = plt.subplots()
        mask = 1 - mask
        # mask  = torch.pow(1+mask, 2)
        vis_prob = mask[0].cpu().detach().numpy()

        c, h, w = vis_prob.shape
        if i >= 0:
            ph, pw = h // 20, w // 20
        else:
            ph, pw = 0, 0
        # ax.imshow(vis_prob[0, :, :], cmap="bwr")

        ax.imshow(vis_prob[0, ph:h-ph, pw:w-pw], cmap="bwr")
        # ax.imshow(vis_prob[0, :, :], cmap="cool")
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")
        # plt.show()
        # plt.close()
        plt.savefig(save_dir+"_"+str(i+1)+"r.png", dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close()
class DGIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = DGIR(is_train=False)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=180)

        return [optimizer],[scheduler]

def r2n16(x):
    return round(x/16) * 16

def test_Denoise(net, dataset, sigma=15):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    
    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()
    sim_1 = None
    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored, masks, sims = net(degrad_patch)

            # if sim_1 is not None:
            #     sim_1 = sim_1 + torch.mean(sims[0].flatten(0,1),dim=0)
            #     sim_1 /= 2
            # else:
            #     sim_1 = torch.mean(sims[0].flatten(-2,-1),dim=-1)

            if sim_1 is not None:
                sim_1 = torch.concat([sim_1, torch.mean(sims[0],dim=1)], dim=0)
            else:
                sim_1 = torch.mean(sims[0],dim=1)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            # save_mask(masks, output_path + clean_name[0])
            # save_image_tensor(degrad_patch, output_path + clean_name[0] + '_input.png')

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, output_path + clean_name[0] + '.png')
            # save_image_tensor(degrad_patch, output_path + clean_name[0] + '.png')

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))
    draw_dr(sim_1)
    return psnr.avg, ssim.avg

def draw_dr(sim):
    # plt.rc('font',family='Times New Roman')
    # sim = sim - torch.min(sim)
    sim_array = sim.cpu().detach().numpy()
    # x = range(len(sim_array))
    # plt.bar(x, sim_array)
    #
    # plt.title("deraining")
    # plt.xlim(-1,72)
    # plt.savefig('sim.png')
    # plt.close()
    # print(sim_array.shape)
    # print(sim_array.shape)

    # np.savetxt("a.csv", sim_array, delimiter=",")
    np.save("a.npy", sim_array)

def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()
    ind = 0

    sim_1 = None
    with torch.no_grad():
        # for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            # print(degrad_patch.shape)
            # sys.exit()
            _,_,oh,ow = degrad_patch.shape
            # s = 4
            s = 2
            if s == 1:
                pass
            else:
                # degrad_patch = nn.functional.interpolate(degrad_patch, size=(r2n16(oh//s),r2n16(ow//s)), mode='bicubic', align_corners=True)
                # clean_patch = nn.functional.interpolate(clean_patch, size=(r2n16(oh//s),r2n16(ow//s)), mode='bicubic', align_corners=True)

                degrad_patch = nn.functional.interpolate(degrad_patch, size=(r2n16(oh // s), r2n16(ow // s)))
                clean_patch = nn.functional.interpolate(clean_patch, size=(r2n16(oh // s), r2n16(ow // s)))


            restored, masks, sims = net(degrad_patch)
            # if sim_1 is not None:
            #     sim_1 = sim_1 + torch.mean(sims[0].flatten(0,1),dim=0)
            #     sim_1 /= 2
            # else:
            #     sim_1 = torch.mean(sims[0].flatten(-2,-1),dim=-1)

            # print(sims[0].shape)
            if sim_1 is not None:
                sim_1 = torch.concat([sim_1, torch.mean(sims[0],dim=1)], dim=0)
            else:
                sim_1 = torch.mean(sims[0],dim=1)


            # print(sims[0].shape)
            # restored, masks = net(restored)
            # restored, masks = net(restored)
            # restored, masks = net(restored)

            # RA = 5
            # restored = restored*RA + clean_patch
            # restored = restored / (RA+1)


            # save_image_tensor(degrad_patch, output_path + degraded_name[0] + '.png')

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored[:], clean_patch)

            # print(temp_psnr, temp_ssim)
            save_mask(masks, output_path + degraded_name[0])

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
            save_image_tensor(degrad_patch, output_path + degraded_name[0] + '_d.png')
            save_image_tensor(clean_patch, output_path + degraded_name[0] + '_c.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
    print(sim_1.shape)
    draw_dr(sim_1)
    return psnr.avg, ssim.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=6,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for deblur, 4 for enhance, 5 for all-in-one (three tasks), 6 for all-in-one (five tasks)')
    
    parser.add_argument('--gopro_path', type=str, default="data/test/deblur/", help='save path of test hazy images')
    parser.add_argument('--enhance_path', type=str, default="data/test/enhance/", help='save path of test hazy images')
    parser.add_argument('--denoise_path', type=str, default="data/test/denoise/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="data/test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="data/test/dehaze/", help='save path of test hazy images')

    parser.add_argument('--output_path', type=str, default="AdaIR_results/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="adair5d.ckpt", help='checkpoint save path')
    testopt = parser.parse_args()
    
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = "ckpt/" + testopt.ckpt_name

    denoise_splits = ["bsd68png/"]
    # denoise_splits = ["bsd68/"]

    # derain_splits = ["Rain100L/"]
    # derain_splits = ["LNB/"]
    # derain_splits = ["LB/"]
    # derain_splits = ["NB/"]
    # derain_splits = ["NightRain/"]
    # derain_splits = ["HR/"]
    # derain_splits = ["blur/"]
    # derain_splits = ["haze/"]
    # derain_splits = ["RealRain/"]
    # derain_splits = ["UHDLL/"]
    # derain_splits = ["HazeRD/"]
    derain_splits = ["RainDrop/"]

    deblur_splits = ["gopro/"]
    enhance_splits = ["lol/"]

    denoise_tests = []
    derain_tests = []

    base_path = testopt.denoise_path
    for i in denoise_splits:
        testopt.denoise_path = os.path.join(base_path,i)
        denoise_testset = DenoiseTestDataset(testopt)
        denoise_tests.append(denoise_testset)

    print("CKPT name : {}".format(ckpt_path))

    net  = DGIRModel.load_from_checkpoint(ckpt_path, strict=False).cuda()
    net.eval()

    if testopt.mode == 0:
        for testset,name in zip(denoise_tests,denoise_splits) :
            acc = ""
            # print('Start {} testing Sigma=15...'.format(name))
            # psnr, ssim = test_Denoise(net, testset, sigma=15)
            # acc += "  " + str(psnr)[:6] + "  " + str(ssim)[:7]

            # print('Start {} testing Sigma=25...'.format(name))
            # psnr, ssim = test_Denoise(net, testset, sigma=25)
            # acc += "  " + str(psnr)[:6] + "  " + str(ssim)[:7]

            print('Start {} testing Sigma=50...'.format(name))
            psnr, ssim = test_Denoise(net, testset, sigma=50)
            acc += "  " + str(psnr)[:6] + "  " + str(ssim)[:7]

            with open("performance.txt", "a+") as f:
                f.write(testopt.ckpt_name + "  " + acc + "\n")

            # print('Start {} testing Sigma=75...'.format(name))
            # test_Denoise(net, testset, sigma=75)
    elif testopt.mode == 1:
        print('Start testing rain streak removal...')
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
            psnr, ssim = test_Derain_Dehaze(net, derain_set, task="derain")

            with open("performance.txt", "a+") as f:
                f.write(testopt.ckpt_name + "  " + str(psnr)[:6] + "  " + str(ssim)[:7] + "\n")

    elif testopt.mode == 2:
        print('Start testing SOTS...')
        derain_base_path = testopt.derain_path
        name = derain_splits[0]
        testopt.derain_path = os.path.join(derain_base_path,name)
        derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
        psnr, ssim = test_Derain_Dehaze(net, derain_set, task="dehaze")

        with open("performance.txt", "a+") as f:
            f.write(testopt.ckpt_name + "  " + str(psnr)[:6] + "  "+ str(ssim)[:7] + "\n")

    elif testopt.mode == 3:
        print('Start testing GOPRO...')
        deblur_base_path = testopt.gopro_path
        name = deblur_splits[0]
        testopt.gopro_path = os.path.join(deblur_base_path,name)
        derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15, task='deblur')
        psnr, ssim = test_Derain_Dehaze(net, derain_set, task="deblur")
        with open("performance.txt", "a+") as f:
            f.write(testopt.ckpt_name + "  " + str(psnr)[:6] + "  "+ str(ssim)[:7] + "\n")

    elif testopt.mode == 4:
        print('Start testing LOL...')
        enhance_base_path = testopt.enhance_path
        name = enhance_splits[0]
        # testopt.enhance_path = os.path.join(enhance_base_path,name, task='enhance')
        testopt.enhance_path = os.path.join(enhance_base_path,name)
        derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15, task='enhance')
        psnr, ssim = test_Derain_Dehaze(net, derain_set, task="enhance")
        with open("performance.txt", "a+") as f:
            f.write(testopt.ckpt_name + "  " + str(psnr)[:6] + "  "+ str(ssim)[:7] + "\n")

    elif testopt.mode == 5:
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)

        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:

            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=55)
            test_Derain_Dehaze(net, derain_set, task="derain")

        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="dehaze")

    elif testopt.mode == 6:
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)

        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:

            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=55)
            test_Derain_Dehaze(net, derain_set, task="derain")

        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="dehaze")

        deblur_base_path = testopt.gopro_path
        for name in deblur_splits:
            print('Start testing GOPRO...')

            # print('Start testing {} rain streak removal...'.format(name))
            testopt.gopro_path = os.path.join(deblur_base_path,name)
            deblur_set = DerainDehazeDataset(testopt,addnoise=False,sigma=55, task='deblur')
            test_Derain_Dehaze(net, deblur_set, task="deblur")

        enhance_base_path = testopt.enhance_path
        for name in enhance_splits:

            print('Start testing LOL...')
            testopt.enhance_path = os.path.join(enhance_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=55, task='enhance')
            test_Derain_Dehaze(net, derain_set, task="enhance")
