import argparse
import metrics as Metrics
from PIL import Image
import numpy as np
import glob
import torch
from tqdm import tqdm
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p_gt', '--path_gt', type=str,
                        default='/mnt/media/luigi/SR3_results')
    parser.add_argument('-p', '--path', type=str,
                        default='/mnt/media/luigi/low_res_eval') 
    args = parser.parse_args()
    #SR3 has different naming convention and file structure.
    # infact in the folder it mantains the HR (ground truth) images
    if "SR3" in args.path_gt:
        real_names = list(glob.glob('{}/*_hr.png'.format(args.path_gt)))
    else:
        real_names = list(glob.glob('{}/*.jpg'.format(args.path_gt)))
    # print(real_names, args.path_gt)
    if "SR3" in args.path:
        fake_names = list(glob.glob('{}/*_sr.png'.format(args.path)))
    else:
        fake_names = list(glob.glob('{}/*.png'.format(args.path)))

    real_names.sort()
    fake_names.sort()

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_fid = 0.0
    fid_img_list_real, fid_img_list_fake = [],[]
    idx = 0
    print(len(real_names), len(fake_names))   
    for rname, fname in tqdm(zip(real_names, fake_names), total=len(real_names)):
        idx += 1
            
        ridx = rname.rsplit("_hr")[0]
        if "SR3" in args.path:
            fidx = fname.rsplit("_sr")[0]
        elif "SWIN_IR" in args.path:
            fidx = fname.rsplit("_SwinIR")[0]
        else:
            fidx = fname.rsplit("_inf")[0]
        if not "SR3" in args.path:
            ridx = ridx.rsplit("/")[-1]
            fidx = fidx.rsplit("/")[-1]
        # print(ridx, fidx)
        assert ridx == fidx, 'Image ridx:{ridx}!=fidx:{fidx}'.format(
            ridx, fidx)

        hr_img = np.array(Image.open(rname))
        sr_img = np.array(Image.open(fname))
        psnr = Metrics.calculate_psnr(sr_img, hr_img)
        ssim = Metrics.calculate_ssim(sr_img, hr_img)
        fid_img_list_real.append(torch.from_numpy(hr_img).permute(2,0,1).unsqueeze(0))
        fid_img_list_fake.append(torch.from_numpy(sr_img).permute(2,0,1).unsqueeze(0))
        avg_psnr += psnr
        avg_ssim += ssim
        # if idx % 10 == 0:
        # # fid = Metrics.calculate_FID(torch.cat(fid_img_list_real,dim=0), torch.cat(fid_img_list_fake,dim=0))
        # # fid_img_list_real, fid_img_list_fake = [],[]                        
        # # avg_fid += fid
        #     print('Image:{}, PSNR:{:.4f}, SSIM:{:.4f}'.format(idx, psnr, ssim))


    #last FID
    fid = Metrics.calculate_FID(torch.cat(fid_img_list_real,dim=0), torch.cat(fid_img_list_fake,dim=0))
    avg_fid += fid

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    # avg_fid = avg_fid / idx

    # log
    print('# Validation # PSNR: {}'.format(avg_psnr))
    print('# Validation # SSIM: {}'.format(avg_ssim))
    print('# Validation # FID: {}'.format(avg_fid))

