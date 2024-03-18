## Ship in sight

[Paper](https://arxiv.org/abs/) | [Project Page]() | [Video]() | [WebUI]() |


[Luigi Sigillo](https://luigisigillo.github.io/), [Riccardo Fosco Gramaccioni](), [Alessandro Nicolosi](), [Danilo Comminiello]()

ISPAMMM-Lab, Sapienza University of Rome 

<img src="assets/network.png" width="800px"/>

### Update
- **2024.03.18**: Repo is released.


### Demo

[<img src="assets/imgsli_1.jpg" height="223px"/>](https://imgsli.com/MTc2MTI2) [<img src="assets/imgsli_2.jpg" height="223px"/>](https://imgsli.com/MTc2MTE2) [<img src="assets/imgsli_3.jpg" height="223px"/>](https://imgsli.com/MTc2MTIw)
[<img src="assets/imgsli_8.jpg" height="223px"/>](https://imgsli.com/MTc2MjUy) [<img src="assets/imgsli_4.jpg" height="223px"/>](https://imgsli.com/MTc2MTMy) [<img src="assets/imgsli_5.jpg" height="223px"/>](https://imgsli.com/MTc2MTMz)
[<img src="assets/imgsli_9.jpg" height="214px"/>](https://imgsli.com/MTc2MjQ5) [<img src="assets/imgsli_6.jpg" height="214px"/>](https://imgsli.com/MTc2MTM0) [<img src="assets/imgsli_7.jpg" height="214px"/>](https://imgsli.com/MTc2MTM2) [<img src="assets/imgsli_10.jpg" height="214px"/>](https://imgsli.com/MTc2MjU0)

For more evaluation, please refer to our [paper]() for details.

### Dependencies and Installation
- Pytorch == 1.12.1
- CUDA == 11.7
- pytorch-lightning==1.4.2
- xformers == 0.0.16 (Optional)
- Other required packages in `environment.yaml`
```
# git clone this repository
git clone https://github.com/luigisigillo/ShipinSight.git
cd ShipinSight

# Create a conda environment and activate it
conda env create --file environment.yaml
conda activate ShipinSight

# Install xformers
conda install xformers -c xformers/label/dev

# Install taming & clip
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e .
```

### Running Examples

#### Train
Download the pretrained Stable Diffusion models from [[HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)]

```
python main.py --train --base configs/shipinsight/v2-finetune_text_T_512.yaml --gpus GPU_ID, --name NAME --scale_lr False
```


#### Resume

```
python main.py --train --base configs/shipinsight/v2-finetune_text_T_512.yaml --gpus GPU_ID, --resume RESUME_PATH --scale_lr False
```

#### Test directly

Download the Diffusion and autoencoder pretrained models from [[HuggingFace]() | [Google Drive]() | [OneDrive]() | [OpenXLab]()].
We use the same color correction scheme introduced in paper by default.
You may change ```--colorfix_type wavelet``` for better color correction.
You may also disable color correction by ```--colorfix_type nofix```

- Test on 128 512: You need at least 10G GPU memory to run this script (batchsize 2 by default)
```
python scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/shipinsight/v2-finetune_text_T_512.yaml --ckpt CKPT_PATH --vqgan_ckpt VQGANCKPT_PATH --init-img INPUT_PATH --outdir OUT_DIR --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain
```
- Test on arbitrary size w/o chop for autoencoder (for results beyond 512): The memory cost depends on your image size, but is usually above 10G.
```
python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas.py --config configs/shipinsight/v2-finetune_text_T_512.yaml --ckpt CKPT_PATH --vqgan_ckpt VQGANCKPT_PATH --init-img INPUT_PATH --outdir OUT_DIR --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain
```

- Test on arbitrary size w/ chop for autoencoder: Current default setting needs at least 18G to run, you may reduce the autoencoder tile size by setting ```--vqgantile_size``` and ```--vqgantile_stride```.
Note the min tile size is 512 and the stride should be smaller than the tile size. A smaller size may introduce more border artifacts.
```
python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py --config configs/shipinsight/v2-finetune_text_T_512.yaml --ckpt CKPT_PATH --vqgan_ckpt VQGANCKPT_PATH --init-img INPUT_PATH --outdir OUT_DIR --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain
```

- For test on 768 model, you need to set ```--config configs/shipinsight/v2-finetune_text_T_768v.yaml```, ```--input_size 768``` and ```--ckpt```. You can also adjust ```--tile_overlap```, ```--vqgantile_size``` and ```--vqgantile_stride``` accordingly. We did not finetune CFW. -->


### Citation
If our work is useful for your research, please consider citing:

```
@INPROCEEDINGS{
    Sigi2406:Ship,
    AUTHOR="Luigi Sigillo and Riccardo Fosco Gramaccioni and Alessandro Nicolosi and
    Danilo Comminiello",
    TITLE="Ship in Sight: Diffusion Models for {Ship-Image} Super Resolution",
    BOOKTITLE="2024 International Joint Conference on Neural Networks (IJCNN) (IJCNN 2024)",
    ADDRESS="Yokohama, Japan",
    PAGES="7.98",
    DAYS=28,
    MONTH=jun,
    YEAR=2024,
}
```


### Acknowledgement

This project is based on [stablediffusion](https://github.com/Stability-AI/stablediffusion), [latent-diffusion](https://github.com/CompVis/latent-diffusion), [SPADE](https://github.com/NVlabs/SPADE), [mixture-of-diffusers](https://github.com/albarji/mixture-of-diffusers) and [StableSR](https://github.com/iceclear/stablesr). Thanks for their awesome work.