# checkpoints/

Pretrained backbones and warm-start checkpoints live here. Not tracked in
git. External models come from Hugging Face:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login   # needed for gated repos (e.g. the DROID-3D calibration dataset)

# --exclude skips the unused ~19 GB original-format svd*.safetensors files
hf download stabilityai/stable-video-diffusion-img2vid --exclude "svd*" \
    --local-dir checkpoints/stable-video-diffusion-img2vid
hf download openai/clip-vit-base-patch32 \
    --local-dir checkpoints/clip-vit-base-patch32
hf download facebook/VGGT-1B model.pt --local-dir checkpoints/vggt
```

Download links for depthworld-trained checkpoints are listed in the
top-level README.

Expected layout (all paths overridable via env vars / CLI flags):

```
checkpoints/
  stable-video-diffusion-img2vid/   # HF snapshot of stabilityai/stable-video-diffusion-img2vid
                                    #   (diffusers layout: unet/, vae/, image_encoder/, ...)
                                    #   env: CTRLWORLD_SVD_MODEL_PATH, flag: --svd_model_path
  clip-vit-base-patch32/            # HF snapshot of openai/clip-vit-base-patch32
                                    #   env: CTRLWORLD_CLIP_MODEL_PATH, flag: --clip_model_path
  vggt/model.pt                     # facebook/VGGT-1B — DPT point-head warm-start
                                    #   env: CTRLWORLD_VGGT_CKPT, flag: --pointmap_vggt_ckpt
  horiz_40k_merged.pt               # (example, not released) consolidated horiz checkpoint used as the
                                    #   warm-start for pointmap training
                                    #   env: CTRLWORLD_WARM_START_UNET_CKPT
```
