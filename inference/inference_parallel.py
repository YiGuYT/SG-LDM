import os
import argparse
import shutil
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import torch.nn as nn
import safetensors
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.models import AutoencoderKL
from data.sk import SemanticKITTILoader
from model.pipelines import DDIMPipelineRangeSemantic
from utils.utils import replace_attn, replace_conv, replace_down
from accelerate import PartialState


def parse_args():
    parser = argparse.ArgumentParser(description="SemanticKITTI Inference with 3D label generation.")
    parser.add_argument("--output_dir", type=str, default="semanticKITTI")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=1,
                        help="Scale for classifier-free guidance. A value of 1.0 corresponds to standard")
    
    args = parser.parse_args()

    # Typically your model artifacts are in these locations:
    args.unet_config = os.path.join(args.output_dir, 'unet', 'config.json')
    args.unet_checkpoint = os.path.join(args.output_dir, 'unet', 'diffusion_pytorch_model.safetensors')
    args.channel_map = os.path.join(args.output_dir, 'channel_mapper.bin')
    args.scheduler_config = os.path.join(args.output_dir, 'scheduler', 'scheduler_config.json')
    args.out = os.path.join(args.output_dir, 'generated_DDIM',str(args.guidance_scale))
    os.makedirs(args.out, exist_ok=True)

    return args


def main():
    # ------------------------------------------------
    # 1) Parse arguments, initialize accelerator
    # ------------------------------------------------
    args = parse_args()
    distributed_state = PartialState()
    device = distributed_state.device

    # ------------------------------------------------
    # 2) Prepare DataLoader
    # ------------------------------------------------
    loader = SemanticKITTILoader(
        os.environ.get('SemanticKITTI_DATASET'),
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers
    )
    test_dataloader = loader.test_dataloader()
    to_range = loader.train_dataset.to_range_image  # for .to_pc_torch()

    # ------------------------------------------------
    # 3) Load model (UNet)
    # ------------------------------------------------
    unet_config = UNet2DModel.load_config(args.unet_config)
    model = UNet2DModel.from_config(unet_config)

    channel_mapper = nn.Conv2d(
    in_channels=20,  # from your dataset one-hot
    out_channels=8,  # compress to 8 channels
    kernel_size=1,
    bias=False
    )
    channel_mapper.load_state_dict(torch.load(args.channel_map))
    channel_mapper.eval().to(device)

    # Load UNet weights
    safetensors.torch.load_model(model, args.unet_checkpoint)
    if distributed_state.is_main_process:
        print("Loaded UNet with params: ",
              sum(p.numel() for p in model.parameters())/1e6, "M")

    # ------------------------------------------------
    # 4) Scheduler
    # ------------------------------------------------
    sched_config = DDPMScheduler.load_config(args.scheduler_config)
    scheduler = DDPMScheduler.from_config(sched_config)

    # ------------------------------------------------
    # 5) Construct pipeline
    # ------------------------------------------------
    Pipeline = DDIMPipelineRangeSemantic
    pipe = Pipeline(
        unet=model,
        scheduler=scheduler,
    )
    pipe = pipe.to(device)
    if not distributed_state.is_main_process:
        pipe.set_progress_bar_config(disable=True)

    # ------------------------------------------------
    # 6) Inference loop over test set
    # ------------------------------------------------
    pipe.unet.eval()

    # Optionally fix the random seed across processes
    generator = None  # or torch.Generator(device=device).manual_seed(42)

    for batch in tqdm(test_dataloader, disable=not distributed_state.is_main_process):
        # shape: (B, num_classes, W, H)
        onehot_sem_map = batch["onehot_sem"].to(device)
        origin_sem_map = batch["origin_sem"].to(device)
        onehot_sem_map = channel_mapper(onehot_sem_map)
        out = pipe(
            generator=generator,
            sem=onehot_sem_map,
            batch_size=onehot_sem_map.shape[0],
            num_inference_steps=50,
            guidance_scale = args.guidance_scale,
            output_type="torch",
        )
        range_imgs = out
        label_imgs = origin_sem_map.squeeze(1)

        # ============================
        # (D) Convert range image -> 3D points
        # ============================
        pc_all = to_range.to_pc_torch(range_imgs)  # (B, N, 3 or 4)

        B, W, H = label_imgs.shape
        label_imgs_flat = label_imgs.reshape(B, W*H)  # (B, N)

        # ------------------------------------------------
        # 7) Save in "sequences/XX/velodyne/XXXXXX.bin" + label
        # ------------------------------------------------
        B = pc_all.shape[0]
        for j in range(B):
            # Original file path => e.g. /.../sequences/08/velodyne/000123.bin
            pts_path_in = batch["pts_path"][j]
            seq_id = os.path.basename(os.path.dirname(os.path.dirname(pts_path_in)))  # "08"
            file_id = os.path.basename(pts_path_in)                                  # "000123.bin"

            # Output .bin path
            out_bin_path = os.path.join(args.out, "sequences", seq_id, "velodyne", file_id)
            os.makedirs(os.path.dirname(out_bin_path), exist_ok=True)

            # Convert to numpy
            pc_np = pc_all[j].detach().cpu().numpy()  # (N, 3 or 4)
            depth = np.linalg.norm(pc_np[:, :3], axis=1)
            mask = (depth > 3.0) & (depth < 90.0)  # e.g. filter out far points
            pc_np = pc_np[mask]

            # Save the generated point cloud
            pc_np.astype(np.float32).tofile(out_bin_path)

            lbl_np = label_imgs_flat[j].detach().cpu().numpy()  # shape (N,)
            lbl_np = lbl_np[mask]  # same mask as geometry
            # Cast to uint32 for .label file
            lbl_np = lbl_np.astype(np.uint32)

            # Construct .label path
            label_filename = file_id.replace('.bin', '.label')  # e.g. "000123.label"
            out_label_path = os.path.join(args.out, "sequences", seq_id, "labels", label_filename)
            os.makedirs(os.path.dirname(out_label_path), exist_ok=True)
            lbl_np.tofile(out_label_path)

    if distributed_state.is_main_process:
        print("Inference complete!")
        print(f"Generated data under: {args.out}")


if __name__ == "__main__":
    main()
