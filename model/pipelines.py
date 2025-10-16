from typing import List, Optional, Tuple, Union

import torch
import inspect
try:
    from diffusers.utils import randn_tensor
except:
    from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DDIMScheduler

import argparse
    
class DDIMPipelineRangeSemantic(DiffusionPipeline):
    """
    Example pipeline for conditional generation with a semantic map using a DDIM-based scheduler.
    This version employs classifier-free guidance. When guidance_scale != 1.0, an unconditional branch
    (using a "null" semantic map, here implemented as a tensor of zeros) is used to steer the denoising process.
    
    This pipeline expects:
      - unet: A UNet2DModel with in_channels = out_channels + sem_channels.
      - scheduler: a DDIMScheduler.
      - sem_channels: number of channels in your semantic map (e.g. 1 for gray, or multiple for one-hot).
      - out_channels: number of channels your model outputs (e.g., 1 for gray, 3 for RGB).
    """

    def __init__(self, unet, scheduler, sem_channels=1):
        super().__init__()

        # Convert the passed-in scheduler to DDIM if needed
        if not isinstance(scheduler, DDIMScheduler):
            scheduler = DDIMScheduler.from_config(scheduler.config)

        self.register_modules(unet=unet, scheduler=scheduler)
        self.sem_channels = sem_channels

    @torch.no_grad()
    def __call__(
        self,
        sem: torch.Tensor,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        output_type: Optional[str] = "torch",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Generate images conditioned on a semantic map using classifier-free guidance.

        Args:
            sem (torch.Tensor): shape [B, sem_channels, H, W]. The semantic label map(s) to condition on.
            batch_size (int): how many samples to produce; should match sem.shape[0] if sem is given.
            generator (torch.Generator or List[torch.Generator]): for reproducible noise sampling.
            eta (float): DDIM parameter.
            num_inference_steps (int): number of timesteps for the denoising loop.
            guidance_scale (float): Scale for classifier-free guidance. A value of 1.0 corresponds to standard
                                    conditional sampling, while values > 1.0 strengthen the conditioning.
            output_type (str): "torch" or "pil".
            return_dict (bool): whether to return an ImagePipelineOutput or tuple.

        Returns:
            If `output_type="torch"`, returns a torch.Tensor of shape [B, out_channels, H, W].
            Otherwise returns PIL images or numpy arrays via the pipeline’s usual postprocessing.
        """
        # Check shapes
        if sem.shape[0] != batch_size:
            raise ValueError(f"Mismatch: sem has batch={sem.shape[0]}, but batch_size={batch_size}.")
        # We assume unet.config.out_channels is your image channels (e.g. 1 or 3)
        # and that unet.config.in_channels == out_channels + sem_channels.

        # 1) Sample Gaussian noise as starting point
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.out_channels,  # typically the final image channels
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.out_channels, *self.unet.config.sample_size)

        # Sample initial noise
        image = randn_tensor(
            image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype
        )

        # 2) Configure the DDIM scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # 3) Move semantic map to correct device & dtype
        sem = sem.to(self._execution_device, dtype=self.unet.dtype)

        # 4) Iterative denoising
        for t in self.scheduler.timesteps:
            if guidance_scale != 1.0:
                # For classifier-free guidance, we need both the unconditional and the conditional predictions.
                # We create an unconditional semantic map (here, a tensor of zeros) and duplicate the image.
                # The first half of the batch is unconditional, the second is conditional.
                image_in = image.repeat(2, 1, 1, 1)  # Shape: [2*B, out_channels, H, W]
                unconditional_sem = torch.zeros_like(sem)
                sem_in = torch.cat([unconditional_sem, sem], dim=0)  # Shape: [2*B, sem_channels, H, W]

                # Concatenate along the channel dimension: resulting shape [2*B, out_channels + sem_channels, H, W]
                model_input = torch.cat([image_in, sem_in], dim=1)

                # Get noise predictions for both unconditional and conditional inputs in one forward pass.
                model_output = self.unet(model_input, t).sample
                noise_pred_uncond, noise_pred_cond = model_output.chunk(2, dim=0)
                # Combine predictions using classifier-free guidance formula.
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # Standard conditional sampling.
                model_input = torch.cat([image, sem], dim=1)
                noise_pred = self.unet(model_input, t).sample

            # Perform the DDIM step.
            image = self.scheduler.step(
                noise_pred, t, image, eta=eta, generator=generator
            ).prev_sample

        # 5) Postprocessing
        if output_type == "torch":
            return image

        # If your model was trained with data in [-1,1], you might map it to [0,1] like this:
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)