from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers import DDIMScheduler
import gc
import os
from diffusers import StableDiffusion3Pipeline 
from diffusers.models import AutoencoderKL 
import torch
import torch.nn as nn
import gc
from safetensors.torch import load_file
from transformers import CLIPTokenizer, CLIPTextModel


class SDFeaturizer:
    def __init__(
        self, 
        model_path: str, 
        sd_version: str = "2-1", 
        null_prompt: str = "", 
        auth_token: Optional[str] = None,
        patch_size: int = 2,
        use_patches: bool = True
    ):
        """
        Initialize the SDFeaturizer for Stable Diffusion 3.
        :param model_path: Path to the .safetensors file containing the MM-DiT weights.
        :param null_prompt: Optional prompt to use as a null reference.
        """
        self.patch_size = patch_size
        self.use_patches = use_patches

        # Set the environment variable to reduce memory fragmentations 
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        with torch.no_grad():
            # Load model weights from safetensors file
            state_dict = load_file(model_path)

            # Clear unused memory
            torch.cuda.empty_cache()
            gc.collect()


            # Initialize MM-DiT (U-Net equivalent)
            self.mm_dit = self._initialize_mm_dit(state_dict).to("cuda")

            # Clear unused memory
            torch.cuda.empty_cache()
            gc.collect()

            self.sd_version = sd_version
            self.auth_token = auth_token

            if self.sd_version == "2-1":
                self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae").to("cuda")
                # Scheduler (for diffusion steps) (brauchen wir das?)
                self.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="scheduler")

                # Load CLIP tokenizer and text encoder (brauchen wir doch nicht?)
                self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
                self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
            else:
                # Load SD3-components from pretrained Pipeline
                sd3_pipeline = StableDiffusion3Pipeline.from_pretrained(
                    "stabilityai/stable-diffusion-3-medium-diffusers",
                    torch_dtype=torch.float32,
                    use_auth_token=self.auth_token,
                    low_cpu_mem_usage=False
                ).to("cuda")
                self.tokenizer= sd3_pipeline.tokenizer
                self.text_encoder = sd3_pipeline.text_encoder
                self.vae= sd3_pipeline.vae
                self.scheduler = sd3_pipeline.scheduler

            # Extract scaling and shift factors from VAE config
            self.scaling_factor = self.vae.config.scaling_factor
            self.shift_factor = self.vae.config.shift_factor if hasattr(self.vae.config, "shift_factor") else 0.0
            print(f"VAE scaling factor: {self.scaling_factor}, shift factor: {self.shift_factor}")

            # Clear unused memory
            torch.cuda.empty_cache()
            gc.collect()

            # Prepare null prompt embedding
            self.null_prompt_embeds = self.get_prompt_embedding(null_prompt)

        # Clear unused memory
        torch.cuda.empty_cache()
        gc.collect()

    def _initialize_mm_dit(self, state_dict):
        """
        Initialize the MM-DiT (transformer-based model) from the provided state_dict.
        """
        mm_dit = nn.Transformer(
            d_model=1024,
            nhead=16,
            num_encoder_layers=24,
            num_decoder_layers=24,
            dim_feedforward=2048,
            dropout=0.1,
        )
        mm_dit = mm_dit.to("cuda")
        # Clear unused memory
        torch.cuda.empty_cache()
        gc.collect()
        with torch.no_grad():
            mm_dit.load_state_dict(state_dict, strict=False)
        return mm_dit

    def get_prompt_embedding(self, prompt: str) -> torch.Tensor:
        """
        Generate text embeddings for a given prompt using CLIP's text encoder.
        :param prompt: Text prompt for embedding.
        :return: Tensor of shape (batch_size, seq_len, d_model).
        """
        with torch.no_grad():
            inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to("cuda")
            return self.text_encoder(**inputs).last_hidden_state

    def _align_sequence_lengths(self, noisy_latents: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        Align the sequence lengths of noisy latents and prompt embeddings by padding or truncation.
        """
        latent_len = noisy_latents.size(1)
        prompt_len = prompt_embeds.size(1)

        if latent_len > prompt_len:
            # Pad prompt embeddings
            padding = torch.zeros((prompt_embeds.size(0), latent_len - prompt_len, prompt_embeds.size(2)),
                                  device=prompt_embeds.device, dtype=prompt_embeds.dtype)
            prompt_embeds = torch.cat([prompt_embeds, padding], dim=1)
        elif prompt_len > latent_len:
            # Pad noisy latents
            padding = torch.zeros((noisy_latents.size(0), prompt_len - latent_len, noisy_latents.size(2)),
                                  device=noisy_latents.device, dtype=noisy_latents.dtype)
            noisy_latents = torch.cat([noisy_latents, padding], dim=1)

        return noisy_latents, prompt_embeds


    def vae_encode(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode an image tensor into the VAE's latent space.
        :param img_tensor: Tensor of shape [1, C, H, W], normalized to [-1, 1].
        :return: Encoded latent tensor of shape [1, latent_channels, latent_height, latent_width].
        """
        assert img_tensor.ndim == 4, "Input image tensor must have shape [1, 3, H, W]."
        img_tensor = img_tensor.to(dtype=torch.float32, device="cuda")  # Use full precision for encoding

        with torch.no_grad():
            if self.sd_version == "2-1":
                latents = self.vae.encode(img_tensor).latent_dist.sample()
                print(f"Latents before scaling: min={latents.min().item()}, max={latents.max().item()}")
                latents = latents * self.scaling_factor 
                print(f"Latents after scaling and shifting: min={latents.min().item()}, max={latents.max().item()}")
            else:
                latents = self.vae.encode(img_tensor).latent_dist.sample()
                print(f"Latents before scaling: min={latents.min().item()}, max={latents.max().item()}")
                latents = latents * self.scaling_factor + self.shift_factor
                print(f"Latents after scaling and shifting: min={latents.min().item()}, max={latents.max().item()}")
                
            # Optionally clear memory
            torch.cuda.empty_cache()
            gc.collect()
            return latents

    def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent tensor into an image tensor.
        :param latents: Tensor of shape [1, latent_channels, latent_height, latent_width].
        :return: Decoded image tensor of shape [1, C, H, W], normalized to [-1, 1].
        """
        assert latents.ndim == 4, "Latents must have shape [1, 16, H/8, W/8]."
        latents = latents.to(dtype=torch.float32, device="cuda")  # Use full precision for decoding

        print(f"Latents before decoding: min={latents.min().item()}, max={latents.max().item()}")
        with torch.no_grad():
            if self.sd_version == "2-1":
                latents = latents / self.scaling_factor
                decoded = self.vae.decode(latents)
                decoded_sample = decoded.sample  # Extract the sample tensor
                print(f"Decoded tensor: min={decoded_sample.min().item()}, max={decoded_sample.max().item()}")
            else:
                latents = (latents - self.shift_factor) / self.scaling_factor
                decoded = self.vae.decode(latents)
                decoded_sample = decoded.sample  # Extract the sample tensor
                print(f"Decoded tensor: min={decoded_sample.min().item()}, max={decoded_sample.max().item()}")


        # Optionally clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        return decoded_sample

    
    def get_noisy_latents(self, img_tensor: torch.Tensor, t: int) -> torch.Tensor:
        """
        Get noised latent tensor from the VAE for the specified timestep.
        :param img_tensor: Tensor of shape [1, C, H, W], normalized to [-1, 1].
        :param t: Timestep for noise addition.
        :return: Noised latent tensor of shape [1, latent_channels, latent_height, latent_width].
        """

        # Ensure the input tensor matches the precision of the model
        img_tensor = img_tensor.to(dtype=torch.float32, device="cuda")

        # Encode image into latent space
        with torch.no_grad():
            if self.sd_version == "2-1":
                latents = self.vae.encode(img_tensor).latent_dist.sample() * self.scaling_factor
            else:
                latents = self.vae.encode(img_tensor).latent_dist.sample() * self.scaling_factor + self.shift_factor

            # Add noise to latents for the specified timestep
            t_tensor = torch.tensor([t], device=latents.device).long()
            noise = torch.randn_like(latents)

            if self.sd_version == "2-1":
                # Add noise to latents for the specified timestep
                t_tensor = torch.tensor([t], device=latents.device).long()
                noisy_latents = self.scheduler.add_noise(latents, noise, t_tensor)
            else:
                # Add noise to latents for the specified timestep
                timestep = self.scheduler.timesteps[t]
                noisy_latents = self.scheduler.scale_noise(latents, timestep, noise)
        
        # Optionally clear memory
        torch.cuda.empty_cache()
        gc.collect()

        return noisy_latents
    
    def process_noisy_latents(self, noisy_latents, timesteps=50):
        """
        Process noisy latents using Stable Diffusion 3 pipeline to generate an image.
        :param noisy_latents: Tensor of shape [1, 4, 64, 64].
        :param timesteps: Number of diffusion steps to process.
        :return: Decoded image.
        """
        with torch.no_grad():
            # Ensure noisy latents are in the correct device and format
            noisy_latents = noisy_latents.to(dtype=torch.float16, device="cuda")
            
            # Use the pipeline to process noisy latents into an image
            decoded_image = self.pipeline.decode_latents(noisy_latents)
        
        return decoded_image  
    
    def _patch_latents(self, latents: torch.Tensor) -> torch.Tensor:

        B, C, H, W = latents.size()  # Input shape
        patch_size = self.patch_size

        # Ensure H and W are divisible by patch_size
        assert H % patch_size == 0 and W % patch_size == 0, (
            f"Height and Width must be divisible by patch_size={patch_size}."
        )

        # Unfold the image into patches
        patches = latents.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # Shape after unfold: [B, C, num_patches_h, num_patches_w, patch_size, patch_size]

        # Reshape patches into [B, num_patches, patch_size*patch_size*C]
        patches = patches.contiguous().view(B, C, -1, patch_size * patch_size)
        # Shape: [B, C, num_patches, patch_size*patch_size]

        patches = patches.permute(0, 2, 3, 1).reshape(B, -1, C * patch_size * patch_size)
        # Shape: [B, num_patches, C * patch_size * patch_size]

        return patches    


    def forward(self, img_tensor: torch.Tensor, prompt: str = "", t: int = 261, use_patches: Optional[bool] = None) -> torch.Tensor:
        """
        Forward pass to extract features from MM-DiT.
        :param img_tensor: Tensor of shape [1, C, H, W].
        :param prompt: Text prompt for conditioning.
        :param t: Timestep for noise addition.
        :param use_patches: Optional boolean to override the default use_patches setting.
        :return: Features extracted by MM-DiT.
        """
        # Determine whether to use patches for this forward pass
        apply_patches = self.use_patches if use_patches is None else use_patches

        img_tensor = img_tensor.to("cuda")
        
        # Encode image into latent space
        #with torch.no_grad():
            #latents = self.vae.encode(img_tensor).latent_dist.sample() * self.scaling_factor + self.shift_factor

        # Add noise to latents
        noisy_latents = self.get_noisy_latents(img_tensor, t)

        if apply_patches:
            # Patch the noisy latents
            patches = self._patch_latents(noisy_latents)
            # Shape of patches: [B, num_patches, C * patch_size * patch_size]

            # Project latents to match MM-DiT's d_model dimension if necessary
            if patches.size(-1) != self.mm_dit.d_model:
                projector = nn.Linear(patches.size(-1), self.mm_dit.d_model).to("cuda")
                patches = projector(patches)

            input_tensor = patches
        else:
            # Flatten the noisy_latents without patching
            B, C, H, W = noisy_latents.shape
            input_tensor = noisy_latents.view(B, C, H * W).permute(0, 2, 1)  # Shape: [B, H*W, C]

            # Project latents to match MM-DiT's d_model dimension if necessary
            if input_tensor.size(-1) != self.mm_dit.d_model:
                projector = nn.Linear(input_tensor.size(-1), self.mm_dit.d_model).to("cuda")
                input_tensor = projector(input_tensor)

        # Get prompt embeddings
        prompt_embeds = self.null_prompt_embeds if prompt == "" else self.get_prompt_embedding(prompt)
        if prompt_embeds.size(-1) != self.mm_dit.d_model:
            projector = nn.Linear(prompt_embeds.size(-1), self.mm_dit.d_model).to("cuda")
            prompt_embeds = projector(prompt_embeds)

        # Align sequence lengths (padding or truncation)
        input_tensor, prompt_embeds = self._align_sequence_lengths(input_tensor, prompt_embeds)

        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Prompt embeds shape: {prompt_embeds.shape}")

        # Process the inputs and prompt embeddings through MM-DiT
        ###########################################################
        # Extract features from MM-DiT
        ###########################################################
        with torch.cuda.amp.autocast(enabled=True):
            features = self.mm_dit(input_tensor, prompt_embeds)
        torch.cuda.empty_cache()
        gc.collect()
        return features


    def _project_to_d_model(self, tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        """
        Project input tensor to the target dimension if required.
        :param tensor: Input tensor of shape (batch_size, seq_len, input_dim).
        :param target_dim: Target dimension for projection.
        :return: Projected tensor of shape (batch_size, seq_len, target_dim).
        """
        input_dim = tensor.size(-1)
        if input_dim != target_dim:
            projection_layer = nn.Linear(input_dim, target_dim).to("cuda")
            tensor = projection_layer(tensor)
        return tensor