import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers.models import UNet2DConditionModel
from diffusers import DDIMScheduler
import gc
import os
from PIL import Image
from torchvision.transforms import PILToTensor

from diffusers import StableDiffusionPipeline
from transformers import PreTrainedModel, PreTrainedTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers import StableDiffusionPipeline
from transformers import PreTrainedModel, PreTrainedTokenizer
import gc

from safetensors.torch import load_file
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.models.autoencoder_kl import AutoencoderKL
from typing import Dict, Optional, Any, Tuple, List



class SDFeaturizer:
    def __init__(self, model_path: str, null_prompt: str = ""):
        """
        Initialize the SDFeaturizer for Stable Diffusion 3.
        :param model_path: Path to the .safetensors file containing the MM-DiT weights.
        :param null_prompt: Optional prompt to use as a null reference.
        """

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
            
            #Instead of loading the entire model onto the GPU at once, load parts of it incrementally
            #self.mm_dit = self._initialize_mm_dit(state_dict).half()  # Keep in CPU
            #self.mm_dit.to("cuda", non_blocking=True)  # Move to GPU in stages
            
            # Clear unused memory
            torch.cuda.empty_cache()
            gc.collect()
            self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae").to("cuda")
            
            
             # Clear unused memory
            torch.cuda.empty_cache()
            gc.collect()
            # Load CLIP tokenizer and text encoder (brauchen wir doch nicht?)
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
            
            # Scheduler (for diffusion steps) (brauchen wir das?)
            self.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="scheduler")
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

        # Ensure the input tensor matches the precision of the model
        img_tensor = img_tensor.to(dtype=torch.float32, device="cuda")  # Use full precision for encoding

        # Encode image into latent space
        with torch.no_grad():
            latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor

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

        # Ensure the input tensor matches the precision of the model
        latents = latents.to(dtype=torch.float32, device="cuda")  # Use full precision for decoding

        # Decode latents into image space
        with torch.no_grad():
            decoded = self.vae.decode(latents).sample  # Explicitly access the decoded sample if applicable

        # Optionally clear memory
        torch.cuda.empty_cache()
        gc.collect()

        return decoded

    def forward(self, img_tensor, prompt="", t=261):
        """
        Forward pass to extract features from MM-DiT.
        :param img_tensor: Tensor of shape [1, C, H, W].
        :param prompt: Text prompt for conditioning.
        :param t: Timestep for noise addition.
        :return: Features extracted by MM-DiT.
        """
        img_tensor = img_tensor.to("cuda")
         # Encode image into latent space
        with torch.no_grad():
            latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        #######################################################  
        ## TODO: decode for testing if vae encode/decode works
        ## Visualisierungen machen vom (noised?) latent und vom decoded image
        #######################################################  

        #######################################################
        ## TODO: noised latent von SD3 zurück rechnen lassen
        #######################################################
            
            
            # Add noise to latents for the specified timestep
            t_tensor = torch.tensor([t], device=latents.device).long()
            noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(latents, noise, t_tensor)
            
            # Reshape latents to transformer-friendly format
            b, c, h, w = noisy_latents.shape
            noisy_latents = noisy_latents.view(b, h * w, c)
        
        # Project latents to match MM-DiT's d_model dimension
        if noisy_latents.size(-1) != self.mm_dit.d_model:
            projector = nn.Linear(noisy_latents.size(-1), self.mm_dit.d_model).to("cuda")
            noisy_latents = projector(noisy_latents)
        
        # Get prompt embeddings
        prompt_embeds = self.null_prompt_embeds if prompt == "" else self.get_prompt_embedding(prompt)
        if prompt_embeds.size(-1) != self.mm_dit.d_model:
            projector = nn.Linear(prompt_embeds.size(-1), self.mm_dit.d_model).to("cuda")
            prompt_embeds = projector(prompt_embeds)
        
        # Align sequence lengths (padding or truncation)
        noisy_latents, prompt_embeds = self._align_sequence_lengths(noisy_latents, prompt_embeds)
        
        print(f"Noisy latents shape: {noisy_latents.shape}")
        print(f"Prompt embeds shape: {prompt_embeds.shape}")


        # Process the latents and prompt embeddings through MM-DiT
        ###########################################################
        ## TODO: Try to extract from different transformer layers (right now we extract them at the end)
        ###########################################################
        with torch.cuda.amp.autocast():
            features = self.mm_dit(noisy_latents, prompt_embeds)
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