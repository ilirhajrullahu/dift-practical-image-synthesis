import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
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
from diffusers.models.autoencoder_kl import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel

class SDFeaturizer:
    def __init__(self, model_path, null_prompt=""):
        """
        Initialize the SDFeaturizer for Stable Diffusion 3 with memory-efficient settings.
        """
        # Load model weights from safetensors file
        state_dict = load_file(model_path)

        # Initialize MM-DiT (U-Net equivalent)
        #self.mm_dit = self._initialize_mm_dit(state_dict)

        #Use .half() for memory efficiency (mixed precision)
        self.mm_dit = self._initialize_mm_dit(state_dict).to("cuda").half()

        # Load VAE for latent encoding
        # Replace the VAE loading in the initializer
        #self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae").to("cuda")

        #Use .half() for memory efficiency
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae").to("cuda").half()

        # Load CLIP tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")

        # Scheduler (for diffusion steps)
        self.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="scheduler")

        # Prepare null prompt embedding
        self.null_prompt_embeds = self.get_prompt_embedding(null_prompt)

    def _initialize_mm_dit(self, state_dict):
        """
        Initialize the MM-DiT (transformer-based U-Net) from the provided state_dict.
        """
        # Example configuration for MM-DiT (customize as per the architecture)
        mm_dit = torch.nn.Transformer(
            d_model=4096,  # Adjust dimensions
            nhead=16,      # Number of attention heads
            num_encoder_layers=24,
            num_decoder_layers=24,
            dim_feedforward=8192,
        ).half().to("cuda")  # Use half precision to reduce memory usage

        # Load the state_dict into the MM-DiT model
        mm_dit.load_state_dict(state_dict, strict=False)
        return mm_dit
    
    def get_prompt_embedding(self, prompt):
        """
        Encode the given text prompt using CLIP.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True).to("cuda")
        return self.text_encoder(**inputs).last_hidden_state

    def _compute_prompt_embeds(self, prompt):
        """
        Compute embeddings for a given text prompt.
        """
        tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="text_encoder").to("cuda")
        tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad(), torch.cuda.amp.autocast():
            embeds = text_encoder(**tokens.to("cuda")).last_hidden_state
        return embeds

    @torch.no_grad()
    def forward(self, img_tensor, prompt="", t=500):
        """
        Extract features from an image using MM-DiT.
        """
        # Encode the image to latent space
        with torch.cuda.amp.autocast():
            latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor

        # Add noise to the latents
        noise = torch.randn_like(latents).to("cuda")
        noisy_latents = latents + noise * t

        # Use null prompt embeddings if no prompt is provided
        #prompt_embeds = self.null_prompt_embeds if prompt == self.null_prompt else self._compute_prompt_embeds(prompt)
        prompt_embeds = self._compute_prompt_embeds(prompt)


        # Process the noisy latents through MM-DiT
        with torch.cuda.amp.autocast():
            features = self.mm_dit(noisy_latents, prompt_embeds)

        # Clear unused memory
        torch.cuda.empty_cache()
        gc.collect()

        return features
