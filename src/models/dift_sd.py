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

        # Set the environment variable to reduce memory fragmentations 
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    
        # Load model weights from safetensors file
        state_dict = load_file(model_path)

        # Clear unused memory
        torch.cuda.empty_cache()
        gc.collect()

        # Initialize MM-DiT (U-Net equivalent)
        #self.mm_dit = self._initialize_mm_dit(state_dict)

        #Use .half() for memory efficiency (mixed precision)
        with torch.no_grad():
            self.mm_dit = self._initialize_mm_dit(state_dict).to("cuda").half()

            #if Vram is not enough, use the following code to load the model in stages
            #self.mm_dit = self._initialize_mm_dit(state_dict).to("cuda").half()

        #nstead of loading the entire model onto the GPU at once, load parts of it incrementally
        #self.mm_dit = self._initialize_mm_dit(state_dict).half()  # Keep in CPU
        #self.mm_dit.to("cuda", non_blocking=True)  # Move to GPU in stages


         # Clear unused memory
        torch.cuda.empty_cache()
        gc.collect()

        # Load VAE for latent encoding
        # Replace the VAE loading in the initializer
        #self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae").to("cuda")

        #Use .half() for memory efficiency
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae").to("cuda").half()

         # Clear unused memory
        torch.cuda.empty_cache()
        gc.collect()

        # Load CLIP tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")

        # Scheduler (for diffusion steps)
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
        with torch.no_grad():
            mm_dit = nn.Transformer(
            d_model=4096,  # Adjust to match latent and embedding dimensions
            nhead=16,
            num_encoder_layers=24,
            num_decoder_layers=24,
            dim_feedforward=8192,
            )
    
        # Enable gradient checkpointing
        #mm_dit.gradient_checkpointing_enable()

        # Move model to CUDA and convert to half-precision
        #mm_dit = mm_dit.to("cuda").half()
        mm_dit = mm_dit.half().to("cuda")

        #self.mm_dit.load_state_dict(state_dict, strict=False)
        #self.mm_dit = self.mm_dit.to("cuda", non_blocking=True).half()

        # Clear unused memory
        torch.cuda.empty_cache()
        gc.collect()

        # Load the state_dict into MM-DiT
        mm_dit.load_state_dict(state_dict, strict=False)
        return mm_dit

    def get_prompt_embedding(self, prompt):
        """
        Encode the given text prompt using CLIP.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True).to("cuda")
        return self.text_encoder(**inputs).last_hidden_state

    def _project_to_d_model(self, tensor, target_dim):
        """
        Project tensor to the target dimension using a linear layer.
        """
        input_dim = tensor.size(-1)
        if input_dim != target_dim:
            projection_layer = nn.Linear(input_dim, target_dim).to("cuda").half()
            tensor = tensor.to(dtype=torch.float16)  # Convert to half-precision
            tensor = projection_layer(tensor)
        return tensor

    @torch.no_grad()
    def forward(self, img_tensor, prompt="", t=500):
        """
        Extract features from an image using MM-DiT.
        """

        #img_tensor = img_tensor.unsqueeze(0)  # Ensure batch size is 1

        # Encode the image to latent space
        with torch.cuda.amp.autocast():
            latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor

        # Add noise to the latents
        noise = torch.randn_like(latents).to("cuda")
        noisy_latents = latents + noise * t

        # Flatten and project noisy latents to match d_model
        noisy_latents = noisy_latents.flatten(2).transpose(1, 2)  # Shape: (batch_size, seq_len, latent_dim)
        noisy_latents = self._project_to_d_model(noisy_latents, self.mm_dit.d_model)

        # Use null prompt embeddings if no prompt is provided
        prompt_embeds = self.null_prompt_embeds if prompt == "" else self.get_prompt_embedding(prompt)
        prompt_embeds = self._project_to_d_model(prompt_embeds, self.mm_dit.d_model)

        # Process the noisy latents through MM-DiT
        with torch.cuda.amp.autocast():
            features = self.mm_dit(noisy_latents, prompt_embeds)

        # Clear unused memory
        torch.cuda.empty_cache()
        gc.collect()

        return features
