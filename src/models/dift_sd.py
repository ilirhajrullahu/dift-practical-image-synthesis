from typing import Optional
import gc
import os
from diffusers import StableDiffusion3Pipeline
import torch
import gc
from safetensors.torch import load_file
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import hf_hub_download


class CustomTransformer(nn.Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        memory_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        encoder_layer_idx=None,
        decoder_layer_idx=None,
    ):
        """
        - encoder_layer_idx: Index of the encoder layer after which to extract features (0-based).
        - decoder_layer_idx: Index of the decoder layer after which to extract features (0-based).
        """
        # Pass through encoder
        memory = src
        encoder_outputs = []
        for idx, layer in enumerate(self.encoder.layers):
            memory = layer(
                memory, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
            )
            encoder_outputs.append(memory)
            if encoder_layer_idx is not None and idx == encoder_layer_idx:
                return memory

        # Pass through decoder
        output = tgt
        decoder_outputs = []
        for idx, layer in enumerate(self.decoder.layers):
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            decoder_outputs.append(output)
            if decoder_layer_idx is not None and idx == decoder_layer_idx:
                return output

        return output


class SDFeaturizer:
    def __init__(
        self,
        auth_token: Optional[str] = None,
        visualize_transformer_blocks: bool = False,
    ):
        """
        Initialize the SDFeaturizer for Stable Diffusion 3.
        :param auth_token: Optional Hugging Face API token for model download.
        """
        # Load the model from the specified path
        self.auth_token = auth_token
        self.visualise_transformer_blocks = visualize_transformer_blocks

        # Set the environment variable to reduce memory fragmentations
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        if visualize_transformer_blocks:
            file_path = hf_hub_download(
                repo_id="stabilityai/stable-diffusion-3-medium",
                filename="sd3_medium.safetensors",
                use_auth_token=auth_token,
            )
            state_dict = load_file(file_path)
            with torch.no_grad():
                self.custom_mm_dit = (
                    self._initialize_mm_dit(state_dict).to("cuda")
                )
            # Use .half() for memory efficiency
            self.vae = (
                AutoencoderKL.from_pretrained(
                    "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae"
                )
                .to("cuda")
                
            )
            self.scaling_factor = self.vae.config.scaling_factor
            self.shift_factor = (
                    self.vae.config.shift_factor
                    if hasattr(self.vae.config, "shift_factor")
                    else 0.0
                )

            # Clear unused memory
            torch.cuda.empty_cache()
            gc.collect()

            # Load CLIP tokenizer and text encode
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self.text_encoder = (
                CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
                .to("cuda")
                
            )

            # Scheduler (for diffusion steps)
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="scheduler"
            )

            # Prepare null prompt embedding
            self.null_prompt_embeds = self.get_prompt_embedding("")
            # Clear unused memory
            torch.cuda.empty_cache()
            gc.collect()
        else:
            with torch.no_grad():
                # Load SD3-components from pretrained Pipeline
                sd3_pipeline = StableDiffusion3Pipeline.from_pretrained(
                    "stabilityai/stable-diffusion-3-medium-diffusers",
                    torch_dtype=torch.float32,
                    use_auth_token=self.auth_token,
                    low_cpu_mem_usage=False,
                ).to("cuda")
                self.pipeline = sd3_pipeline
                self.tokenizer = sd3_pipeline.tokenizer_3
                self.text_encoder = sd3_pipeline.text_encoder_3
                self.vae = sd3_pipeline.vae
                self.scheduler = sd3_pipeline.scheduler
                self.transformer = sd3_pipeline.transformer
                del sd3_pipeline  # delete pipeline to free up memory

                # Extract scaling and shift factors from VAE config
                self.scaling_factor = self.vae.config.scaling_factor
                self.shift_factor = (
                    self.vae.config.shift_factor
                    if hasattr(self.vae.config, "shift_factor")
                    else 0.0
                )
                print(
                    f"VAE scaling factor: {self.scaling_factor}, shift factor: {self.shift_factor}"
                )

            # Clear unused memory
            torch.cuda.empty_cache()
            gc.collect()

    def get_prompt_embedding(self, prompt: str) -> torch.Tensor:
        """
        Generate text embeddings for a given prompt using CLIP's text encoder.
        :param prompt: Text prompt for embedding.
        :return: Tensor of shape (batch_size, seq_len, d_model).
        """
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
        :param img_tensor: Tensor of shape [1, C, H, W].
        :return: Encoded latent tensor of shape [1, latent_channels, latent_height, latent_width].
        """
        assert img_tensor.ndim == 4, "Input image tensor must have shape [1, 3, H, W]."
        img_tensor = img_tensor.to(
            dtype=torch.float32, device="cuda"
        )  # Use full precision for encoding

        with torch.no_grad():
            latents = self.vae.encode(img_tensor).latent_dist.sample()
            print(
                f"Latents before scaling: min={latents.min().item()}, max={latents.max().item()}"
            )
            latents = latents * self.scaling_factor + self.shift_factor
            print(
                f"Latents after scaling and shifting: min={latents.min().item()}, max={latents.max().item()}"
            )

        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()

        return latents

    def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent tensor into an image tensor.
        :param latents: Tensor of shape [1, latent_channels, latent_height, latent_width].
        :return: Decoded image tensor of shape [1, C, H, W].
        """
        assert latents.ndim == 4, "Latents must have shape [1, 16, H/8, W/8]."
        latents = latents.to(
            dtype=torch.float32, device="cuda"
        )  # Use full precision for decoding

        print(
            f"Latents before decoding: min={latents.min().item()}, max={latents.max().item()}"
        )
        with torch.no_grad():
            latents = (latents - self.shift_factor) / self.scaling_factor
            decoded = self.vae.decode(latents)
            decoded_sample = decoded.sample  # Extract the sample tensor
            print(
                f"Decoded tensor: min={decoded_sample.min().item()}, max={decoded_sample.max().item()}"
            )

        # clear memory
        torch.cuda.empty_cache()
        gc.collect()

        return decoded_sample

    def _initialize_mm_dit(self, state_dict):
        """
        Initialize the MM-DiT (transformer-based model) from the provided state_dict.
        """
        custom_mm_dit = CustomTransformer(
            d_model=1024,
            nhead=16,
            num_encoder_layers=24,
            num_decoder_layers=24,
            dim_feedforward=2048,
            dropout=0.1,
        )

        custom_mm_dit = custom_mm_dit.to("cuda")
        # Clear unused memory
        torch.cuda.empty_cache()
        gc.collect()
        custom_mm_dit.load_state_dict(state_dict, strict=False)
        return custom_mm_dit

    def get_noisy_latents(self, img_tensor: torch.Tensor, t: int) -> torch.Tensor:
        """
        Get noised latent tensor from the VAE for the specified timestep.
        :param img_tensor: Tensor of shape [1, C, H, W].
        :param t: Timestep for noise addition.
        :return: Noised latent tensor of shape [1, latent_channels, latent_height, latent_width].
        """

        # Ensure the input tensor matches the precision of the model
        img_tensor = img_tensor.to(dtype=torch.float32, device="cuda")

        # Encode image into latent space
        with torch.no_grad():
            latents = (
                self.vae.encode(img_tensor).latent_dist.sample() * self.scaling_factor
                + self.shift_factor
            )
            noisy_latents = None
            noise = torch.randn_like(latents)
            # Add noise to latents for the specified timestep
            timestep = self.scheduler.timesteps[t]
            noisy_latents = self.scheduler.scale_noise(latents, timestep, noise)
        
        # Optionally clear memory
        torch.cuda.empty_cache()
        gc.collect()

        return noisy_latents

    def process_noisy_latents(
        self, noisy_latents: torch.Tensor, timesteps: int
    ) -> torch.Tensor:
        """
        Process noisy latents using the transformer and scheduler to generate a denoised image.
        :param noisy_latents: Tensor of shape [1, 16, 64, 64].
        :param timesteps: Number of diffusion steps to process.
        :return: Decoded image tensor of shape [1, 3, 512, 512].
        """
        with torch.no_grad():
            # Ensure noisy latents are in the correct device and dtype
            noisy_latents = noisy_latents.to(dtype=self.vae.dtype, device="cuda")

            # Set timesteps for the scheduler
            self.scheduler.set_timesteps(timesteps)

            # Initialize latents for denoising
            current_latents = noisy_latents.clone()

            # Define projection layers for encoder_hidden_states
            projection_layer = torch.nn.Linear(
                self.transformer.config.caption_projection_dim,
                self.transformer.config.joint_attention_dim,
            ).to("cuda")

            # Loop through timesteps for denoising
            for t in self.scheduler.timesteps:
                # Prepare timestep tensor
                timestep_tensor = torch.tensor(
                    [int(t)], device=current_latents.device, dtype=torch.long
                )

                # Generate pooled projections and encoder hidden states
                batch_size = current_latents.shape[0]
                pooled_projections = torch.zeros(
                    (batch_size, self.transformer.config.pooled_projection_dim),
                    device="cuda",
                )
                encoder_hidden_states = torch.zeros(
                    (batch_size, 1, self.transformer.config.caption_projection_dim),
                    device="cuda",
                )

                # Apply projection layer to match required dimension
                encoder_hidden_states = projection_layer(
                    encoder_hidden_states.squeeze(1)
                ).unsqueeze(1)

                # Predict noise with the transformer
                noise_pred = self.transformer(
                    current_latents,
                    timestep=timestep_tensor,
                    pooled_projections=pooled_projections,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # Update latents using the scheduler
                current_latents = self.scheduler.step(
                    noise_pred, t, current_latents
                ).prev_sample

            # Decode the final latents back to image space
            decoded_image = self.vae_decode(current_latents)

        return decoded_image

    # ---------------------------------------------------------------------
    # New helper methods to keep forward(...) cleaner
    # ---------------------------------------------------------------------
    def _encode_and_scale(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode the image tensor into latents and scale/shift them.
        """
        latents = self.vae.encode(
            img_tensor.to(device="cuda", dtype=torch.float32)
        ).latent_dist.sample()

        latents = latents * self.scaling_factor + self.shift_factor
        return latents

    def _add_noise_to_latents(self, latents: torch.Tensor, t: int) -> torch.Tensor:
        """
        Add noise to the latents for the specified timestep t.
        """
        noise = torch.randn_like(latents)

        noisy_latents = None
        noise = torch.randn_like(latents)
        # Add noise to latents for the specified timestep
        timestep = self.scheduler.timesteps[t]
        noisy_latents = self.scheduler.scale_noise(latents, timestep, noise)

        return noisy_latents

    def _tokenize_and_encode_prompt(
        self, prompt: str, device: str = "cuda"
    ) -> torch.Tensor:
        """
        Tokenize the prompt (or use null prompt if empty), then encode with T5.
        """

        actual_prompt = prompt if prompt else ""
        inputs = self.tokenizer(
            [actual_prompt], return_tensors="pt", padding=True, truncation=True
        ).to(device)
        text_embeds = self.text_encoder(**inputs).last_hidden_state
        return text_embeds

    def _prepare_pooled(
        self, batch_size: int, dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        """
        Prepare a zero-filled pooled projection tensor (commonly used in SD3).
        """
        return torch.zeros(
            (batch_size, self.transformer.config.pooled_projection_dim),
            device=device,
            dtype=dtype,
        )

    def _run_sd3_transformer(
        self,
        noisy_latents: torch.Tensor,
        text_embeds: torch.Tensor,
        pooled_projections: torch.Tensor,
        t_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pass latents + text embeddings through the SD3 transformer, returning the final sample.
        """
        transformer_out = self.transformer(
            hidden_states=noisy_latents,
            encoder_hidden_states=text_embeds,
            pooled_projections=pooled_projections,
            timestep=t_tensor,
            return_dict=True,
        )
        return transformer_out.sample

    def forward_with_specific_layers(
        self,
        img_tensor,
        prompt="",
        t=261,
        encoder_layer_idx=None,
        decoder_layer_idx=None,
    ):
        """
        Forward pass to extract features from MM-DiT.
        :param img_tensor: Tensor of shape [1, C, H, W].
        :param prompt: Text prompt for conditioning.
        :param t: Timestep for noise addition.
        :return: Features extracted by MM-DiT.
        """
        # Ensure the input tensor matches the precision of the model
        img_tensor = img_tensor.to("cuda")

        # Encode image into latent space
        with torch.no_grad():
            latents = (
                self.vae.encode(img_tensor).latent_dist.sample()
                * self.vae.config.scaling_factor
            )

        # Add noise to latents for the specified timestep
        noise = torch.randn_like(latents)
        t_tensor = torch.tensor([t], device=latents.device).long()
        # Add noise to latents for the specified timestep
        noisy_latents = self.scheduler.scale_noise(latents, noise, t_tensor)

        # Reshape latents to transformer-friendly format
        b, c, h, w = noisy_latents.shape
        noisy_latents = noisy_latents.view(b, h * w, c)

        # Project latents to match MM-DiT's d_model dimension
        if noisy_latents.size(-1) != self.custom_mm_dit.d_model:
            projector = (
                nn.Linear(noisy_latents.size(-1), self.custom_mm_dit.d_model)
                .to("cuda")
                
            )
            noisy_latents = projector(noisy_latents)

        # Get prompt embeddings
        prompt_embeds = (
            self.null_prompt_embeds
            if prompt == ""
            else self.get_prompt_embedding(prompt)
        )

        if prompt_embeds.size(-1) != self.custom_mm_dit.d_model:
            projector = (
                nn.Linear(prompt_embeds.size(-1), self.custom_mm_dit.d_model)
                .to("cuda")
                
            )
            prompt_embeds = projector(prompt_embeds)

        # Align sequence lengths (padding or truncation)
        noisy_latents, prompt_embeds = self._align_sequence_lengths(
            noisy_latents, prompt_embeds
        )

        print(f"Noisy latents shape: {noisy_latents.shape}")
        print(f"Prompt embeds shape: {prompt_embeds.shape}")

        with torch.cuda.amp.autocast():
            if encoder_layer_idx is not None:
                features = self.custom_mm_dit(
                    noisy_latents, prompt_embeds, encoder_layer_idx=encoder_layer_idx
                )
            elif decoder_layer_idx is not None:
                features = self.custom_mm_dit(
                    noisy_latents, prompt_embeds, decoder_layer_idx=decoder_layer_idx
                )
            features = self.custom_mm_dit(noisy_latents, prompt_embeds)

        # Clear unused memory
        torch.cuda.empty_cache()
        gc.collect()

        return features

    def forward(
        self, img_tensor: torch.Tensor, prompt: str = "", t: int = 261
    ) -> torch.Tensor:
        """
        Forward pass that:
        1) Encodes the input image via the VAE,
        2) Adds noise at timestep t,
        3) Tokenizes & encodes the prompt via T5,
        4) Passes everything through the SD3 (MM-DiT) transformer,
        5) Returns the transformer's output latents.

        Args:
            img_tensor (torch.Tensor): Image of shape [1, 3, H, W].
            prompt (str): Prompt string to condition the transformer.
            t (int): Timestep index for adding noise to the latents.

        Returns:
            torch.Tensor: Transformer's output tensor (e.g. shape [1, in_channels, H//8, W//8]).
        """
        with torch.no_grad():
            # Step A: Encode & scale the image
            latents = self._encode_and_scale(img_tensor)

            # Step B: Add noise to latents at the specified timestep
            noisy_latents = self._add_noise_to_latents(latents, t)
            t_tensor = torch.tensor([t], device=noisy_latents.device, dtype=torch.long)

            # Step C: Tokenize & encode prompt
            text_embeds = self._tokenize_and_encode_prompt(prompt, device="cuda")

            # Step D: Prepare pooled projections
            batch_size = noisy_latents.shape[0]
            pooled_projections = self._prepare_pooled(
                batch_size, noisy_latents.dtype, noisy_latents.device
            )

            # Step E: Run the SD3 transformer
            transformer_output = self._run_sd3_transformer(
                noisy_latents, text_embeds, pooled_projections, t_tensor
            )

        # Step F: Return the transformer's output
        return transformer_output
