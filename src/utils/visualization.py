import gc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.cm as cm
import matplotlib.colors as mcolors

class Demo:
    """
    Class for visualizing the feature tensor and the images.
    Uses the feature tensor to compute the cosine similarity between the source feature vector and the target feature vectors.
    Takes as input the feature tensor, the images, and the image size.
    """
    def __init__(self, imgs, ft, img_size):
        self.ft = ft # N+1, C, H, W
        self.imgs = imgs
        self.num_imgs = len(imgs)
        self.img_size = img_size

    def plot_img_pairs(self, fig_size=3, alpha=0.45, scatter_size=70):
        """
        Plots the images and the cosine similarity heatmaps.
        fig_size: size of the figure
        alpha: transparency of the heatmap
        scatter_size: size of the scatter plot
        """

        fig, axes = plt.subplots(1, self.num_imgs, figsize=(fig_size*self.num_imgs, fig_size))

        plt.tight_layout()

        # Show source and target images
        for i in range(self.num_imgs):
            axes[i].imshow(self.imgs[i])
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('source image')
            else:
                axes[i].set_title('target image')

        # Process self.ft without modifying it, if it's 4D
        # Otherwise, assume it's 3D because of transformer permutation, and convert it to 4D
        if len(self.ft.shape) == 3:
            batch_size, hw, channels = self.ft.shape  # Extract dimensions
            h = w = int(hw**0.5)  # Assume the height and width are equal
            self.ft = self.ft.permute(0, 2, 1).reshape(batch_size, channels, h, w)

        print("self_ft size:", self.ft.size(1))
        print("self_ft shape:", self.ft.shape)
        num_channel = self.ft.size(1)

        def onclick(event):
            """
            Event handler for mouse click.
            Computes the cosine similarity between the source feature vector and the target feature vectors on mouse click location.
            """
            if event.inaxes == axes[0]:
                with torch.no_grad():

                    x, y = int(np.round(event.xdata)), int(np.round(event.ydata))

                    print(f"self.ft shape (before unsqueeze): {self.ft.shape}")
                    src_ft = self.ft[0].unsqueeze(0) # -> 3D
                    print(f"src_ft shape (after unsqueeze): {src_ft.shape}")

                    # Process the source feature tensor
                    src_ft_resized = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(src_ft) #nur umbenannt kein relevanter change                    
                    src_vec = src_ft_resized[0, :, y, x].view(1, num_channel)  # Shape: [1, C], nur umbenannt kein relevanter change
                    print(f"src_ft shape: {src_ft_resized.shape}")
                    print(f"src_vec shape: {src_vec.shape}")
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Process the target feature tensors
                    trg_ft_resized = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(self.ft[1:]) # N, C, H, W 
                    trg_vec = trg_ft_resized.view(self.num_imgs - 1, num_channel, -1)  # Shape: [N, C, HW]
                    print(f"trg_ft shape: {trg_ft_resized.shape}")
                    print(f"trg_vec shape: {trg_vec.shape}")
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Normalize vectors
                    src_vec = F.normalize(src_vec, dim=1)  # Shape: [1, C]
                    trg_vec = F.normalize(trg_vec, dim=1)  # Shape: [N, C, HW]

                    # Matrix multiplication
                    try:
                        cos_map = torch.matmul(src_vec, trg_vec).view(self.num_imgs - 1, self.img_size, self.img_size).cpu().numpy() # N, H, W
                    except RuntimeError as e:
                        print(f"Error during matmul: {e}")
                        print(f"src_vec shape: {src_vec.shape}")
                        print(f"trg_vec shape: {trg_vec.shape}")
                        raise

                    axes[0].clear()
                    axes[0].imshow(self.imgs[0])
                    axes[0].axis('off')
                    axes[0].scatter(x, y, c='r', s=scatter_size)
                    axes[0].set_title('source image')

                    for i in range(1, self.num_imgs):
                        max_yx = np.unravel_index(cos_map[i-1].argmax(), cos_map[i-1].shape)
                        axes[i].clear()

                        heatmap = cos_map[i-1]
                        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
                        axes[i].imshow(self.imgs[i])
                        axes[i].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
                        axes[i].axis('off')
                        axes[i].scatter(max_yx[1].item(), max_yx[0].item(), c='r', s=scatter_size)
                        axes[i].set_title('target image')

                    fig.subplots_adjust(right=0.85)  # Some extra space on the right
                    for cax in fig.axes:
                        if cax != axes[0] and cax in axes:
                            # skip if it's just a normal subplot axis
                            continue
                        # If the figure might have a prior colorbar axis, close it:
                        # (some advanced usage might store colorbar axes, but typically not mandatory)
                    
                    # Create a "ScalarMappable" with the same colormap, 0-255 range
                    sm = cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=1), cmap='viridis')
                    cbar = fig.colorbar(
                        sm, ax=axes.ravel().tolist(),
                        orientation='vertical',
                        fraction=0.02, pad=0.1
                    )

                    # If you prefer a label on the colorbar:
                    cbar.set_label('Similarity', rotation=90)
                    plt.figure(figsize=(5,4))
                    plt.hist(heatmap.ravel(), bins=50, color='blue', alpha=0.7)
                    plt.title("Distribution of Heatmap Values")
                    plt.xlabel("Heatmap Value")
                    plt.ylabel("Frequency")
                    plt.show()

                    del cos_map
                    del heatmap
                    gc.collect()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()