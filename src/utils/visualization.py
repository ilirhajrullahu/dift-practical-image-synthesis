import gc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Demo:

    def __init__(self, imgs, ft, img_size):
        self.ft = ft # N+1, C, H, W
        self.imgs = imgs
        self.num_imgs = len(imgs)
        self.img_size = img_size

    def plot_img_pairs(self, fig_size=3, alpha=0.45, scatter_size=70):

        fig, axes = plt.subplots(1, self.num_imgs, figsize=(fig_size*self.num_imgs, fig_size))

        plt.tight_layout()

        for i in range(self.num_imgs):
            axes[i].imshow(self.imgs[i])
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('source image')
            else:
                axes[i].set_title('target image')



        # Process self.ft without modifying it
        if len(self.ft.shape) == 3:
            batch_size, hw, channels = self.ft.shape  # Extract dimensions
            h = w = int(hw**0.5)  # Assume the height and width are equal -> ODER LIEGT HIER DER FEHLER? TODO!
            self.ft = self.ft.permute(0, 2, 1).reshape(batch_size, channels, h, w)
            print("if")
        else:
            src_ft = self.ft  # Assuming it's already [batch_size, channels, h, w]
            print("else")

        print("self_ft size:", self.ft.size(1))
        print("self_ft shape:", self.ft.shape)
        num_channel = self.ft.size(1)
        #num_channel = self.ft.size(2)

        def onclick(event):
            if event.inaxes == axes[0]:
                with torch.no_grad():

                    x, y = int(np.round(event.xdata)), int(np.round(event.ydata))

                    print(f"self.ft shape (before unsqueeze?): {self.ft.shape}")
                    src_ft = self.ft[0].unsqueeze(0) # -> 3D
                    print(f"src_ft shape (after unsqueeze?): {src_ft.shape}")

                    # Process the source feature tensor
                    src_ft_resized = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(src_ft) #nur umbenannt kein relevanter change                    
                    src_vec = src_ft_resized[0, :, y, x].view(1, num_channel)  # Shape: [1, C], nur umbenannt kein relevanter change
                    print(f"src_ft shape: {src_ft_resized.shape}")
                    print(f"src_vec shape: {src_vec.shape}")
                    #del src_ft #muss das auskommentiert sein?
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Process the target feature tensors
                    #trg_ft_resized = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(src_ft[1:]) # N, C, H, W #warum auf scr_ft statt self.ft?
                    trg_ft_resized = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(self.ft[1:]) # N, C, H, W ## original, was ist hier genau der Unterschied?
                    trg_vec = trg_ft_resized.view(self.num_imgs - 1, num_channel, -1)  # Shape: [N, C, HW], nur umbenannt kein relevanter change
                    print(f"trg_ft shape: {trg_ft_resized.shape}")
                    print(f"trg_vec shape: {trg_vec.shape}")
                    #del trg_ft  #muss das auskommentiert sein?
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Normalize vectors
                    src_vec = F.normalize(src_vec, dim=1)  # Shape: [1, C]
                    trg_vec = F.normalize(trg_vec, dim=1)  # Shape: [N, C, HW]

                    # Matrix multiplication
                    try:
                        cos_map = torch.matmul(src_vec, trg_vec).view(self.num_imgs - 1, self.img_size, self.img_size).cpu().numpy() # N, H, W is unchanged
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

                    del cos_map
                    del heatmap
                    gc.collect()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()