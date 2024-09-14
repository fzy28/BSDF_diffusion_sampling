import imageio 
import matplotlib.pyplot
import numpy as np
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn
from mpl_toolkits.axes_grid1 import make_axes_locatable

image_path = './images/'

def load_pytorch_model_to_tinycuda(model,state_dict,input_dims,output_dims):
    state_dict = list(state_dict.values())
    tinycuda_state = []
    tinycuda_state.append(nn.functional.pad(state_dict[0],
                                    pad=(0, 16 - (input_dims % 16), 0, 0)).flatten())
    for weight in state_dict[1:-1]:
        tinycuda_state.append(weight.flatten())
    tinycuda_state.append(nn.functional.pad(state_dict[-1],
                                    pad=(0, 0, 0, 16 - (output_dims % 16))).flatten())
    tinycuda_weights = torch.cat(tinycuda_state).half().to("cuda")
    model.params.data[...] = tinycuda_weights

def eval_arg(value):
    try:
        return eval(value)
    except NameError:
        return value
def save_model(model, save_dir, exp_name):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"{exp_name}.pth"))
def export(x, filename, pdf_func,loc=-1,scale=2):
    x = x.cpu().numpy()
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(x, bins='auto', density=True, alpha=0.6, color='g')
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    pdf = pdf_func(bin_centers,loc=loc,scale=200)  
    plt.plot(bin_centers, pdf, color='r')
    plt.title('Histogram of Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.savefig(os.path.join(image_path,filename))
    plt.close()


def export_withpdf(x, pdf_learn, pdf_func, filename,image_path_more = './'):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    pdf_learn = pdf_learn.cpu().detach().numpy()
    
    pdf_gt = pdf_func(x)
    
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    axs[0].scatter(x, pdf_gt, color='red', label='GT PDF',s=5)
    n, bins, patches = axs[0].hist(x, bins=100, density=True, alpha=0.6, color='g')
    
    axs[0].set_title('Ground Truth PDF')
    axs[0].legend()

    axs[1].scatter(x, pdf_learn, color='blue', label='Learned PDF',s=5)
    n, bins, patches = axs[1].hist(x, bins=100, density=True, alpha=0.6, color='r')

    
    axs[1].set_title('Learned PDF')
    axs[1].legend()

    axs[2].scatter(x, np.abs(pdf_gt - pdf_learn), color='green', label='Difference')
    axs[2].set_title('Difference in PDFs')
    axs[2].legend()

    image_path_tmp = os.path.join(image_path,image_path_more)
    os.makedirs(image_path_tmp, exist_ok=True)
    plt.savefig(os.path.join(image_path_tmp,filename))
    plt.close()


def export_1d(x_alpha,title, filename, range = [[0,np.pi / 2],[0,np.pi]],bins = 200):
    
    
    
    plt.figure(figsize=(10, 8))

    plt.hist(x_alpha.cpu().detach().numpy(),  bins=bins,density=True)
    plt.title(title)

    plt.savefig(os.path.join(image_path,filename))
    plt.close()

def export_2d(x_alpha,title, filename, range = [[-np.pi/2,np.pi / 2],[-np.pi,np.pi]],bins = 200):
    
    
    
    plt.figure(figsize=(10, 8))

    counts_target, xedges, yedges, c = plt.hist2d(x_alpha[:,0], x_alpha[:,1],range=range, bins=bins,density=True)
    plt.title(title)
    plt.colorbar(c)

    plt.savefig(os.path.join(image_path,filename))
    plt.close()

def export_2d_result_pdf(theta,phi,brdf,gt_brdf,filename_prefix,path=image_path,gamma=0.35):

    os.makedirs(path, exist_ok=True)
    vmin = np.min(gt_brdf)
    vmax = np.max(gt_brdf)
    norm = matplotlib.colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    datasets = [brdf, gt_brdf]
    titles = ['learned_pdf', 'ground_truth_pdf']

    for i, data in enumerate(datasets):
        fig, ax = plt.subplots(figsize=(6, 6))  # 设置统一的主图像大小为6x6
        img = ax.pcolormesh(theta, phi, data, cmap='viridis', norm=norm) 
        ax.axis('off')

        if i == 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(img, cax=cax)

        plt.savefig(os.path.join(path, f'{filename_prefix}_{titles[i]}.png'), bbox_inches='tight')
        plt.close(fig)
    diff = brdf - gt_brdf
    fig, ax = plt.subplots(figsize=(6, 6))  # 设置统一的主图像大小为6x6
    norm = matplotlib.colors.PowerNorm(gamma=0.35, vmin=diff.min(), vmax=diff.max())
    img = ax.pcolormesh(theta, phi, diff, cmap='cool')
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(img, cax=cax)
    plt.savefig(os.path.join(path, f'{filename_prefix}_diff.png'), bbox_inches='tight')
    plt.close(fig)

def export_2d_result_withsamples(x,theta,phi,brdf,gt_brdf,filename_prefix,path=image_path,range=[[-1, 1], [-1, 1]],bins=50):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()



    # 使用相同的颜色标准化
    vmin = np.min(gt_brdf)
    vmax = np.max(gt_brdf)
    norm = matplotlib.colors.PowerNorm(gamma=0.3, vmin=vmin, vmax=vmax)

    # 分别为直方图、学习的PDF和真实的PDF创建和保存图像
    datasets = [x, brdf, gt_brdf]
    titles = ['learned_histogram', 'learned_pdf', 'ground_truth_pdf']

    for i, data in enumerate(datasets):
        fig, ax = plt.subplots(figsize=(6, 6))  # 设置统一的主图像大小为6x6
        img = ax.pcolormesh(theta, phi, data, cmap='viridis', norm=norm) if i > 0 else ax.hist2d(x[:, 0], x[:, 1], bins=bins, density=True, range=range, cmap='viridis', norm=norm)
        ax.axis('off')

        # 为最后一个图添加colorbar
        if i == 2:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(img, cax=cax)

        # 保存图像
        plt.savefig(os.path.join(path, f'{filename_prefix}_{titles[i]}.png'), bbox_inches='tight')
        plt.close(fig)

def export_withpdf_2d(x, theta, phi, brdf, gt_brdf, filename, path=image_path, range=[[-1, 1], [-1, 1]], bins=300):
    sum_gt = np.sum(gt_brdf)
    sum_brdf = np.sum(brdf)
    
    # norm_term = sum_gt / sum_brdf
    # print(norm_term)
    #gt_brdf = gt_brdf / norm_term
    
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
        
    fig, axs = plt.subplots(1, 4, figsize=(36, 9))
    
    # 第一个子图
    counts, xedges, yedges, c = axs[0].hist2d(x[:, 0], x[:, 1], bins=bins, density=True, range=range)
    axs[0].set_title('Learned Histogram')

    # 第二个子图
    im = axs[1].pcolormesh(theta, phi, brdf)
    axs[1].set_title('Learned PDF')

    # 第三个子图
    dm = axs[2].pcolormesh(theta, phi, gt_brdf)
    axs[2].set_title('Ground Truth PDF')
    
    from scipy.interpolate import griddata

    # 插值 hist2d 结果以匹配 gt_brdf 的网格
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    histogram_interp = griddata((x_grid.flatten(), y_grid.flatten()), counts.flatten(),
                                (phi, theta), method='cubic', fill_value=0)
    
    # 第四个子图
    diff = np.abs(brdf - gt_brdf)
    cm = axs[3].pcolormesh(theta, phi, diff)
    axs[3].set_title('Difference in PDFs')
    
    brdf_norm = brdf / np.sum(brdf)
    gt_brdf_norm = gt_brdf / np.sum(gt_brdf)
    
    mask = (brdf_norm > 0) & (gt_brdf_norm > 0)
    kl_divergence = np.sum(brdf_norm[mask] * np.log(brdf_norm[mask] / gt_brdf_norm[mask]))
    print(f'KL Divergence: {kl_divergence}')
    
    vmin = gt_brdf.min() + 1e-6
    vmax = gt_brdf.max()
    norm = matplotlib.colors.PowerNorm(gamma=0.5, vmin=gt_brdf.min(), vmax=gt_brdf.max())

    # 设置第一个子图的颜色映射和标准化方法
    c.set_cmap('viridis')
    c.set_norm(norm)

    # 设置第二个子图的颜色映射和标准化方法
    im.set_cmap('viridis')
    im.set_norm(norm)

    dm.set_cmap('viridis')
    dm.set_norm(norm)
    
    cm.set_cmap('viridis')
    cm.set_norm(norm)

    # 创建一个新的轴用于绘制colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    # 绘制colorbar
    cbar = fig.colorbar(im, cax=cbar_ax)

    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path,filename))
    plt.close()
    
