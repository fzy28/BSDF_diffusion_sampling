import imageio 
import matplotlib.pyplot
import numpy as np
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm,uniform
from scipy.stats import truncnorm
from scipy.integrate import quad,quad_vec
from scipy import signal
import OpenEXR
import Imath
import numpy as np

image_path = './images/'
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


def read_exr(filename):
    # 打开EXR文件
    exr_file = OpenEXR.InputFile(filename)
    # 读取图片的元数据
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # 定义数据格式
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    # 读取RGB通道
    redstr = exr_file.channel('R', pt)
    greenstr = exr_file.channel('G', pt)
    bluestr = exr_file.channel('B', pt)

    # 将读取的字符串转换为numpy数组
    red = np.frombuffer(redstr, dtype=np.float32)
    green = np.frombuffer(greenstr, dtype=np.float32)
    blue = np.frombuffer(bluestr, dtype=np.float32)

    red.shape = green.shape = blue.shape = (size[1], size[0])  # 注意：形状是高度x宽度
    return np.stack([red, green, blue], axis=-1)  # 合并成一个多通道数组

def compute_mse(image1, image2):
    # 计算均方误差
    return np.mean((image1 - image2) ** 2)

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

    counts_target, xedges, yedges, c = plt.hist2d(x_alpha[:,0].cpu().detach().numpy(), x_alpha[:,1].cpu().detach().numpy(),range=range, bins=bins,density=True)
    plt.title(title)
    plt.colorbar(c)

    plt.savefig(os.path.join(image_path,filename))
    plt.close()

def export_withpdf_2d(x,theta,phi,brdf,gt_brdf, filename,path = image_path,range = [[-np.pi/2,np.pi / 2],[-np.pi,np.pi]],bins = 200):
    
    sum_gt = np.sum(gt_brdf)
    sum_brdf = np.sum(brdf)
    
    norm_term = sum_gt / sum_brdf
    print(norm_term)
    gt_brdf = gt_brdf 
    
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
        
    fig, axs = plt.subplots(1, 4, figsize=(36, 9))

    # 第一个子图
    counts, _, _, c = axs[0].hist2d(x[:,0], x[:,1], bins=bins, density=True, range=range)
    axs[0].set_title('Learned Histogram')

    # 第二个子图
    im = axs[1].pcolormesh(theta, phi, brdf)
    axs[1].set_title('Learned PDF')

    # 第三个子图
    dm = axs[2].pcolormesh(theta, phi, gt_brdf)
    axs[2].set_title('Ground Truth PDF')
    
    diff = np.abs(brdf - gt_brdf)
    # 第四个子图
    cm = axs[3].pcolormesh(theta, phi, diff)
    axs[3].set_title('Difference in PDFs')
    
    # 创建一个标准化对象,用于将数据映射到[0, 1]区间
    vmin = min(counts.min(), brdf.min())
    vmax = max(counts.max(), brdf.max())
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

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
    # 调整子图之间的间距

    # 创建一个新的轴用于绘制colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    # 绘制colorbar
    cbar = fig.colorbar(im, cax=cbar_ax)

    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path,filename))
    plt.close()
    
# PDF functions
def UniformPDF(x,loc=-1,scale=200):
    return uniform.pdf(x, loc = loc, scale = scale).reshape(-1, 1)

def TrunGaussianPDF(x,clip_a=-1,clip_b=1,loc=0,scale=1):
    a, b = (clip_a - loc) / scale, (clip_b - loc) / scale
    return truncnorm.pdf(x, a, b).reshape(-1, 1)

def GaussianPDF(x,loc=0,scale=1):
    return norm.pdf(x,loc=loc,scale=scale).reshape(-1, 1)

# Sampling functions
    
eps = 1e-6
def generateSamplesFromUniform(Ndata,loc=-1,scale=200):
    p = np.linspace(loc+eps,loc+scale-eps,Ndata).astype("float32").reshape(Ndata, 1)
    j = np.random.rand(Ndata, 1) * scale / Ndata
    return torch.from_numpy(p + j).to("cuda").type(torch.float32)

def generateSamplesFromTrunGaussian(Ndata,clip_a=-1,clip_b=1,loc=0,scale=1):
    a, b = (clip_a - loc) / scale, (clip_b - loc) / scale
    p = truncnorm.ppf(np.clip(np.linspace(0, 1, Ndata) + np.random.rand(Ndata) * 1 / Ndata, eps, 1-eps), a,b).astype("float32").reshape(Ndata, 1)
    return torch.from_numpy(p).to("cuda").type(torch.float32)

def generateSamplesFromGaussian(Ndata,loc=0,scale=1):
    p = norm.ppf(np.clip(np.linspace(0, 1, Ndata) + np.random.rand(Ndata) * 1 / Ndata, eps, 1-eps),loc=loc,scale=scale).astype("float32").reshape(Ndata, 1)
    return torch.from_numpy(p).to("cuda").type(torch.float32)

def generateSamplesFromTwoDistributionsWithAlpha(Ndata,pdf1,pdf2,alpha):
    x_0 = pdf1(Ndata)
    x_1 = pdf2(Ndata)
    x_alpha = (1 - alpha) * x_0 + alpha * x_1
    return x_alpha,x_0,x_1

delta = 1e-4
big_grid = np.arange(-10,10,delta)

def numerical_conv_twopdfs(pdf1,pdf2):

    pmf1 = pdf1(-big_grid)*delta
    pmf2 = pdf2(big_grid)*delta
    conv_pmf = signal.fftconvolve(pmf1,pmf2,'same')
    conv_pdf = conv_pmf/delta
    conv_pdf = conv_pdf.squeeze()
    mean = np.sum(big_grid * conv_pdf) * delta
    variance = np.sum(((big_grid - mean) ** 2) * conv_pdf) * delta
    return conv_pdf, mean, variance
    
    
def analytical_conv_unif_gauss():
    def pdf_conv_unif_gauss(z):
        return 0.5 * (norm.cdf(z+1) - norm.cdf(z-1))
    mean,_ = quad(lambda x: x * pdf_conv_unif_gauss(x), -np.inf, np.inf)
    mean_square,_ = quad(lambda x: x**2 * pdf_conv_unif_gauss(x), -np.inf, np.inf)
    variance = mean_square - mean**2
    return pdf_conv_unif_gauss(big_grid),mean, variance


def mean_posterior_unif_gauss(x_alpha,alpha):
    x_alpha = x_alpha.cpu().detach().numpy().squeeze()
    alpha = alpha.cpu().detach().numpy().squeeze()
    t = alpha / (1-alpha)
    r = (x_alpha +  alpha) / (1-alpha)
    l = (x_alpha - alpha) / (1 - alpha)
    p =  (norm.cdf(r) - norm.cdf(l))
    results = np.zeros_like(x_alpha)
    def joint_pdf(x,i):
        return UniformPDF(x) * GaussianPDF((x_alpha[i]-alpha[i] * x)/(1-alpha[i]))
    for i in range((x_alpha.shape[0])):
        inte,_ = quad(lambda x: (x_alpha[i] - x) * joint_pdf(x,i), -np.inf, np.inf)
        results[i] = inte
    mean = results / p * t
    return mean.reshape(-1,1)