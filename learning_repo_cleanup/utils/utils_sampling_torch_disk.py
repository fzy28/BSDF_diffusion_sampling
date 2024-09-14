import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.analytical_brdf_torch import classic_shading_pdf_disk as classic_shading_pdf

# from analytical_brdf_torch import classic_shading_pdf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

M_PI = 3.14159265359
PI_over_4 = 0.78539816339
PI_over_2 = 1.57079632679

def meshgrid_sampling(batchsize):
    theta_l = torch.linspace(0, torch.pi / 2 * (batchsize - 1) / batchsize, batchsize).to(device) 
    phi_l = torch.linspace(-np.pi, -np.pi + 2 * torch.pi * (batchsize - 1) / batchsize, batchsize).to(device) 
    theta, phi = torch.meshgrid(theta_l, phi_l,indexing='ij')
    theta = theta.reshape(-1)
    phi = phi.reshape(-1)
    return theta, phi

def solve_linear_inverse(a, b, t, u):

    v = (b - a) / (2 * t)
    w = u * (b + a) / 2 * t
    x = torch.where(
        v == 0,
        w / a,
        torch.where(
            v > 0,
            ((torch.sqrt(a**2 + 4 * w * v)) - a) / (2 * v),
            -((torch.sqrt(a**2 + 4 * w * v)) + a) / (2 * v),
        ),
    )
    return x


def samples_withjitter(idx, pdf, linespace):

    theta_idx = idx // linespace
    phi_idx = idx % linespace
    num_samples = idx.shape[0]
    batchsize = idx.shape[1]
    # batchsize_indices = torch.arange(idx.shape[1])

    # batchsize_indices = torch.arange(idx.shape[1])[torch.newaxis, :]
    # theta_a = pdf[theta_idx,phi_idx,batchsize_indices]
    # theta_b = pdf[theta_idx+1,phi_idx,batchsize_indices]
    theta_t = (torch.pi / 2) / linespace
    jitter = torch.rand(num_samples, batchsize).to(device)
    theta_jitter = jitter * theta_t
    # theta_jitter = solve_linear_inverse(theta_a,theta_b,theta_t,jitter)
    # phi_a = pdf[theta_idx,phi_idx,:]
    # phi_b = pdf[theta_idx,phi_idx+1,:]
    phi_t = (2 * torch.pi) / linespace 
    jitter = torch.rand(num_samples, batchsize).to(device)
    phi_jitter = jitter * phi_t
    # phi_jitter = solve_linear_inverse(phi_a,phi_b,phi_t,jitter)

    theta_samples = theta_idx * theta_t + theta_jitter
    phi_samples = phi_idx * phi_t + phi_jitter - torch.pi
    omega_o_samples = torch.stack([theta_samples, phi_samples], axis=1)
    return omega_o_samples

def sampling_init(linespace_i,linespace_o):
    theta_i, phi_i = meshgrid_sampling(linespace_i + 1)
    theta_o, phi_o = meshgrid_sampling(linespace_o + 1)
    return theta_i, phi_i, theta_o, phi_o
def neusample_init(WI_RES):
    wiX = torch.linspace(-1.0,1.0, steps = WI_RES)
    wiY = torch.linspace(-1.0,1.0, steps = WI_RES)
    grid_z1, grid_z2 = torch.meshgrid(wiX, wiY)
    gridwi = torch.stack([grid_z1, grid_z2], dim = -1)
    light_dir = gridwi.reshape((-1, 2))
    invalid_dirs = torch.square(light_dir[...,0]) + torch.square(light_dir[...,1]) > 0.995
    return light_dir,invalid_dirs

def stratified_sampling_2d(spp_):
        #round spp to square number
        # torch.manual_seed(42)
        side = 1
        while (side*side< spp_):
            side += 1
        # spp_square = side*side
        # side = int(math.sqrt(spp_square))
        us = torch.arange(0, side)/side 
        vs = torch.arange(0, side)/side
        u, v = torch.meshgrid(us, vs)
        uv = torch.stack([u,v], dim = -1)

        uv = uv.reshape(-1,2)
        # uv = uv[:spp_,...]
        uv = uv[torch.randperm(uv.shape[0]), ...]  #TODO CHECK
        jitter = torch.rand((spp_, 2))/side
        # print("~~~~~~~~~~~~~~~~~~~~~~ stratified samples: ", uv.shape)
        return uv + jitter

def stratified_sample_wo(num_samples):
        wo = stratified_sampling_2d(num_samples) * 2.0 - 1.0

        woSample = torch.zeros(wo.shape)
        zero_positions = torch.logical_and(wo[:, 0] == 0 , wo[:, 1]==0)
        nonzero_positions = ~zero_positions
        condition1 = torch.logical_and(torch.abs(wo[:,0]) > torch.abs(wo[:,1]),nonzero_positions)
        condition2 = torch.logical_and(~condition1 ,nonzero_positions)

        woSample[condition1,0] = wo[condition1,0] * torch.cos(PI_over_4 * wo[condition1,1]/wo[condition1,0])
        woSample[condition1,1] = wo[condition1,0] * torch.sin(PI_over_4 * wo[condition1,1]/wo[condition1,0])

        woSample[condition2,0] = wo[condition2,1] * torch.cos(PI_over_2 - PI_over_4 * wo[condition2,0]/wo[condition2,1])
        woSample[condition2,1] = wo[condition2,1] * torch.sin(PI_over_2 - PI_over_4 * wo[condition2,0]/wo[condition2,1])
        
        return woSample

def online_sampling_neusample(
    brdf_func,light_dir, invalid_dirs,LOADING_SIZE,NUM_WI_SAMPLES):
    camera_dir = stratified_sample_wo(LOADING_SIZE)
    # camera_dir = torch.Tensor([-0.5,-0.5]).to(torch.device("cuda")).reshape(1,2).repeat(LOADING_SIZE,1)
    
    camera_dir_tensor = torch.tile(camera_dir, (light_dir.shape[0], )).reshape(-1,2).cuda()
    light_dir_tensor = torch.tile(light_dir, (camera_dir.shape[0], 1)).reshape(-1,2).cuda()
    with torch.no_grad():
        rgb_pred = brdf_func(camera_dir_tensor, light_dir_tensor)
        rgb_pred[rgb_pred < 0] = 0
    with torch.no_grad():
        rgb_pred = rgb_pred.reshape(LOADING_SIZE,light_dir.shape[0])
        rgb_pred[:, invalid_dirs] = 0

        # rgb_tmp = rgb_pred[0,:].reshape(256, 256)
        # wiX = torch.linspace(-1.0,1.0, steps = 256)
        # wiY = torch.linspace(-1.0,1.0, steps = 256)
        # plt.figure(figsize=(10,10))
        # plt.pcolormesh(wiX.numpy(), wiY.numpy(), rgb_tmp.cpu().numpy())
        
        
        gt_pdf = rgb_pred.flatten().detach().cpu()
        wi_samples = torch.Tensor(samplewi.samplewi(gt_pdf, LOADING_SIZE, NUM_WI_SAMPLES)).reshape(-1, 2)
        
        # wi_samples_tmp = wi_samples.reshape(LOADING_SIZE, NUM_WI_SAMPLES, 2)
        # wi_samples_tmp = wi_samples_tmp[0, ...]
        # plt.figure(figsize=(10,10))
        # plt.hist2d(wi_samples_tmp[:,0].flatten().cpu().numpy(), wi_samples_tmp[:,1].flatten().cpu().numpy(), bins=400,density=True,range=[[-1,1],[-1,1]])
        # plt.show()
        camera_dir_tensor = torch.tile(camera_dir, (NUM_WI_SAMPLES,)).reshape(-1,2)
        wi_samples_in = torch.cat((camera_dir_tensor, wi_samples), dim=-1)   
    wi_samples_in = torch.Tensor(wi_samples_in).to(torch.device("cuda"))
    idx = torch.randperm(wi_samples_in.shape[0])
    return wi_samples_in[idx]
    
def online_sampling(
    brdf_func,light_dir, invalid_dirs,LOADING_SIZE,NUM_WI_SAMPLES,WI_RES
):
    
    camera_dir = stratified_sample_wo(LOADING_SIZE)
    
    camera_dir_tensor = torch.tile(camera_dir, (light_dir.shape[0], )).reshape(-1,2).cuda()
    light_dir_tensor = torch.tile(light_dir, (camera_dir.shape[0], 1)).reshape(-1,2).cuda()
    
    # theta_i_batch = theta_i[i * batchsize : (i + 1) * batchsize]
    # phi_i_batch = phi_i[i * batchsize : (i + 1) * batchsize]
    

    with torch.no_grad():
        rgb_pred = brdf_func(camera_dir_tensor, light_dir_tensor)
        rgb_pred[rgb_pred < 0] = 0
    with torch.no_grad():
        rgb_pred = rgb_pred.reshape(LOADING_SIZE,light_dir.shape[0])
        rgb_pred[:, invalid_dirs] = 0
        
        pdf = rgb_pred.view(LOADING_SIZE, WI_RES, WI_RES).permute(1, 2, 0)

        pmf = (pdf[:-1, :, :] + pdf[1:, :, :]) / 2
        pmf = (pmf[:, :-1, :] + pmf[:, 1:, :]) / 2

        pmf = pmf.reshape(WI_RES * WI_RES, LOADING_SIZE)
        pmf = pmf / torch.sum(pmf, axis=0, keepdims=True)
        cdf = torch.cumsum(pmf, axis=0)
        cdf = cdf.permute(1, 0)
        cdf_contiguous = cdf.contiguous()

        samples_indx = torch.stack([torch.searchsorted(cdf_contiguous[i,:], torch.linspace(0,1,NUM_WI_SAMPLES+1)[:-1].to(device) + torch.randn(NUM_WI_SAMPLES).cuda() / NUM_WI_SAMPLES) for i in range(cdf.shape[0])], dim=1)

    omega_o_samples = samples_withjitter(samples_indx, pdf, WI_RES)
    omega_o_samples = omega_o_samples.permute(2, 0, 1).reshape(-1, 2)
    #omega_o_samples = omega_o_samples.cpu().numpy()
    
   
    
    omega_i = torch.tile(camera_dir, (NUM_WI_SAMPLES,)).reshape(-1,2)
    #omega_i = omega_i.repeat(num_samples, 1)
    return omega_o_samples, omega_i


if __name__ == "__main__":

    def pdf_func(omega_i, omega_o):
        return classic_shading_pdf(omega_i, omega_o, 0.5)

    theta_i, phi_i, theta_o, phi_o = sampling_init(400, 400)
    omega_o_samples, omega_i_condi = online_sampling(pdf_func, (theta_i, phi_i), (theta_o, phi_o), 400, 400,batchsize=2,num_samples=2**22)
    omega_o_samples = omega_o_samples.cpu().numpy()
    omega_o = torch.stack([theta_o, phi_o], dim=1)
    omega_i_condi = omega_i_condi[0 * 2**22,:].repeat(401 * 401,1)
    pdf = pdf_func(omega_i_condi, omega_o)
    plt.figure(figsize=(6, 6))
    plt.pcolormesh(theta_o.cpu().numpy().reshape(401,401), phi_o.cpu().numpy().reshape(401,401), pdf.cpu().numpy().reshape(401,401))
    plt.figure(figsize=(6, 6))
    counts_target, xedges, yedges, c = plt.hist2d(omega_o_samples[0 * 2**22:2**22 * 1,0], omega_o_samples[0 * 2**22:2**22 * 1,1], bins=200,density=True)
    plt.figure(figsize=(6, 6))
    counts_target, xedges, yedges, c = plt.hist2d(omega_o_samples[2 * 2**22:2**22 * 3,0], omega_o_samples[2 * 2**22:2**22 * 3,1], bins=200,density=True)
    omega_o_samples = np.concatenate([omega_o_samples[0 * 2**22:2**22 * 1],omega_o_samples[2 * 2**22:2**22 * 3]],axis=0)
    plt.figure(figsize=(6, 6))
    counts_target, xedges, yedges, c = plt.hist2d(omega_o_samples[:,0], omega_o_samples[:,1], bins=200,density=True)
    plt.show()
    
    print(omega_o_samples.shape)
    print(omega_i_condi.shape)
