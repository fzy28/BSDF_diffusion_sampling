import emcee
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.analytical_brdf_torch import *
from utils.utils_sampling_torch_disk import stratified_sample_wo,stratified_sampling_2d
import torch


def lnprob_brdf_disk(p,pdf_func,rmax,rmin):
        p = p.reshape(-1,4)
        x0,y0,x1,y1 = p[:,0],p[:,1],p[:,2],p[:,3]
        mask_omegai = x0 ** 2 + y0 ** 2 > rmax ** 2 or x0 ** 2 + y0 ** 2 < rmin ** 2
        
        if x1 ** 2 + y1 ** 2 > 1 or mask_omegai  == 0:
            return -np.inf
        
        pdf_value = pdf_func(p)
        if pdf_value == 0:
            return -np.inf
        return np.log(np.clip(pdf_value, 0, None))

def lnprob_brdf_hemispheri(p,pdf_func,rmax,rmin):
        p = p.reshape(-1,4)
        x0,y0,x1,y1 = p[:,0],p[:,1],p[:,2],p[:,3]
        mask_phi = y0 < np.pi and y0 > -np.pi and y1 < np.pi and y1 > -np.pi
        mask_theta = x1 < np.pi/2 and x1 > 0 and x0 < rmax and x0 > rmin
        if ~mask_phi or ~mask_theta:
            return -np.inf
        pdf_value = pdf_func(p)
        if pdf_value == 0:
            return -np.inf
        return np.log(np.clip(pdf_value, 0, None))

def lnprob_brdf_allspheri(p,pdf_func,rmax,rmin):
        p = p.reshape(-1,4)
        x0,y0,x1,y1 = p[:,0],p[:,1],p[:,2],p[:,3]
        mask_phi = y0 < np.pi and y0 > -np.pi and y1 < np.pi and y1 > -np.pi
        mask_theta = x1 < np.pi and x1 > 0 and x0 < rmax and x0 > rmin
        if ~mask_phi or ~mask_theta:
            return -np.inf
        pdf_value = pdf_func(p)
        if pdf_value == 0:
            return -np.inf
        return np.log(np.clip(pdf_value, 0, None))

def lnprob_bsdf(p,pdf_func):
        p = p.reshape(-1,4)
        x0,y0,x1,y1 = p[:,0],p[:,1],p[:,2],p[:,3]
        mask1 = (x0 -1) ** 2 + y0 ** 2 > 1 and (x0 + 1) ** 2 + y1 ** 2 > 1
        mask2 = (x1 -1) ** 2 + y1 ** 2 > 1 and (x1 + 1) ** 2 + y1 ** 2 > 1
        pdf_value = pdf_func(p)
        if mask1 or mask2 or pdf_value == 0:
            return -np.inf
        return np.log(np.clip(pdf_value, 0, None))

def find_omegao(omegai, pdf_func,is_spherical = False):
    while True:
        if is_spherical:
            omegao = stratified_sampling_2d(1)
            omegao[:,0] = omegao[:,0] * np.pi / 2
            omegao[:,1] = omegao[:,1] * 2 * np.pi - np.pi
        else:
            omegao = stratified_sample_wo(1)
        p = np.concatenate([omegai,omegao],axis=1).reshape(1,4)
        pdf_value = pdf_func(p)
        if pdf_value != 0:
            break
    return omegao

def find_omegao_bsdf(omegai, pdf_func):
    while True:
        
        omegao = stratified_sampling_2d(1)
        omegao[:,0] = omegao[:,0] * np.pi 
        omegao[:,1] = omegao[:,1] * 2 * np.pi - np.pi
        p = np.concatenate([omegai,omegao],axis=1).reshape(1,4)
        pdf_value = pdf_func(p)
        if pdf_value != 0:
            break
    return omegao

def emcee_mcmc_brdf_disk(pdf_func, nsteps, ndim=4,nwalkers = 49,piecewise=10,burn_in=10000):
    omegao = stratified_sample_wo(nwalkers)
    omegao = omegao[np.random.choice(nwalkers, nwalkers, replace=False)]
    omegai_base = stratified_sample_wo(2**22)
    all_samples = []
    for i in range(0, piecewise):
        radius_current_max = (i+1) / piecewise
        radius_current_min = i / piecewise
        mask = torch.logical_and((omegai_base[:,0] ** 2 + omegai_base[:,1] ** 2) < radius_current_max ** 2,  
                                 (omegai_base[:,0] ** 2 + omegai_base[:,1] ** 2) > radius_current_min ** 2)
        omegai = omegai_base[mask]
        omegai = omegai[np.random.choice(omegai.shape[0], nwalkers, replace=False)]
        omegao = []
        for i in range(omegai.shape[0]):
            omegao_i = find_omegao(omegai[i].reshape(1,2),pdf_func)
            omegao.append(omegao_i)
        omegao = np.concatenate(omegao)
        p0 = np.concatenate([omegai,omegao],axis=1).reshape(nwalkers,ndim)
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_brdf_disk,args=(pdf_func,radius_current_max,radius_current_min), pool=pool)
            state = sampler.run_mcmc(p0, burn_in, progress=True)
            print("Burn-in done")
            sampler.reset()
            sampler.run_mcmc(state, nsteps, progress=True)  
        samples = sampler.get_chain(flat=True)
        all_samples.append(samples)
    samples = np.concatenate(all_samples)
    return samples

def emcee_mcmc_brdf_spherical(pdf_func,nsteps, ndim=4,nwalkers = 49,piecewise=10,burn_in=10000):
    omegai_base = stratified_sampling_2d(2**22)
    omegai_base[:,0] = omegai_base[:,0] * np.pi / 2
    omegai_base[:,1] = omegai_base[:,1] * 2 * np.pi - np.pi
    all_samples = []
    for i in range(0, piecewise):
        radius_current_max = (i+1) / piecewise * np.pi / 2
        radius_current_min = i / piecewise * np.pi / 2
        mask = torch.logical_and(omegai_base[:,0] < radius_current_max, omegai_base[:,0] > radius_current_min)
        omegai = omegai_base[mask]
        selected_indices = np.random.choice(omegai.shape[0], nwalkers, replace=False)
        omegai = omegai[selected_indices]
        omegao = []
        for i in range(omegai.shape[0]):
            omegao_i = find_omegao(omegai[i].reshape(1,2),pdf_func,is_spherical=True)
            omegao.append(omegao_i)
        omegao = np.concatenate(omegao)
        p0 = np.concatenate([omegai,omegao],axis=1).reshape(nwalkers,ndim) 
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_brdf_hemispheri,args=(pdf_func,radius_current_max,radius_current_min), pool=pool)
            state = sampler.run_mcmc(p0, burn_in, progress=True)
            print("Burn-in done")
            sampler.reset()
            sampler.run_mcmc(state, nsteps, progress=True)  
        samples = sampler.get_chain(flat=True)
        all_samples.append(samples)
    samples = np.concatenate(all_samples)
    return samples

def emcee_mcmc_bsdf(pdf_func, nsteps, ndim=4,nwalkers = 49,piecewise=10,burn_in=10000):
    omegai_base = stratified_sampling_2d(2**22)
    omegai_base[:,0] = omegai_base[:,0] * np.pi 
    omegai_base[:,1] = omegai_base[:,1] * 2 * np.pi - np.pi
    all_samples = []
    for i in range(0, piecewise):
        radius_current_max = (i+1) / piecewise * np.pi 
        radius_current_min = i / piecewise * np.pi 
        mask = torch.logical_and(omegai_base[:,0] < radius_current_max, omegai_base[:,0] > radius_current_min)
        omegai = omegai_base[mask]
        
        selected_indices = np.random.choice(omegai.shape[0], nwalkers, replace=False)
        omegai = omegai[selected_indices]
        omegao = []
        for i in range(omegai.shape[0]):
            omegao_i = find_omegao_bsdf(omegai[i].reshape(1,2),pdf_func)
            omegao.append(omegao_i)
        omegao = np.concatenate(omegao)
        p0 = np.concatenate([omegai,omegao],axis=1).reshape(nwalkers,ndim) 
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_brdf_allspheri,args=(pdf_func,radius_current_max,radius_current_min), pool=pool)
            state = sampler.run_mcmc(p0, burn_in, progress=True)
            print("Burn-in done")
            sampler.reset()
            sampler.run_mcmc(state, nsteps, progress=True)  
        samples = sampler.get_chain(flat=True)
        all_samples.append(samples)
    samples = np.concatenate(all_samples)
    return samples

if __name__ == "__main__":
    roughness = 0.2
    nwalkers = 49
    nsteps = 200000
    ndim = 4
    samples = emcee_mcmc_brdf(pdf_func_np, nwalkers, nsteps, ndim)
    np.save("brdf_samples.npy", samples)
    print(samples.shape)
