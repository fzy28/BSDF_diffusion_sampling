# %%
import imageio 
import matplotlib.pyplot
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm, truncnorm
from utils.utils import *
from utils.model import *
from utils.distribution import *
import argparse
import ast


import time
def network_sampling_disk(D_base,D_sample,omega_i,T = 4):
    x_target_y = omega_i
    total_time = 0
    x_alpha = D_base.sample(x_target_y,x_target_y.shape[0])
    tmp_J = torch.ones(x_alpha.shape[0],device='cuda')
    ones = torch.ones(x_alpha.shape[0],1,device='cuda').reshape(-1,1)
    zeros = torch.zeros(x_alpha.shape[0],1,device='cuda').reshape(-1,1)
    x_alpha_pdf = D_base.log_prob(x_alpha,x_target_y).exp()    
    x_alpha = x_alpha.detach().requires_grad_(True)
    for t in (range(T)):
        alpha = ( t / T) * torch.ones(x_alpha.shape[0], 1, device="cuda")
        if x_alpha.grad is not None:
            x_alpha.grad.zero_()
        
        d_output = D_sample(x_alpha, alpha,x_target_y)
        #[1,0]
        Jacobian_row_1 = torch.cat((ones, zeros), dim=1)
        d_output.backward(Jacobian_row_1, retain_graph=True) 
        J_grad_x_alpha_1 = x_alpha.grad.clone()
        if x_alpha.grad is not None:
            x_alpha.grad.zero_()
        #[0,1]
        Jacobian_row_2 = torch.cat((zeros, ones), dim=1)
        d_output.backward(Jacobian_row_2)
        J_grad_x_alpha_2 = x_alpha.grad.clone()
        x_alpha = x_alpha + 1 / T * d_output
        x_alpha = x_alpha.detach().requires_grad_(True)
        J_1 = Jacobian_row_1 + 1 / T * J_grad_x_alpha_1
        J_2 = Jacobian_row_2 + 1 / T * J_grad_x_alpha_2
        J = J_1[:,0] * J_2[:,1] - J_1[:,1] * J_2[:,0]
        tmp_J /= J

    x_target_pdf = x_alpha_pdf * tmp_J
    x_alpha = x_alpha.detach()
    return x_alpha, x_target_pdf


def network_sampling_disk_tiny(x_alpha,D_sample,omega_i,T = 4):
    x_target_y = omega_i
    
    tmp_J = torch.ones(x_alpha.shape[0],device='cuda')
    with torch.no_grad():
        for t in (range(T)):
            alpha = ( t / T) * torch.ones(x_alpha.shape[0], 1, device="cuda")
            x_input = torch.cat([x_alpha,alpha,x_target_y],dim = 1)
            d_output = D_sample(x_input)
            d_output = D_sample(x_input)

            x_alpha = x_alpha + 1 / T * d_output

        x_target_pdf = tmp_J
    return x_alpha, x_target_pdf
def network_pdf_disk(D_base,D_sample,omega_o,omega_i,T = 4):
    omega_o = omega_o.detach()
    x_alpha = omega_o.to("cuda",dtype=torch.float32).requires_grad_(True)
    x_target_y = omega_i
    tmp_J = torch.ones(x_alpha.shape[0],device='cuda')
    ones = torch.ones(x_alpha.shape[0],1,device='cuda').reshape(-1,1)
    zeros = torch.zeros(x_alpha.shape[0],1,device='cuda').reshape(-1,1)
    
    for t in (range(T)):
        alpha = (1 - t / T) * torch.ones(x_alpha.shape[0], 1, device="cuda")
        if x_alpha.grad is not None:
            x_alpha.grad.zero_()
        d_output = D_sample(x_alpha, alpha,x_target_y)
        #[1,0]
        Jacobian_row_1 = torch.cat((ones, zeros), dim=1)
        d_output.backward(Jacobian_row_1, retain_graph=True) 
        J_grad_x_alpha_1 = x_alpha.grad.clone()
        if x_alpha.grad is not None:
            x_alpha.grad.zero_()
        #[0,1]
        Jacobian_row_2 = torch.cat((zeros, ones), dim=1)
        d_output.backward(Jacobian_row_2)
        J_grad_x_alpha_2 = x_alpha.grad.clone()
        
        x_alpha = x_alpha - 1 / T * d_output

        x_alpha = x_alpha.detach().requires_grad_(True)
        J_1 = Jacobian_row_1 - 1 / T * J_grad_x_alpha_1
        J_2 = Jacobian_row_2 - 1 / T * J_grad_x_alpha_2
        J = J_1[:,0] * J_2[:,1] - J_1[:,1] * J_2[:,0]
        tmp_J *= J
    with torch.no_grad():
        x_alpha_pdf = D_base.log_prob(x_alpha,x_target_y).exp()
        x_target_pdf = x_alpha_pdf * tmp_J.detach()
    return x_target_pdf


def network_sampling_spherical(D_base,D_sample,omega_i,T = 8):
    x_target_y = omega_i
    x_alpha = D_base.sample(x_target_y,x_target_y.shape[0])
    tmp_J = torch.ones(x_alpha.shape[0],device='cuda')
    ones = torch.ones(x_alpha.shape[0],1,device='cuda').reshape(-1,1)
    zeros = torch.zeros(x_alpha.shape[0],1,device='cuda').reshape(-1,1)
    x_alpha_pdf = D_base.log_prob(x_alpha,x_target_y).exp()    
    x_alpha = x_alpha.detach().requires_grad_(True)
    
    for t in (range(T)):
        alpha = ( t / T) * torch.ones(x_alpha.shape[0], 1, device="cuda")
        if x_alpha.grad is not None:
            x_alpha.grad.zero_()
        x_alpha_predioc = torch.cat([torch.sin( x_alpha[:,1]).reshape(-1,1),torch.cos( x_alpha[:,1]).reshape(-1,1)],dim=1)
        x_alpha_2d = torch.cat([x_alpha[:,0].reshape(-1,1),x_alpha_predioc],dim=1)
        d_output = D_sample(x_alpha_2d, alpha,x_target_y)
        #[1,0]
        Jacobian_row_1 = torch.cat((ones, zeros), dim=1)
        d_output.backward(Jacobian_row_1, retain_graph=True) 
        J_grad_x_alpha_1 = x_alpha.grad.clone()
        if x_alpha.grad is not None:
            x_alpha.grad.zero_()
        #[0,1]
        Jacobian_row_2 = torch.cat((zeros, ones), dim=1)
        d_output.backward(Jacobian_row_2)
        J_grad_x_alpha_2 = x_alpha.grad.clone()
        x_alpha = x_alpha + 1 / T * d_output
        x_alpha = x_alpha.detach().requires_grad_(True)
        J_1 = Jacobian_row_1 + 1 / T * J_grad_x_alpha_1
        J_2 = Jacobian_row_2 + 1 / T * J_grad_x_alpha_2
        J = J_1[:,0] * J_2[:,1] - J_1[:,1] * J_2[:,0]
        tmp_J /= J 
    x_target_pdf = x_alpha_pdf * tmp_J
    x_alpha = x_alpha.detach()
    return x_alpha, x_target_pdf



def network_pdf_spherical(D_base,D_sample,omega_o,omega_i,T = 8):
    omega_o = omega_o.detach()
    x_alpha = omega_o.to("cuda",dtype=torch.float32).requires_grad_(True)
    x_target_y = omega_i
    tmp_J = torch.ones(x_alpha.shape[0],device='cuda')
    ones = torch.ones(x_alpha.shape[0],1,device='cuda').reshape(-1,1)
    zeros = torch.zeros(x_alpha.shape[0],1,device='cuda').reshape(-1,1)
    
    for t in (range(T)):
        alpha = (1 - t / T) * torch.ones(x_alpha.shape[0], 1, device="cuda")
        if x_alpha.grad is not None:
            x_alpha.grad.zero_()
        x_alpha_predioc = torch.cat([torch.sin( x_alpha[:,1]).reshape(-1,1),torch.cos( x_alpha[:,1]).reshape(-1,1)],dim=1)
        x_alpha_2d = torch.cat([x_alpha[:,0].reshape(-1,1),x_alpha_predioc],dim=1)
        d_output = D_sample(x_alpha_2d, alpha,x_target_y)
        #[1,0]
        Jacobian_row_1 = torch.cat((ones, zeros), dim=1)
        d_output.backward(Jacobian_row_1, retain_graph=True) 
        J_grad_x_alpha_1 = x_alpha.grad.clone()
        if x_alpha.grad is not None:
            x_alpha.grad.zero_()
        #[0,1]
        Jacobian_row_2 = torch.cat((zeros, ones), dim=1)
        d_output.backward(Jacobian_row_2)
        J_grad_x_alpha_2 = x_alpha.grad.clone()
        
        x_alpha = x_alpha - 1 / T * d_output

        x_alpha = x_alpha.detach().requires_grad_(True)
        J_1 = Jacobian_row_1 - 1 / T * J_grad_x_alpha_1
        J_2 = Jacobian_row_2 - 1 / T * J_grad_x_alpha_2
        J = J_1[:,0] * J_2[:,1] - J_1[:,1] * J_2[:,0]
        tmp_J *= J
    x_alpha = x_alpha.detach()
    with torch.no_grad():
        x_alpha_pdf = D_base.log_prob(x_alpha,x_target_y).exp()
        x_target_pdf = x_alpha_pdf * tmp_J.detach()
    return x_target_pdf