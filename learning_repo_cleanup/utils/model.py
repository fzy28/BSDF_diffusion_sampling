import torch
import numpy as np
import torch.nn as nn
N_NEURONS = 64
POSITIONAL_ENCODING_BASIS_NUM = 5

# architecture

def positional_encoding_1(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
  r"""Apply positional encoding to the input.

  Args:
    tensor (torch.Tensor): Input tensor to be positionally encoded.
    num_encoding_functions (optional, int): Number of encoding functions used to
        compute a positional encoding (default: 6).
    include_input (optional, bool): Whether or not to include the input in the
        computed positional encoding (default: True).
    log_sampling (optional, bool): Sample logarithmically in frequency space, as
        opposed to linearly (default: True).
  
  Returns:
    (torch.Tensor): Positional encoding of the input tensor.
  """
  encoding = [tensor] if include_input else []
  # Now, encode the input using a set of high-frequency functions and append the
  # resulting values to the encoding.
  frequency_bands = None
  
  if log_sampling:
      frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
  else:
      frequency_bands = torch.linspace(
          2.0 ** 0.0,
          2.0 ** (num_encoding_functions - 1),
          num_encoding_functions,
          dtype=tensor.dtype,
          device=tensor.device,
      )

  for freq in frequency_bands:
      for func in [torch.sin, torch.cos]:
          encoding.append(func(tensor * freq))

  # Special case, for no positional encoding
  if len(encoding) == 1:
      return encoding[0]
  else:
    #   print("shape of encoding: ", len(encoding))
      return torch.cat(encoding, dim=-1)

class NN_albedo(torch.nn.Module):
    def __init__(self,input_dim=2,output_dim=1,N_NEURONS=32):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim,N_NEURONS)  # input = (x_alpha, alpha)
        # self.linear2 = torch.nn.Linear(N_NEURONS, N_NEURONS)  
        # self.linear3 = torch.nn.Linear(N_NEURONS, N_NEURONS)   
        # self.linear4 = torch.nn.Linear(N_NEURONS, N_NEURONS)   
        self.output  = torch.nn.Linear(N_NEURONS, output_dim) 
        self.relu = torch.nn.SiLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        res = self.relu(self.linear1(x))
        # res = self.relu(self.linear2(res))
        # res = self.relu(self.linear3(res))
        # res = self.relu(self.linear4(res))
        res = self.sigmoid(self.output(res))
        return res

class NN(torch.nn.Module):
    def __init__(self,input_dim=2,output_dim=1):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim,N_NEURONS)  # input = (x_alpha, alpha)
        self.linear2 = torch.nn.Linear(N_NEURONS, N_NEURONS)  
        self.linear3 = torch.nn.Linear(N_NEURONS, N_NEURONS)   
        self.linear4 = torch.nn.Linear(N_NEURONS, N_NEURONS)   
        self.output  = torch.nn.Linear(N_NEURONS, output_dim) 
        self.relu = torch.nn.SiLU()

    def forward(self, x, alpha):
        # print(x.dtype)
        # print(alpha.dtype)
        # print(self.linear1.weight.dtype)
        res = torch.cat([x, alpha], dim=1)
        res = self.relu(self.linear1(res))
        res = self.relu(self.linear2(res))
        res = self.relu(self.linear3(res))
        res = self.relu(self.linear4(res))
        res = self.output(res)
        return res


class NN_simpler(torch.nn.Module):
    def __init__(self,input_dim=2,output_dim=1,N_NEURONS=32):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim,N_NEURONS)  # input = (x_alpha, alpha)
        self.linear2 = torch.nn.Linear(N_NEURONS, N_NEURONS)  
        self.output  = torch.nn.Linear(N_NEURONS, output_dim) 
        self.relu = torch.nn.SiLU()

    def forward(self, x, alpha):
        # print(x.dtype)
        # print(alpha.dtype)
        # print(self.linear1.weight.dtype)
        res = torch.cat([x, alpha], dim=1)
        res = self.relu(self.linear1(res))
        res = self.relu(self.linear2(res))
        res = self.output(res)
        return res

class NN_cond(torch.nn.Module):
    def __init__(self,input_dim=3,output_dim=1):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim,N_NEURONS)  # input = (x_alpha, alpha)
        self.linear2 = torch.nn.Linear(N_NEURONS, N_NEURONS)  
        self.linear3 = torch.nn.Linear(N_NEURONS, N_NEURONS)   
        self.linear4 = torch.nn.Linear(N_NEURONS, N_NEURONS)   
        #self.linear5 = torch.nn.Linear(N_NEURONS, N_NEURONS)
        self.output  = torch.nn.Linear(N_NEURONS, output_dim) 
        self.relu = torch.nn.SiLU()

    def forward(self, x, alpha,x_co):
        # print(x.dtype)
        # print(alpha.dtype)
        # print(self.linear1.weight.dtype)
        res = torch.cat([x, alpha,x_co], dim=1)
        res = self.relu(self.linear1(res))
        res = self.relu(self.linear2(res))
        res = self.relu(self.linear3(res))
        res = self.relu(self.linear4(res))
        #res = self.relu(self.linear5(res))
        res = self.output(res)
        return res
    

class NN_cond_pos_neusample(torch.nn.Module):
    def __init__(self,input_dim=3,output_dim=1,N_NEURONS=64):
        super().__init__()
        self.input_dim = input_dim + 2 * POSITIONAL_ENCODING_BASIS_NUM * 2
        self.linear1 = torch.nn.Linear(self.input_dim,N_NEURONS)  # input = (x_alpha, alpha)
        self.linear2 = torch.nn.Linear(N_NEURONS, N_NEURONS)  
        self.linear3 = torch.nn.Linear(N_NEURONS, N_NEURONS)   
        #self.linear5 = torch.nn.Linear(N_NEURONS, N_NEURONS)
        self.output  = torch.nn.Linear(N_NEURONS, output_dim) 
        self.relu = torch.nn.SiLU()

    def forward(self, x, alpha,x_co):
        # print(x.dtype)
        # print(alpha.dtype)
        # print(self.linear1.weight.dtype)
        x_co = positional_encoding_1(x_co, num_encoding_functions=POSITIONAL_ENCODING_BASIS_NUM)
        res = torch.cat([x, alpha,x_co], dim=1)
        res = self.relu(self.linear1(res))
        res = self.relu(self.linear2(res))
        res = self.relu(self.linear3(res))
        #res = self.relu(self.linear5(res))
        res = self.output(res)
        return res



class NN_cond_pretrain_spherical(torch.nn.Module):
    def __init__(self, input_dim=3, n_modes= 2, N_NEURONS=64):
        super().__init__()
        self.input_dim = input_dim + 2 * POSITIONAL_ENCODING_BASIS_NUM * 2
        self.linear1 = torch.nn.Linear(self.input_dim, N_NEURONS)
        self.n_modes = n_modes
        self.output = torch.nn.Linear(N_NEURONS, n_modes * 4 + n_modes )
        self.eps = 1e-6
        self.relu = torch.nn.SiLU()
        self.softplus = torch.nn.Softplus()
    def forward(self, x_co):
        x_co = positional_encoding_1(x_co, num_encoding_functions=POSITIONAL_ENCODING_BASIS_NUM)
        res = self.relu(self.linear1(x_co))
        res = self.output(res)
        return res
    def get_param(self,x_co):
        pred = self.forward(x_co)
        loc,log_scale, weights = pred[:,:self.n_modes],pred[:,self.n_modes:self.n_modes*2],pred[:,self.n_modes*2:self.n_modes*3]
        loc_von,concentration = pred[:,self.n_modes*3:self.n_modes*4],self.softplus(pred[:,self.n_modes*4:]) + self.eps
        weights = torch.abs(weights)
        weights /= torch.sum(weights, dim=1).unsqueeze(-1)
        loc = loc.reshape(-1,self.n_modes,1)
        log_scale = log_scale.reshape(-1,self.n_modes,1)
        weights = weights.reshape(-1,self.n_modes)
        return loc,log_scale,weights,loc_von,concentration
    def sample(self,x_co,numsamples = 1):
        loc,log_scale, weights,loc_von,concentration = self.get_param(x_co)        
        rdn = torch.rand(numsamples, device=x_co.device)
        cumulative_weights = torch.cumsum(weights, dim=-1)
        mode = torch.zeros((numsamples, 1), dtype=torch.long, device=x_co.device)
        for i in range(self.n_modes):
            if i == 0:
                mask = rdn < cumulative_weights[...,i]
            else:
                mask = (rdn >= cumulative_weights[...,i-1]) & (rdn < cumulative_weights[...,i])
            mode[mask] = i
        mode_1h = nn.functional.one_hot(mode, self.n_modes).squeeze().unsqueeze(-1)
        eps = torch.randn(numsamples, 1, device=x_co.device)  
        scale_sample = torch.sum((torch.exp(log_scale) + self.eps) * mode_1h, 1)
        loc_sample = torch.sum(loc * mode_1h, 1)
        loc_von_sample = torch.sum(loc_von * mode_1h.squeeze(), 1)
        concentration_sample = torch.sum(concentration * mode_1h.squeeze(), 1)
        
        x = loc_sample + scale_sample * eps
        y = torch.distributions.von_mises.VonMises(loc_von_sample,concentration_sample).sample((1,)).transpose(0,1)
        samples = torch.cat([x,y],dim=1)
        return samples
    def log_prob(self,x,x_co):
        loc,log_scale, weights,loc_von,concentration = self.get_param(x_co)
        eps = (x[:, None, 0:1] - loc) / (torch.exp(log_scale)+self.eps)
        loggau = (
            -0.5 * 1 * np.log(2 * np.pi)
            + torch.log(weights+1e-6) 
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )
        loc_von = loc_von.flatten()
        concentration = concentration.flatten()
        tmpx = x[:,1].repeat_interleave(self.n_modes)
        logvon = torch.distributions.von_mises.VonMises(loc_von,concentration).log_prob(tmpx)
        logvon = logvon.reshape(-1,self.n_modes)
        logp = loggau + logvon
        logp = torch.logsumexp(logp, 1)
        return logp



class NN_cond_pretrain_spherical_modified_after(torch.nn.Module):
    def __init__(self, input_dim=3, N_NEURONS=16,POSITIONAL_ENCODING_BASIS_NUM=3):
        super().__init__()
        self.input_dim = input_dim + 2 * POSITIONAL_ENCODING_BASIS_NUM * 2
        self.linear1 = torch.nn.Linear(self.input_dim, N_NEURONS)
        self.n_modes = 1
        self.POSITIONAL_ENCODING_BASIS_NUM = POSITIONAL_ENCODING_BASIS_NUM
        self.output = torch.nn.Linear(N_NEURONS, 4)
        self.relu = torch.nn.SiLU()
        self.previous = None
    def forward(self, x_co):
        x_co = positional_encoding_1(x_co, num_encoding_functions=self.POSITIONAL_ENCODING_BASIS_NUM)
        res = self.relu(self.linear1(x_co))
        res = self.output(res)
        return res
    def get_param(self,x_co):
        pred = self.forward(x_co)
        loc,log_scale = pred[:,:self.n_modes],pred[:,self.n_modes:self.n_modes*2]
        loc_von,log_concentration = pred[:,self.n_modes*2:self.n_modes*3],pred[:,self.n_modes*3:self.n_modes*4] 
        return loc,log_scale,loc_von,log_concentration
    def sample(self,x_co,numsamples = 1):
        loc,log_scale,loc_von,log_concentration = self.get_param(x_co)        
        eps = torch.randn_like(loc)
        x = loc + eps * torch.exp(log_scale) 
        loc_von = loc_von.flatten()
        log_concentration = log_concentration.flatten()
        y = torch.distributions.von_mises.VonMises(loc_von,torch.exp(log_concentration)).sample().reshape(-1,1)
        samples = torch.cat([x,y],dim=1)
        return samples
    def log_prob(self,x,x_co):
        loc,log_scale,loc_von,log_concentration = self.get_param(x_co)
        eps = (x[:,0:1] - loc) / torch.exp(log_scale)
        loggau = -0.5 * 1 * np.log(2 * torch.pi) - torch.sum(log_scale, dim=1) - 0.5 * torch.sum(eps ** 2, dim=1)
        loc_von = loc_von.flatten()
        log_concentration = log_concentration.flatten()
        logvon = torch.distributions.von_mises.VonMises(loc_von,torch.exp(log_concentration)).log_prob(x[:,1])
        logp = loggau + logvon
        self.previous = logp
        return logp

class NN_cond_pretrain_spherical_one(torch.nn.Module):
    def __init__(self, input_dim=3, N_NEURONS=16,POSITIONAL_ENCODING_BASIS_NUM=3):
        super().__init__()
        self.input_dim = input_dim + 2 * POSITIONAL_ENCODING_BASIS_NUM * 2
        self.linear1 = torch.nn.Linear(self.input_dim, N_NEURONS)
        self.n_modes = 1
        self.POSITIONAL_ENCODING_BASIS_NUM = POSITIONAL_ENCODING_BASIS_NUM
        self.output = torch.nn.Linear(N_NEURONS, 4)
        self.relu = torch.nn.SiLU()
        self.softplus = torch.nn.Softplus()
        self.eps = 1e-3
        self.previous = None
    def forward(self, x_co):
        x_co = positional_encoding_1(x_co, num_encoding_functions=self.POSITIONAL_ENCODING_BASIS_NUM)
        res = self.relu(self.linear1(x_co))
        res = self.output(res)
        return res
    def get_param(self,x_co):
        pred = self.forward(x_co)
        loc,log_scale = pred[:,:self.n_modes],pred[:,self.n_modes:self.n_modes*2]
        loc_von,concentration = pred[:,self.n_modes*2:self.n_modes*3],self.softplus(pred[:,self.n_modes*3:self.n_modes*4]) + self.eps
        return loc,log_scale,loc_von,concentration
    def sample(self,x_co,numsamples = 1):
        loc,log_scale,loc_von,concentration = self.get_param(x_co)        
        eps = torch.randn_like(loc)
        x = loc + eps * (torch.exp(log_scale) + self.eps)
        loc_von = loc_von.flatten()
        concentration = concentration.flatten()
        y = torch.distributions.von_mises.VonMises(loc_von,concentration).sample().reshape(-1,1)
        samples = torch.cat([x,y],dim=1)
        return samples
    def log_prob(self,x,x_co):
        loc,log_scale,loc_von,concentration = self.get_param(x_co)
        eps = (x[:,0:1] - loc) / (torch.exp(log_scale)+self.eps)
        loggau = -0.5 * 1 * np.log(2 * torch.pi) - torch.sum(log_scale, dim=1) - 0.5 * torch.sum(eps ** 2, dim=1)
        loc_von = loc_von.flatten()
        concentration = concentration.flatten()
        logvon = torch.distributions.von_mises.VonMises(loc_von,concentration).log_prob(x[:,1])
        logp = loggau + logvon
        self.previous = logp
        return logp

class NN_cond_pretrain(torch.nn.Module):
    def __init__(self, input_dim=3, n_modes= 2, N_NEURONS=64):
        super().__init__()
        self.input_dim = input_dim + 2 * POSITIONAL_ENCODING_BASIS_NUM * 2
        self.linear1 = torch.nn.Linear(self.input_dim, N_NEURONS)
        self.n_modes = n_modes
        self.output = torch.nn.Linear(N_NEURONS, n_modes * 4 + n_modes)
        self.relu = torch.nn.LeakyReLU()
    def forward(self, x_co):
        x_co = positional_encoding_1(x_co, num_encoding_functions=POSITIONAL_ENCODING_BASIS_NUM)
        res = self.relu(self.linear1(x_co))
        res = self.output(res)
        return res
    def get_param(self,x_co):
        pred = self.forward(x_co)
        loc,log_scale, weights = pred[:,:self.n_modes*2],pred[:,self.n_modes*2:self.n_modes*4],pred[:,self.n_modes*4:]
        weights = torch.abs(weights)
        weights /= torch.sum(weights, dim=1).unsqueeze(-1)
        loc = loc.reshape(-1,self.n_modes,2)
        log_scale = log_scale.reshape(-1,self.n_modes,2)
        weights = weights.reshape(-1,self.n_modes)
        return loc,log_scale,weights
    def sample(self,x_co,numsamples = 1):
        loc,log_scale, weights = self.get_param(x_co)
        
        rdn = torch.rand(numsamples, device=x_co.device)
        cumulative_weights = torch.cumsum(weights, dim=-1)
        mode = torch.zeros((numsamples, 1), dtype=torch.long, device=x_co.device)
        for i in range(self.n_modes):
            if i == 0:
                mask = rdn < cumulative_weights[...,i]
            else:
                mask = (rdn >= cumulative_weights[...,i-1]) & (rdn < cumulative_weights[...,i])
            mode[mask] = i

        mode_1h = nn.functional.one_hot(mode, self.n_modes).squeeze().unsqueeze(-1)

        eps = torch.randn(numsamples, 2, device=x_co.device)  
        scale_sample = torch.sum(torch.exp(log_scale) * mode_1h, 1)
        loc_sample = torch.sum(loc * mode_1h, 1)
        x = loc_sample + scale_sample * eps

        return x
    def log_prob(self,x,x_co):
        loc,log_scale, weights = self.get_param(x_co)
        eps = (x[:, None, :] - loc) / torch.exp(log_scale)
        logp = (
            -0.5 * 2 * np.log(2 * np.pi)
            + torch.log(weights+1e-10) 
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )
        logp = torch.logsumexp(logp, 1)
        return logp

class NN_cond_pretrain_disk_one(torch.nn.Module):
    def __init__(self, input_dim=3, output_dim=4, N_NEURONS=16,POSITIONAL_ENCODING_BASIS_NUM=5):
        super().__init__()
        self.input_dim = input_dim + 2 * POSITIONAL_ENCODING_BASIS_NUM * 2
        self.linear1 = torch.nn.Linear(self.input_dim, N_NEURONS)
        self.POSITIONAL_ENCODING_BASIS_NUM = POSITIONAL_ENCODING_BASIS_NUM
        self.output = torch.nn.Linear(N_NEURONS, output_dim)
        self.relu = torch.nn.SiLU()
    def forward(self, x_co):
        x_co = positional_encoding_1(x_co, num_encoding_functions=self.POSITIONAL_ENCODING_BASIS_NUM)
        res = self.relu(self.linear1(x_co))
        res = self.output(res)
        return res
    def sample(self,x_co,numsamples = 1):
        pred = self.forward(x_co)
        loc,log_scale = pred[:,:2],pred[:,2:]
        eps = torch.randn_like(loc)
        x = loc + eps * torch.exp(log_scale)
        return  x
    def log_prob(self,x,x_co):
        pred = self.forward(x_co)
        loc,log_scale = pred[:,:2],pred[:,2:]
        eps = (x - loc) / (torch.exp(log_scale))
        logp = -0.5 * 2 * np.log(2 * torch.pi) - torch.sum(log_scale, dim=1) - 0.5 * torch.sum(eps ** 2, dim=1)
        return logp
class NN_cond_pos_simpler(torch.nn.Module):
    def __init__(self,input_dim=3,output_dim=1,N_NEURONS=64,POSITIONAL_ENCODING_BASIS_NUM=5):
        super().__init__()
        self.input_dim = input_dim + 2 * POSITIONAL_ENCODING_BASIS_NUM * 2
        self.linear1 = torch.nn.Linear(self.input_dim,N_NEURONS,bias=False)  # input = (x_alpha, alpha)
        self.linear2 = torch.nn.Linear(N_NEURONS, N_NEURONS,bias=False)  
        self.POSITIONAL_ENCODING_BASIS_NUM = POSITIONAL_ENCODING_BASIS_NUM
        #self.linear5 = torch.nn.Linear(N_NEURONS, N_NEURONS)
        self.output  = torch.nn.Linear(N_NEURONS, output_dim,bias=False) 
        self.relu = torch.nn.SiLU()

    def forward(self, x, alpha,x_co):
        # print(x.dtype)
        # print(alpha.dtype)
        # print(self.linear1.weight.dtype)
        x_co = positional_encoding_1(x_co, num_encoding_functions=self.POSITIONAL_ENCODING_BASIS_NUM)
        res = torch.cat([x, alpha,x_co], dim=1)
        res = self.relu(self.linear1(res))
        res = self.relu(self.linear2(res))
        #res = self.relu(self.linear5(res))
        res = self.output(res)
        return res        

class NN_cond_pos(torch.nn.Module):
    def __init__(self,input_dim=3,output_dim=1,N_NEURONS=64,POSITIONAL_ENCODING_BASIS_NUM=5):
        super().__init__()
        self.input_dim = input_dim + 2 * POSITIONAL_ENCODING_BASIS_NUM * 2
        self.pos_num = POSITIONAL_ENCODING_BASIS_NUM
        self.linear1 = torch.nn.Linear(self.input_dim,N_NEURONS,bias=False)  # input = (x_alpha, alpha)
        self.linear2 = torch.nn.Linear(N_NEURONS, N_NEURONS,bias=False)  
        self.linear3 = torch.nn.Linear(N_NEURONS, N_NEURONS,bias=False)   
        self.linear4 = torch.nn.Linear(N_NEURONS, N_NEURONS,bias=False)   
        #self.linear5 = torch.nn.Linear(N_NEURONS, N_NEURONS)
        self.output  = torch.nn.Linear(N_NEURONS, output_dim,bias=False) 
        self.relu = torch.nn.SiLU() 
    def forward(self, x, alpha,x_co):
        # print(x.dtype)
        # print(alpha.dtype)
        # print(self.linear1.weight.dtype)
        x_co = positional_encoding_1(x_co, num_encoding_functions=self.pos_num)
        res = torch.cat([x, alpha,x_co], dim=1)
        res = self.relu(self.linear1(res))
        res = self.relu(self.linear2(res))
        res = self.relu(self.linear3(res))
        res = self.relu(self.linear4(res))
        #res = self.relu(self.linear5(res))
        res = self.output(res)
        return res


class NN_cond_pos_spherical_complicate(torch.nn.Module):
    def __init__(self,input_dim=3,output_dim=1,N_NEURONS=64,POSITIONAL_ENCODING_BASIS_NUM=5):
        super().__init__()
        self.input_dim = input_dim + 2 * POSITIONAL_ENCODING_BASIS_NUM * 2
        self.pos_num = POSITIONAL_ENCODING_BASIS_NUM
        self.linear1 = torch.nn.Linear(self.input_dim,N_NEURONS,bias=False)  # input = (x_alpha, alpha)
        self.linear2 = torch.nn.Linear(N_NEURONS, N_NEURONS,bias=False)  
        self.linear3 = torch.nn.Linear(N_NEURONS, N_NEURONS,bias=False)   
        self.linear4 = torch.nn.Linear(N_NEURONS, N_NEURONS,bias=False)   
        self.linear5 = torch.nn.Linear(N_NEURONS, N_NEURONS,bias=False)   
        self.linear6 = torch.nn.Linear(N_NEURONS, N_NEURONS,bias=False)   

        #self.linear5 = torch.nn.Linear(N_NEURONS, N_NEURONS)
        self.output  = torch.nn.Linear(N_NEURONS, output_dim,bias=False) 
        self.relu = torch.nn.SiLU() 
    def forward(self, x, alpha,x_co):
        # print(x.dtype)
        # print(alpha.dtype)
        # print(self.linear1.weight.dtype)
        x_co = positional_encoding_1(x_co, num_encoding_functions=self.pos_num)
        res = torch.cat([x, alpha,x_co], dim=1)
        res = self.relu(self.linear1(res))
        res = self.relu(self.linear2(res))
        res = self.relu(self.linear3(res))
        res = self.relu(self.linear4(res))
        res = self.relu(self.linear5(res))
        res = self.relu(self.linear6(res))
        res = self.output(res)
        return res

class NN_cond_pos_simpler(torch.nn.Module):
    def __init__(self,input_dim=3,output_dim=1,N_NEURONS=32,POSITIONAL_ENCODING_BASIS_NUM=5):
        super().__init__()
        self.input_dim = input_dim + 2 * POSITIONAL_ENCODING_BASIS_NUM * 2
        self.pos_num = POSITIONAL_ENCODING_BASIS_NUM
        self.linear1 = torch.nn.Linear(self.input_dim,N_NEURONS,bias=False)  # input = (x_alpha, alpha)
        self.linear2 = torch.nn.Linear(N_NEURONS, N_NEURONS,bias=False)  
        self.linear3 = torch.nn.Linear(N_NEURONS, N_NEURONS,bias=False)   
        #self.linear5 = torch.nn.Linear(N_NEURONS, N_NEURONS)
        self.output  = torch.nn.Linear(N_NEURONS, output_dim,bias=False) 
        self.relu = torch.nn.SiLU() 
    def forward(self, x, alpha,x_co):
        # print(x.dtype)
        # print(alpha.dtype)
        # print(self.linear1.weight.dtype)
        x_co = positional_encoding_1(x_co, num_encoding_functions=self.pos_num)
        res = torch.cat([x, alpha,x_co], dim=1)
        res = self.relu(self.linear1(res))
        res = self.relu(self.linear2(res))
        res = self.relu(self.linear3(res))
        #res = self.relu(self.linear5(res))
        res = self.output(res)
        return res

class NN_cond_pos_spherical_simpler(torch.nn.Module):
    def __init__(self,input_dim=3,output_dim=1,N_NEURONS=32,POSITIONAL_ENCODING_BASIS_NUM=5):
        super().__init__()
        self.input_dim = input_dim + 2 * POSITIONAL_ENCODING_BASIS_NUM * 2
        self.pos_num = POSITIONAL_ENCODING_BASIS_NUM
        self.linear1 = torch.nn.Linear(self.input_dim,N_NEURONS,bias=False)  # input = (x_alpha, alpha)
        self.linear2 = torch.nn.Linear(N_NEURONS, N_NEURONS,bias=False)  
        self.linear3 = torch.nn.Linear(N_NEURONS, N_NEURONS,bias=False)   
        self.linear4 = torch.nn.Linear(N_NEURONS, N_NEURONS,bias=False)   
        self.linear5 = torch.nn.Linear(N_NEURONS, N_NEURONS,bias=False)   
        #self.linear5 = torch.nn.Linear(N_NEURONS, N_NEURONS)
        self.output  = torch.nn.Linear(N_NEURONS, output_dim,bias=False) 
        self.relu = torch.nn.SiLU() 
    def forward(self, x, alpha,x_co):
        # print(x.dtype)
        # print(alpha.dtype)
        # print(self.linear1.weight.dtype)
        x_co = positional_encoding_1(x_co, num_encoding_functions=self.pos_num)
        res = torch.cat([x, alpha,x_co], dim=1)
        res = self.relu(self.linear1(res))
        res = self.relu(self.linear2(res))
        res = self.relu(self.linear3(res))
        res = self.relu(self.linear4(res))
        res = self.relu(self.linear5(res))
        res = self.output(res)
        return res


class NN_cond_pos_moresimpler(torch.nn.Module):
    def __init__(self,input_dim=3,output_dim=1,N_NEURONS=32,POSITIONAL_ENCODING_BASIS_NUM=5):
        super().__init__()
        self.input_dim = input_dim + 2 * POSITIONAL_ENCODING_BASIS_NUM * 2
        self.pos_num = POSITIONAL_ENCODING_BASIS_NUM
        self.linear1 = torch.nn.Linear(self.input_dim,N_NEURONS,bias=False)  # input = (x_alpha, alpha)
        self.linear2 = torch.nn.Linear(N_NEURONS, N_NEURONS,bias=False)  
        #self.linear5 = torch.nn.Linear(N_NEURONS, N_NEURONS)
        self.output  = torch.nn.Linear(N_NEURONS, output_dim,bias=False) 
        self.relu = torch.nn.SiLU() 
    def forward(self, x, alpha,x_co):
        # print(x.dtype)
        # print(alpha.dtype)
        # print(self.linear1.weight.dtype)
        x_co = positional_encoding_1(x_co, num_encoding_functions=self.pos_num)
        res = torch.cat([x, alpha,x_co], dim=1)
        res = self.relu(self.linear1(res))
        res = self.relu(self.linear2(res))
        #res = self.relu(self.linear5(res))
        res = self.output(res)
        return res


class NN_cond_pos_2(torch.nn.Module):
    def __init__(self,input_dim=3,output_dim=1):
        super().__init__()
        self.input_dim = input_dim + 2 * POSITIONAL_ENCODING_BASIS_NUM * 2 + 2 * POSITIONAL_ENCODING_BASIS_NUM * 3
        self.linear1 = torch.nn.Linear(self.input_dim,N_NEURONS)  # input = (x_alpha, alpha)
        self.linear2 = torch.nn.Linear(N_NEURONS, N_NEURONS)  
        self.linear3 = torch.nn.Linear(N_NEURONS, N_NEURONS)   
        self.linear4 = torch.nn.Linear(N_NEURONS, N_NEURONS)   
        #self.linear5 = torch.nn.Linear(N_NEURONS, N_NEURONS)
        self.output  = torch.nn.Linear(N_NEURONS, output_dim) 
        self.relu = torch.nn.SiLU()

    def forward(self, x, alpha,x_co):
        # print(x.dtype)
        # print(alpha.dtype)
        # print(self.linear1.weight.dtype)
        x_co = positional_encoding_1(x_co, num_encoding_functions=POSITIONAL_ENCODING_BASIS_NUM)
        x = positional_encoding_1(x, num_encoding_functions=POSITIONAL_ENCODING_BASIS_NUM)
        res = torch.cat([x, alpha,x_co], dim=1)
        res = self.relu(self.linear1(res))
        res = self.relu(self.linear2(res))
        res = self.relu(self.linear3(res))
        res = self.relu(self.linear4(res))
        #res = self.relu(self.linear5(res))
        res = self.output(res)
        return res

class NN_cond_complex(torch.nn.Module):
    def __init__(self,input_dim=3,output_dim=1,N_NEURONS=64):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim,N_NEURONS)  # input = (x_alpha, alpha)
        self.linear2 = torch.nn.Linear(N_NEURONS, N_NEURONS)  
        self.linear3 = torch.nn.Linear(N_NEURONS, N_NEURONS)   
        self.linear4 = torch.nn.Linear(N_NEURONS, N_NEURONS)  
        self.linear5 = torch.nn.Linear(N_NEURONS, N_NEURONS) 
        self.output  = torch.nn.Linear(N_NEURONS, output_dim) 
        self.relu = torch.nn.SiLU()

    def forward(self, x, alpha,x_co):
        # print(x.dtype)
        # print(alpha.dtype)
        # print(self.linear1.weight.dtype)
        res = torch.cat([x, alpha,x_co], dim=1)
        res = self.relu(self.linear1(res))
        res = self.relu(self.linear2(res))
        res = self.relu(self.linear3(res))
        res = self.relu(self.linear4(res))
        res = self.relu(self.linear5(res))
        res = self.output(res)
        return res

class NN_cond_simpler(torch.nn.Module):
    def __init__(self,input_dim=3,output_dim=1,N_NEURONS=32):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim,N_NEURONS)  # input = (x_alpha, alpha)
        self.linear2 = torch.nn.Linear(N_NEURONS, N_NEURONS)  
        self.output  = torch.nn.Linear(N_NEURONS, output_dim) 
        self.relu = torch.nn.SiLU()

    def forward(self, x, alpha,x_co):
        # print(x.dtype)
        # print(alpha.dtype)
        # print(self.linear1.weight.dtype)
        res = torch.cat([x, alpha,x_co], dim=1)
        res = self.relu(self.linear1(res))
        res = self.relu(self.linear2(res))
        res = self.output(res)
        return res

# simple Unet architecture
class Unet(torch.nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        # block down 1
        self.block1_conv1 = torch.nn.Conv2d( 2, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=1)
        self.block1_conv2 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=2)
        # block down 2
        self.block2_conv1 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=1)
        self.block2_conv2 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=2)
        # block down 3
        self.block3_conv1 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=1)
        self.block3_conv2 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=1)
        self.block3_conv3 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=1)
        self.block3_conv4 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=2)
        # block up 3
        self.block3_up1 = torch.nn.ConvTranspose2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=2, output_padding=1)
        self.block3_up2 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=1)
        # block up 2
        self.block2_up1 = torch.nn.ConvTranspose2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=2, output_padding=1)
        self.block2_up2 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=1)
        # block up 1
        self.block1_up1 = torch.nn.ConvTranspose2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=2, output_padding=1)
        self.block1_up2 = torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', stride=1)
        # output
        self.conv_output = torch.nn.Conv2d(64, 1, kernel_size=(1,1), padding=(0,0), padding_mode='zeros', stride=1)
        #
        self.relu = torch.nn.ReLU()

    def forward(self, x, alpha):

        b0 = torch.cat([x, alpha[:,None,None,None].repeat(1, 1, 32, 32)], dim=1)

        b1_c1 = self.relu(self.block1_conv1(b0))
        b1_c2 = self.relu(self.block1_conv2(b1_c1))

        b2_c1 = self.relu(self.block2_conv1(b1_c2))
        b2_c2 = self.relu(self.block2_conv2(b2_c1))

        b3_c1 = self.relu(self.block3_conv1(b2_c2))
        b3_c2 = self.relu(self.block3_conv2(b3_c1))
        b3_c3 = self.relu(self.block3_conv3(b3_c2)) + b3_c1
        b3_c4 = self.relu(self.block3_conv4(b3_c3))

        u2_c1 = self.relu(self.block3_up1(b3_c4)) + b3_c3
        u2_c2 = self.relu(self.block3_up2(u2_c1)) + b2_c2

        u1_c1 = self.relu(self.block2_up1(u2_c2)) + b1_c2
        u1_c2 = self.relu(self.block2_up2(u1_c1))

        u0_c1 = self.relu(self.block1_up1(u1_c2)) + b1_c1
        u0_c2 = self.relu(self.block1_up2(u0_c1))

        output = self.conv_output(u0_c2)

        return output