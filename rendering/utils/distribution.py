# Distribution with sampling method and pdf

import numpy as np
import scipy.stats as stats
import math

eps = 1e-5

class BaseDistribution():
    def __init__(self):
        pass
    def pdf(self,x):
        pass
    def sample(self,Ndata):
        pass

class Uniform(BaseDistribution):
    def __init__(self,Ndim,loc=0,scale=1):
        self.Ndim = Ndim
        self.loc = loc
        self.scale = scale
    def pdf(self,x):
        pdf_value = np.ones((x.shape[0],1))
        for i in range(self.Ndim):
            pdf_value *= stats.uniform.pdf(x[:,i],loc=self.loc,scale=self.scale).reshape(-1,1)
        return pdf_value
    def icdf(self, u):
        return u * self.scale + self.loc
    def sample(self,Ndata):
        Nline = math.ceil(Ndata**(1/self.Ndim))
        p_line = np.linspace(self.loc,self.loc+self.scale* (Nline-1)/Nline,Nline).astype("float32").reshape(Nline, 1)
        p_Ndim = np.meshgrid(*[p_line for i in range(self.Ndim)])
        p = np.concatenate([p.reshape(-1,1) for p in p_Ndim],axis=1)
        j = np.random.rand(p.shape[0], self.Ndim) * self.scale / Nline
        results = np.clip(p + j,self.loc,self.loc+self.scale)
        return results[np.random.choice(results.shape[0], Ndata, replace=False),:]

class Gaussian(BaseDistribution):
    # isotropic gaussian
    def __init__(self,Ndim,loc=0,scale=1,left = 0, right =1,offset = 0):
        self.Ndim = Ndim
        self.loc = loc
        self.scale = scale
        self.left = left
        self.right = right
        self.offset = offset
    def pdf(self,x):
        x = x - self.offset
        pdf_value = np.ones((x.shape[0],1))
        for i in range(self.Ndim):
            pdf_value *= 1 / (self.right - self.left) *stats.norm.pdf(x[:,i],loc=self.loc,scale=self.scale).reshape(-1,1)
        return pdf_value
    def icdf(self, u):
        return stats.norm.ppf(u, loc=self.loc, scale=self.scale)
    def sample(self,Ndata):
        Nline = math.ceil(Ndata**(1/self.Ndim))
        p_line = np.linspace(self.left,self.right * (Nline-1)/Nline,Nline).astype("float32").reshape(Nline, 1)
        p_Ndim = np.meshgrid(*[p_line for i in range(self.Ndim)])
        p = np.concatenate([p.reshape(-1,1) for p in p_Ndim],axis=1)
        j = np.random.rand(p.shape[0], self.Ndim)  / Nline * (self.right - self.left)
        unif_lattice = p + j
        unif_lattice = np.clip(unif_lattice,eps,1-eps)
        norm_samples = unif_lattice.flatten()
        norm_samples = stats.norm.ppf(norm_samples, loc=self.loc, scale=self.scale)
        norm_samples = norm_samples.reshape(unif_lattice.shape) + self.offset
        return norm_samples[np.random.choice(norm_samples.shape[0], Ndata, replace=False),:]

class TrunGaussian(BaseDistribution):
    def __init__(self,Ndim,clip_a=-1,clip_b=1,loc=0,scale=1,left = 0, right =1):
        self.Ndim = Ndim
        self.a = (clip_a - loc) / scale
        self.b = (clip_b - loc) / scale
        self.loc = loc
        self.scale = scale
        self.left = left
        self.right = right
    def pdf(self,x):
        pdf_value = np.ones((x.shape[0],1))
        for i in range(self.Ndim):
            pdf_value *= 1 / (self.right - self.left) * stats.truncnorm.pdf(x[:,i],self.a,self.b,loc = self.loc, scale= self.scale).reshape(-1,1)
        return pdf_value
    def icdf(self, u):
        return stats.truncnorm.ppf(u, self.a, self.b,loc = self.loc, scale= self.scale)
    def sample(self,Ndata):
        Nline = math.ceil(Ndata**(1/self.Ndim))
        p_line = np.linspace(self.left,self.right * (Nline-1)/Nline,Nline).astype("float32").reshape(Nline, 1)
        p_Ndim = np.meshgrid(*[p_line for i in range(self.Ndim)])
        p = np.concatenate([p.reshape(-1,1) for p in p_Ndim],axis=1)
        j = np.random.rand(p.shape[0], self.Ndim) / Nline
        unif_lattice = p + j
        unif_lattice = np.clip(unif_lattice,0,1)
        norm_samples = unif_lattice.flatten()
        norm_samples = stats.truncnorm.ppf(norm_samples, self.a, self.b,loc = self.loc, scale= self.scale)
        norm_samples = norm_samples.reshape(unif_lattice.shape)
        return norm_samples[np.random.choice(norm_samples.shape[0], Ndata, replace=False),:]

class Beta(BaseDistribution):
    def __init__(self,Ndim,alpha=2,beta=2,loc=0,scale=1):
        self.Ndim = Ndim
        self.alpha = alpha
        self.beta = beta
        self.loc = loc
        self.scale = scale
    def pdf(self,x):
        pdf_value = np.ones((x.shape[0],1))
        for i in range(self.Ndim):
            pdf_value *= stats.beta.pdf(x[:,i],self.alpha,self.beta,loc = self.loc,scale = self.scale).reshape(-1,1)
        return pdf_value
    def icdf(self, u):
        return stats.beta.ppf(u, self.alpha, self.beta,loc = self.loc,scale = self.scale)
    def sample(self,Ndata):
        Nline = math.ceil(Ndata**(1/self.Ndim))
        p_line = np.linspace(0,1* (Nline-1)/Nline,Nline).astype("float32").reshape(Nline, 1)
        p_Ndim = np.meshgrid(*[p_line for i in range(self.Ndim)])
        p = np.concatenate([p.reshape(-1,1) for p in p_Ndim],axis=1)
        j = np.random.rand(p.shape[0], self.Ndim) / Nline
        unif_lattice = p + j
        unif_lattice = np.clip(unif_lattice,0,1)
        beta_samples = unif_lattice.flatten()
        beta_samples = stats.beta.ppf(beta_samples, self.alpha, self.beta,loc = self.loc,scale = self.scale)
        beta_samples = beta_samples.reshape(unif_lattice.shape)
        #beta_samples[np.isinf(beta_samples)] = 0
        return beta_samples[np.random.choice(beta_samples.shape[0], Ndata, replace=False),:]

class TwoDCombination(BaseDistribution):
    def __init__(self,dist1,dist2):
        self.dist1 = dist1
        self.dist2 = dist2
    def pdf(self,x):
        pdf1 = self.dist1.pdf(x[:,0].reshape(-1,1))
        pdf2 = self.dist2.pdf(x[:,1].reshape(-1,1))
        return pdf1 * pdf2
    def sample(self,Ndata):
        Nline = math.ceil(Ndata**(1/2))
        p_line = np.linspace(0,1* (Nline-1)/Nline,Nline).astype("float32").reshape(Nline, 1)
        unif_lattice = np.meshgrid(*[p_line for i in range(2)])
        unif_lattice = np.concatenate([p.reshape(-1,1) for p in unif_lattice],axis=1)
        j = np.random.rand(unif_lattice.shape[0], 2) / Nline
        unif_lattice = unif_lattice + j
        def clip(x,dist):
            if isinstance(dist,Gaussian):
                return np.clip(x,eps,1-eps)
            else:
                return np.clip(x,0,1)
        unif_lattice[:,0] = clip(unif_lattice[:,0],self.dist1)
        unif_lattice[:,1] = clip(unif_lattice[:,1],self.dist1)

        comb_samples = np.ones_like(unif_lattice)
        comb_samples[:,0] = self.dist1.icdf(unif_lattice[:,0])
        comb_samples[:,1] = self.dist2.icdf(unif_lattice[:,1])
        return comb_samples[np.random.choice(comb_samples.shape[0], Ndata, replace=False),:]
    def sample_parallel(self,Ndata,batchsize):
        Nline = math.ceil(Ndata**(1/2))
        p_line = np.linspace(0,1* (Nline-1)/Nline,Nline).astype("float32").reshape(Nline, 1)
        unif_lattice = np.meshgrid(*[p_line for i in range(2)])
        unif_lattice = np.concatenate([p.reshape(-1,1) for p in unif_lattice],axis=1)
        unif_lattice = unif_lattice.repeat(batchsize,axis=0)
        j = np.random.rand(unif_lattice.shape[0], 2) / Nline
        unif_lattice = unif_lattice + j
        def clip(x,dist):
            if isinstance(dist,Gaussian):
                return np.clip(x,eps,1-eps)
            else:
                return np.clip(x,0,1)
        unif_lattice[:,0] = clip(unif_lattice[:,0],self.dist1)
        unif_lattice[:,1] = clip(unif_lattice[:,1],self.dist1)

        comb_samples = np.ones_like(unif_lattice)
        comb_samples[:,0] = self.dist1.icdf(unif_lattice[:,0])
        comb_samples[:,1] = self.dist2.icdf(unif_lattice[:,1])
        return comb_samples[np.random.choice(comb_samples.shape[0], Ndata*batchsize, replace=False),:]

class StraightLine(BaseDistribution):
    def __init__(self,h,flag = 1,offset =0):
        self.h = h
        self.b = flag * 2 / h
        self.flag = flag
        self.k = - h / self.b
        self.offset = offset
    def pdf(self,x):
        x = x - self.offset
        return np.where(x<np.min([0,self.b]),0,np.where(x>np.max([0,self.b]),0,self.k * x + self.h))
    def sample(self,Ndata):
        p = np.linspace(0,1,Ndata).astype("float32").reshape(Ndata, 1)
        j = np.random.rand(p.shape[0], 1) / Ndata
        unif_lattice = p + j
        unif_lattice = np.clip(unif_lattice,0,1)
        unif_lattice = unif_lattice.flatten()
        def cdf(u):
            if self.flag == -1 :
                return self.b * (1 - np.sqrt(u))
            elif self.flag == 1 :
                return self.b * (1 - np.sqrt(1 - u))
        x_samples = cdf(unif_lattice) + self.offset
        x_samples = x_samples.reshape(-1,1)
        
        return x_samples

class CustomDistribution(BaseDistribution):
    def __init__(self,Ndim,pdf_func,sample_func):
        self.Ndim = Ndim
        self.pdf_func = pdf_func
        self.sample_func = sample_func
    def pdf(self,x):
        return self.pdf_func(x)
    def sample(self,Ndata):
        return self.sample_func(Ndata)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    
    Ndata = 2**4
    dist1 = Beta(1,5,5,loc = -0.5,scale =2 * np.pi)
    dist2 = Uniform(1,-  2 * np.pi, 4 * np.pi)
    dist = TwoDCombination(dist1,dist2)
    dist_gt = Beta(2,5,5,loc = -0.5,scale =2 * np.pi)
    all_x = []
    all_x_gt = []
    for i in range(2**12):
        x = dist.sample(Ndata)
        x_gt = dist_gt.sample(Ndata)
        all_x.append(x)
        all_x_gt.append(x_gt)
        #print(f"loop {i} done")
    x = np.concatenate(all_x,0)
    x_gt = np.concatenate(all_x_gt,0)
    plt.figure(figsize=(6,6))
    plt.hist2d(x[:,0],x[:,1],bins=300,density=True)
    plt.title("learned")
    plt.figure(figsize=(6,6))
    plt.hist2d(x_gt[:,0],x_gt[:,1],bins=300,density=True)
    plt.title("gt")
    plt.show()
    
    
    