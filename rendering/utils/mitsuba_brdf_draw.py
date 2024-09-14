import drjit as dr
import mitsuba as mi
import torch
mi.set_variant('cuda_ad_rgb')
import OpenEXR
import Imath
import numpy as np

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

def rgb2lum(rgb):
    lum = 0.2126*rgb.x + 0.7152*rgb.y + 0.0722*rgb.z
    return lum

def disk_to_cart(wo):
    rr = wo[...,:2].pow(2).sum(-1)
    wo = torch.cat([wo,(1-rr).relu().sqrt().unsqueeze(-1)],-1)
    return wo
def twodisk_to_cart(wo):
    tmpwo = wo[:,0].clone()
    wo[:,0] = torch.where(wo[:,0] < 0,wo[:,0] + 1,wo[:,0] - 1)
    rr = wo[...,:2].pow(2).sum(-1)
    z = torch.where(tmpwo < 0,(1-rr).relu().sqrt(),-((1-rr).relu().sqrt()))
    wo = torch.cat([wo,z.unsqueeze(-1)],-1)
    return wo
def cart_to_twodisk(wo):
    wo[:,0] = torch.where(wo[:,2] < 0,wo[:,0] + 1,wo[:,0] - 1)
    return wo[:,0:2]
def onedisk_to_dr(wi,wo):
    si = dr.zeros(mi.SurfaceInteraction3f)
    wi = disk_to_cart(wi).cuda().float()
    si.wi = mi.Vector3f(wi[...,0], wi[...,1], wi[...,2])
    wo = disk_to_cart(wo).cuda().float()
    wo = mi.Vector3f(wo[...,0], wo[...,1], wo[...,2])

    return si,wo
def twodisk_to_dr(wi,wo):
    si = dr.zeros(mi.SurfaceInteraction3f)
    wi = twodisk_to_cart(wi).cuda().float()
    si.wi = mi.Vector3f(wi[...,0], wi[...,1], wi[...,2])
    wo = twodisk_to_cart(wo).cuda().float()
    wo = mi.Vector3f(wo[...,0], wo[...,1], wo[...,2])
    return si,wo
class roughconductor():
    def __init__(self, material, alpha_u, alpha_v, distribution = 'ggx'):
        self.bsdf = mi.load_dict({
            'type': 'roughconductor',
            'material': material,
            'alpha_u': alpha_u,
            'alpha_v': alpha_v,
            'distribution': distribution
        })
    def eval(self,wi,wo):
        si,wo = onedisk_to_dr(wi,wo)
        values = self.bsdf.eval(mi.BSDFContext(), si, wo) 
        values = rgb2lum(values)
        return values
class principle_bsdf():
    def __init__(self, dict):
        self.bsdf = mi.load_dict({
            'type': 'principled',
            'base_color': {
            'type': 'rgb',
            'value': [1.0, 1.0, 1.0]
            },
            'metallic': dict['metallic'],
            'specular': dict['specular'],
            'roughness': dict['roughness'],
            'spec_tint':   dict['spec_tint'],
            'anisotropic': dict['anisotropic'],
            'sheen': dict['sheen'],
            'sheen_tint': dict['sheen_tint'],
            'clearcoat': dict['clearcoat'],
            'clearcoat_gloss': dict['clearcoat_gloss'],
            'spec_trans': dict['spec_trans']
            })
    def eval(self,wi,wo):
        si,wo = twodisk_to_dr(wi,wo)
        values = self.bsdf.eval(mi.BSDFContext(), si, wo) 
        values = dr.select(wo.z > 0,values * wo.z, values)
        values = rgb2lum(values)
        return values
    
class meaturedbsdf():
    def __init__(self, filename):
        self.bsdf = mi.load_dict({
            'type': 'measured',
            'filename': filename
        })
    def eval(self,wi,wo):
        si,wo = onedisk_to_dr(wi,wo)
        values = self.bsdf.eval(mi.BSDFContext(), si, wo)
        values = rgb2lum(values) 
        return values
class roughdielectric():
    def __init__(self, alpha, int_ior = "bk7", ext_ior="air", distribution = 'beckmann'):
        self.bsdf = mi.load_dict({
            'type': 'roughdielectric',
            'distribution': distribution,
            'alpha': alpha,
            'int_ior': int_ior,
            'ext_ior': ext_ior
        })
    def eval(self,wi,wo):
        si,wo = twodisk_to_dr(wi,wo)
        values = self.bsdf.eval(mi.BSDFContext(), si, wo) 
        values = rgb2lum(values) 
        return values
          
if __name__ == '__main__':
    
    omegai = [-0.9,0.1]

    dict = {
        'metallic': 0.7,
'specular': 0.6,
'roughness': 0.2,
'spec_tint': 0.4,
'anisotropic': 0.5,
'sheen': 0.3,
'sheen_tint': 0.2,
'clearcoat': 0.6,
'clearcoat_gloss': 0.3,
'spec_trans': 0.4
    }
    mybsdf = principle_bsdf(dict)
    condi_omega_i = torch.tensor(omegai).reshape(1,2)
    res = 400
    theta_o, phi_o = torch.meshgrid(
        torch.linspace(-1, 1, res),
        torch.linspace(-1, 1, res)
    )
    wo = torch.stack([theta_o, phi_o], dim=-1).reshape(-1, 2)
    condi_omega_i = condi_omega_i.repeat_interleave(res*res,0)


    values = mybsdf.eval(condi_omega_i,wo)

    import numpy as np
    values_np = np.array(values)

    import matplotlib.pyplot as plt
    plt.imshow((values_np[...,0]).reshape(res,res))
    plt.colorbar()
    plt.show()