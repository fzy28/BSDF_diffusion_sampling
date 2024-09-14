
import drjit as dr
import mitsuba as mi

mi.set_variant('scalar_rgb')
from utils.analytical_brdf_torch import *

def rgb2lum(rgb):
    lum = 0.2126*rgb.x + 0.7152*rgb.y + 0.0722*rgb.z
    return lum

def disk_to_dir(wi,wo):
    si = dr.zeros(mi.SurfaceInteraction3f)
    wi = disk_to_cart(wi).cpu()
    si.wi = mi.Vector3f(wi[...,0], wi[...,1], wi[...,2])
    wo = disk_to_cart(wo).cpu()
    wo = wo
    wo = mi.Vector3f(wo[...,0], wo[...,1], wo[...,2])
    return si,wo
def spheric_to_dir(wi,wo):
    si = dr.zeros(mi.SurfaceInteraction3f)
    wi = spher_to_cart(wi[...,0],wi[...,1]).cpu().float()
    wo = spher_to_cart(wo[...,0],wo[...,1]).cpu().float()
    si.wi = mi.Vector3f(wi[...,0], wi[...,1], wi[...,2])
    wo = mi.Vector3f(wo[...,0], wo[...,1], wo[...,2])
    return si,wo
class roughconductor():
    def __init__(self, material, alpha_u, alpha_v, distribution = 'ggx',is_spherical = False):
        self.bsdf = mi.load_dict({
            'type': 'roughconductor',
            'material': material,
            'alpha_u': alpha_u,
            'alpha_v': alpha_v,
            'distribution': distribution
        })
        self.torch_to_dr = spheric_to_dir if is_spherical else disk_to_dir
        self.is_spherical = is_spherical

    def eval(self,wi,wo):
        si,wo = self.torch_to_dr(wi,wo)
        values = self.bsdf.eval(mi.BSDFContext(), si, wo) 
        values = rgb2lum(values)
        inv_cos = dr.clamp(1.0 / wo.z, 1.0, 1e6)
        values = values * dr.sqrt(1 - wo.z**2)  if self.is_spherical else values * inv_cos        
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
        self.torch_to_dr = spheric_to_dir 

    def eval(self,wi,wo):
        si,wo = self.torch_to_dr(wi,wo)
        values = self.bsdf.eval(mi.BSDFContext(), si, wo) 
        values = rgb2lum(values)
        inv_cos = dr.clamp(1.0 / wo.z, 1.0, 1e6)
        values = values * dr.sqrt(1 - wo.z**2)         
        return values
    
class meaturedbsdf():
    def __init__(self, filename,is_spherical = False):
        self.bsdf = mi.load_dict({
            'type': 'measured',
            'filename': filename
        })
        self.torch_to_dr = spheric_to_dir if is_spherical else disk_to_dir
        self.is_spherical = is_spherical
    def eval(self,wi,wo):
        si,wo = self.torch_to_dr(wi,wo)   
        values = self.bsdf.eval(mi.BSDFContext(), si, wo)
        values = rgb2lum(values) 
        inv_cos = dr.clamp(1.0 / wo.z, 1.0, 1e6)
        values = values * dr.sqrt(1 - wo.z**2)  if self.is_spherical else values * inv_cos          
        return values
class roughdielectric():
    def __init__(self, alpha, int_ior, ext_ior, distribution = 'beckmann'):
        self.bsdf = mi.load_dict({
            'type': 'roughdielectric',
            'distribution': distribution,
            'alpha': alpha,
            'int_ior': int_ior,
            'ext_ior': ext_ior
        })
        self.torch_to_dr = spheric_to_dir 

    def eval(self,wi,wo):
        si,wo = self.torch_to_dr(wi,wo)
        values = self.bsdf.eval(mi.BSDFContext(), si, wo) 
        values = rgb2lum(values)
        inv_cos = dr.clamp(1.0 / wo.z, 1.0, 1e6)
        values = values * dr.sqrt(1 - wo.z**2)               
        return values

# def twodisk_to_cart(wo):
#     tmpwo = wo[:,0].clone()
#     wo[:,0] = torch.where(wo[:,0] < 0,wo[:,0] + 1,wo[:,0] - 1)
#     rr = wo[...,:2].pow(2).sum(-1)
#     z = torch.where(tmpwo < 0,(1-rr).relu().sqrt(),-((1-rr).relu().sqrt()))
#     wo = torch.cat([wo,z.unsqueeze(-1)],-1)
#     return wo    
# def twodisk_to_dr(wi,wo):
#     si = dr.zeros(mi.SurfaceInteraction3f)
#     wi = twodisk_to_cart(wi).cpu()
#     si.wi = mi.Vector3f(wi[...,0], wi[...,1], wi[...,2])
#     wo = twodisk_to_cart(wo).cpu()
#     wo = mi.Vector3f(wo[...,0], wo[...,1], wo[...,2])
#     return si,wo

# class principle_bsdf():
#     def __init__(self, dict):
#         self.bsdf = mi.load_dict({
#             'type': 'principled',
#             'base_color': {
#             'type': 'rgb',
#             'value': [1.0, 1.0, 1.0]
#             },
#             'metallic': dict['metallic'],
#             'specular': dict['specular'],
#             'roughness': dict['roughness'],
#             'spec_tint':   dict['spec_tint'],
#             'anisotropic': dict['anisotropic'],
#             'sheen': dict['sheen'],
#             'sheen_tint': dict['sheen_tint'],
#             'clearcoat': dict['clearcoat'],
#             'clearcoat_gloss': dict['clearcoat_gloss'],
#             'spec_trans': dict['spec_trans']
#             })
#     def eval(self,wi,wo):
#         si,wo = twodisk_to_dr(wi,wo)
#         values = self.bsdf.eval(mi.BSDFContext(), si, wo) 
#         values = rgb2lum(values)
#         return values
# class roughdielectric():
#     def __init__(self, alpha, int_ior = "bk7", ext_ior="air", distribution = 'beckmann'):
#         self.bsdf = mi.load_dict({
#             'type': 'roughdielectric',
#             'distribution': distribution,
#             'alpha': alpha,
#             'int_ior': int_ior,
#             'ext_ior': ext_ior
#         })
#     def eval(self,wi,wo):
#         si,wo = twodisk_to_dr(wi,wo)
#         values = self.bsdf.eval(mi.BSDFContext(), si, wo) 
#         values = rgb2lum(values) 
#         return values          
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