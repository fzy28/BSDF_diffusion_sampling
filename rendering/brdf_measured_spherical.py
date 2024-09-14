# This file contains the implementation of a BSDF that is based on a measured BRDF.
# Used for mitsuba renderer.

import mitsuba as mi
import drjit as dr
from tqdm import tqdm
import torch
from utils.model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os
import sys
p = os.path.abspath('.')
sys.path.insert(1, p)
from utils.mitsuba_brdf_draw import *
from utils.analytical_brdf_torch import *
from utils.mlp_brdf_sampling import *

torch.set_default_dtype(torch.float32)
mi.set_variant("cuda_ad_rgb")
dr.set_flag(dr.JitFlag.VCallRecord, False)
dr.set_flag(dr.JitFlag.LoopRecord, False)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scene_file", type=str, default="scene_measured.xml")

parser = parser.parse_args()

def sph_to_dir(theta, phi):
    st, ct = dr.sincos(theta)
    sp, cp = dr.sincos(phi)
    return mi.Vector3f(cp * st, sp * st, ct)
def cart_to_spher(xyz):
    r = torch.norm(xyz, dim=1)
    theta = torch.acos(xyz[:,2]/(r+1e-8))
    phi = torch.atan2(xyz[:,1], xyz[:,0])
    return torch.stack([theta, phi], dim=1)
class MyBSDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        material = props["filename"]
        self.bsdf = meaturedbsdf(os.path.join("measuredbsdfs",material+".bsdf"))
        self.bsdf = mi.load_dict(
            {
                "type":"measured",
                "filename":"./measuredbsdfs/"+material+".bsdf"

            }
        )
        self.D_sample = NN_cond_pos(input_dim=6,output_dim=2,N_NEURONS=32,POSITIONAL_ENCODING_BASIS_NUM=5).to("cuda")

        self.D_sample.load_state_dict(torch.load("./checkpoints_new/" + material+"_spherical/"+"brdf_rectify_network" + material+".pth"))
        self.D_sample.eval()
        
        
        self.D_base = NN_cond_pretrain_spherical_one(input_dim=2,N_NEURONS=16).to("cuda")
        self.D_base.load_state_dict(torch.load("./checkpoints_new/" + material+"_disk/"+"brdf_pretrain_network" + material+".pth"))

        #self.D_base.load_state_dict(torch.load("./checkpoints/checkpoints/aniso_miro_7_rgb/brdf_pretrainaniso_miro_7_rgb.pth"))
        
        
        self.albedo = mi.Color3f([1, 1,1]) 
        reflection_flags = mi.BSDFFlags.DeltaReflection | mi.BSDFFlags.FrontSide
        self.m_components = [reflection_flags]
        self.m_flags = reflection_flags

    def sample(self, ctx, si, sample1, sample2, active=True):
        # Compute Fresnel terms

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        active &= cos_theta_i > 0

        wi = si.wi.torch()
        wi_input = cart_to_spher(wi)
        wo,pdf = network_sampling_spherical(self.D_base,self.D_sample,wi_input)
        pdf = torch.where(torch.sin(wo[:,0]) > 0.00005, pdf, torch.zeros_like(pdf))
        pdf = torch.where(torch.cos(wo[:,0]) > 0, pdf, torch.zeros_like(pdf))
        wo = mi.Vector2f(wo[...,0], wo[...,1])
        wo = sph_to_dir(wo.x, wo.y)
        
        bs = mi.BSDFSample3f()
        
        bs.wo = wo           
        cos_theta_o = mi.Frame3f.cos_theta(bs.wo)
        
        floatmax = mi.Float(np.array([np.finfo(np.float32).max]))
        invsin_theta_o =dr.clamp(1/ (mi.Frame3f.sin_theta(bs.wo)) ,1,floatmax)
        bs.pdf = mi.Float(pdf) * invsin_theta_o
        # bs.wo = mi.warp.square_to_cosine_hebrdf_onlyphere(sample2)
        # bs.pdf = mi.warp.square_to_cosine_hebrdf_onlyphere_pdf(bs.wo)
        bs.eta = 1.0
        bs.sampled_type = mi.UInt32(+self.m_flags)
        bs.sampled_component = 0
        
        wi_input = si.wi.torch()[...,:2]
        wo_tmp = bs.wo.torch()[...,:2]
        #brdf = self.bsdf.eval(wi_input, wo_tmp)
        brdf = self.bsdf.eval(ctx, si, bs.wo)
        value = brdf / bs.pdf  * self.albedo 
        #value = brdf  * self.albedo / mi.Float(pdf) * mi.Frame3f.sin_theta(bs.wo)

        value = dr.select(active & (bs.pdf > 0.0), value, mi.Vector3f(0))
        value_torch = rgb2lum(value).torch()
        pdf = torch.where(value_torch<30, pdf, torch.zeros_like(pdf))
        bs.pdf = mi.Float(pdf) * invsin_theta_o
        return (bs, dr.select(active & (bs.pdf > 0.0) & (cos_theta_o > 0), value, mi.Vector3f(0)))

    def eval(self, ctx, si, wo, active=True):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        
        brdf = self.bsdf.eval(ctx, si, wo)
        value =  brdf * self.albedo 
        return dr.select(
            (cos_theta_i > 0.0) & (cos_theta_o > 0.0), value, mi.Vector3f(0)
        )

    def pdf(self, ctx, si, wo, active=True):

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        floatmax = mi.Float(np.array([np.finfo(np.float32).max]))
        
        invsin_theta_o =dr.clamp(1/ (mi.Frame3f.sin_theta(wo)) ,1,floatmax)
        wi = si.wi.torch()
        wi_input = cart_to_spher(wi)
        wo = wo.torch()
        wo_input = cart_to_spher(wo)
        pdf = network_pdf_spherical(self.D_base,self.D_sample,wo_input,wi_input) 
        pdf = torch.where(torch.sin(wo_input[:,0]) > 0.00005, pdf, torch.zeros_like(pdf))
        return dr.select(
            (cos_theta_i > 0.0) & (cos_theta_o > 0.0), mi.Float(pdf)* invsin_theta_o, mi.Float(0)
        )

    def eval_pdf(self, ctx, si, wo, active=True):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

    def to_string(self):
        return "MyBSDF[\n" "    albedo=%s,\n" "]" % (self.albedo)


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    import time

    start_time = time.time()

    mi.register_bsdf("mybsdf", lambda props: MyBSDF(props))

    
    scene_path = os.path.join("matpreview", parser.scene_file)
    scene = mi.load_file(scene_path)
    
    SPP = 4
    spp = SPP * 128

    seed = 0
    
    image = mi.render(scene, spp=SPP, seed=seed).numpy()
    print(image.shape)
    for _ in tqdm(range(spp // SPP)):
        image += mi.render(scene, spp=SPP, seed=seed).numpy()
        seed += 1
    image /= (spp // SPP) + 1

    filepath = os.path.join("diffusion_brdf_measured_spherical", f"{parser.scene_file}.png")
    mi.util.write_bitmap(filepath, image)
    
    filepath_exr = os.path.join("diffusion_brdf_measured_spherical", f"{parser.scene_file}.exr")
    mi.util.write_bitmap(filepath_exr, image)
    end_time = time.time()
    print("Render time: " + str(end_time - start_time) + " seconds")
    
