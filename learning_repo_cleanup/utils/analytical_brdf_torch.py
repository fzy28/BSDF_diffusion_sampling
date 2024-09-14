import torch

def NDF_GGX(NdotH, roughness):
    alpha = roughness ** 2
    return alpha ** 2 / (torch.pi * (NdotH * (alpha ** 2 - 1) + 1) ** 2)
def cart_to_spher(xyz):
    r = torch.norm(xyz, dim=1)
    theta = torch.acos(xyz[:,2]/r) 
    phi = torch.atan2(xyz[:,1], xyz[:,0])
    return torch.stack([theta, phi], dim=1)
def G_SmithSchlick_GGX(NdotL, NdotV, roughness):
    k = (roughness + 1) ** 2 / 8
    G1 = NdotL / (NdotL * (1 - k) + k)
    G2 = NdotV / (NdotV * (1 - k) + k)
    return G1 * G2

def fresnel_schlick(cos_theta, F0):
    return F0 + (1 - F0) * (1 - cos_theta) ** 5

def spher_to_cart(theta, phi):
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    vectors = torch.stack([x, y, z], dim=1)
    norms = torch.norm(vectors, dim=1, keepdim=True)
    norm_vector = vectors / norms
    return norm_vector


def disk_to_cart(wo):
    rr = wo[...,:2].pow(2).sum(-1)
    wo = torch.cat([wo,(1-rr).relu().sqrt().unsqueeze(-1)],-1)
    return wo
def classic_shading_pdf_disk(omega_i, omega_o, roughness, F0=0.04, diffuse_prob=0):
    light_dir = disk_to_cart(omega_i)
    view_dir = disk_to_cart(omega_o)

    normal = torch.tensor([0, 0, 1], device=light_dir.device,dtype=light_dir.dtype).reshape(1, 3)
    sum_dir = light_dir + view_dir
    norms = torch.norm(sum_dir, dim=1, keepdim=True)
    half_vector = sum_dir / norms

    NdotH = torch.einsum('ij,ij->i', normal, half_vector)  # 法线与半向量的点积
    NdotL = torch.einsum('ij,ij->i', normal, light_dir)    # 法线与光源方向的点积
    NdotV = torch.einsum('ij,ij->i', normal, view_dir)     # 法线与观察方向的点积
    VdotH = torch.einsum('ij,ij->i', view_dir, half_vector) # 观察方向与半向量的点积

    D = NDF_GGX(NdotH, roughness)
    G = G_SmithSchlick_GGX(NdotL, NdotV, roughness)
    F = fresnel_schlick(VdotH, F0)

    f_specular = (D * G * F) / (4 * NdotL * NdotV + 1e-10)
    cosine_term = torch.clamp(NdotV, min=0)

    return (1 - diffuse_prob) * f_specular * cosine_term + diffuse_prob * (1 / torch.pi) * cosine_term

def classic_shading_pdf_spherical(omega_i, omega_o, roughness, F0=0.04, diffuse_prob=0):
    light_dir = spher_to_cart(omega_i[:, 0], omega_i[:, 1])
    view_dir = spher_to_cart(omega_o[:, 0], omega_o[:, 1])

    normal = torch.tensor([0, 0, 1], device=light_dir.device).float().reshape(1, 3)
    sum_dir = light_dir + view_dir
    norms = torch.norm(sum_dir, dim=1, keepdim=True)
    half_vector = sum_dir / norms

    NdotH = torch.einsum('ij,ij->i', normal, half_vector)  # 法线与半向量的点积
    NdotL = torch.einsum('ij,ij->i', normal, light_dir)    # 法线与光源方向的点积
    NdotV = torch.einsum('ij,ij->i', normal, view_dir)     # 法线与观察方向的点积
    VdotH = torch.einsum('ij,ij->i', view_dir, half_vector) # 观察方向与半向量的点积

    D = NDF_GGX(NdotH, roughness)
    G = G_SmithSchlick_GGX(NdotL, NdotV, roughness)
    F = fresnel_schlick(VdotH, F0)

    f_specular = (D * G * F) / (4 * NdotL * NdotV + 1e-10)
    cosine_term = torch.clamp(NdotV, min=0)

    return (1 - diffuse_prob) * f_specular * cosine_term + diffuse_prob * (1 / torch.pi) * cosine_term

