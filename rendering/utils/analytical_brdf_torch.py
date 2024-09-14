import torch


def cart_to_spher(xyz):
    r = torch.norm(xyz, dim=1)
    theta = torch.acos(xyz[:,2]/r) 
    phi = torch.atan2(xyz[:,1], xyz[:,0])
    return torch.stack([theta, phi], dim=1)
def cart_to_spher0(xyz):
    r = torch.norm(xyz, dim=1)
    costheta = (xyz[:,2]/r) 
    phi = torch.atan2(xyz[:,1], xyz[:,0])
    return torch.stack([costheta, phi], dim=1)
def NDF_GGX(NdotH, roughness):
    alpha = roughness ** 2
    return alpha ** 2 / (torch.pi * (NdotH * (alpha ** 2 - 1) + 1) ** 2)

def G_SmithSchlick_GGX(NdotL, NdotV, roughness):
    k = (roughness + 1) ** 2 / 8
    G1 = NdotL / (NdotL * (1 - k) + k)
    G2 = NdotV / (NdotV * (1 - k) + k)
    return G1 * G2

def fresnel_schlick(cos_theta, F0):
    return F0 + (1 - F0) * (1 - cos_theta) ** 5

def spher_to_cart(wo):
    theta = wo[:, 0]
    phi = wo[:, 1]
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    vectors = torch.stack([x, y, z], dim=1)
    norms = torch.norm(vectors, dim=1, keepdim=True)
    norm_vector = vectors / norms
    return norm_vector
def spher0_to_cart(wo):
    costheta = wo[:, 0]
    phi = wo[:, 1]
    x = torch.sqrt(1 - costheta ** 2) * torch.cos(phi)
    y = torch.sin(1 - costheta ** 2) * torch.sin(phi)
    z = costheta
    vectors = torch.stack([x, y, z], dim=1)
    norms = torch.norm(vectors, dim=1, keepdim=True)
    norm_vector = vectors / norms
    return norm_vector


def classic_shading_pdf(omega_i, omega_o, roughness, F0=0.04, diffuse_prob=0):
    light_dir = spher_to_cart(omega_i)
    view_dir = spher_to_cart(omega_o)

    normal = torch.tensor([0, 0, 1], device=light_dir.device).float().reshape(1, 3)
    sum_dir = light_dir + view_dir
    norms = torch.norm(sum_dir, dim=1, keepdim=True)
    half_vector = sum_dir / norms

    NdotH = torch.sum(normal * half_vector, dim=1)
    NdotL = torch.sum(normal * light_dir, dim=1)
    NdotV = torch.sum(normal * view_dir, dim=1)
    VdotH = torch.sum(view_dir * half_vector, dim=1)

    D = NDF_GGX(NdotH, roughness)
    G = G_SmithSchlick_GGX(NdotL, NdotV, roughness)
    F = fresnel_schlick(VdotH, F0)

    f_specular = (D * G * F) / (4 * NdotL * NdotV + 1e-10)
    f_specular = torch.nan_to_num(f_specular,nan = 0, posinf = 0, neginf = 0.0)
    cosine_term = torch.clamp(NdotV, min=0)

    return (1 - diffuse_prob) * f_specular * cosine_term + diffuse_prob * (1 / torch.pi) * cosine_term

