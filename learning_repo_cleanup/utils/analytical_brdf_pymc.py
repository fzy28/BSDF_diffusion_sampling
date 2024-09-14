# This is some custom brdf, like classic GGX.

import numpy as np
import theano.tensor as tt
def NDF_GGX(NdotH, roughness):
    alpha = roughness * roughness
    return alpha**2 / (np.pi * (NdotH * (alpha**2 - 1) + 1)**2)
def G_SmithSchlick_GGX(NdotL, NdotV, roughness):
    k = (roughness + 1)**2 / 8
    G1 = NdotL / (NdotL * (1 - k) + k)
    G2 = NdotV / (NdotV * (1 - k) + k)
    return G1 * G2
def fresnel_schlick(cos_theta, F0):
    return F0 + (1 - F0) * (1 - cos_theta)**5
def spher_to_cart(theta,phi):
    x = tt.sin(theta) * tt.cos(phi)
    y = tt.sin(theta) * tt.sin(phi)
    z = tt.cos(theta)
    vector = tt.stack([x,y,z])
    norm_vector = vector / tt.sqrt(tt.sum(vector**2))
    return norm_vector
def disk_to_cart(wo):
    rr = tt.sum(wo[:2]**2)
    z = tt.sqrt(tt.clip(1 - rr, 0, 1))
    wo_cart = tt.concatenate([wo[:2], z.reshape((1,))], axis=0)
    return wo_cart
def classic_shading_pdf(omega_i,omega_o,roughness,F0=0.04,diffuse_prob=0):
    # copy from unreal engine's presentation
    # https://de45xmedrsdbp.cloudfront.net/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf

    # omega_i and omega_o are spherical coordinates with theta and phi
    
    out_of_roughness_range = tt.or_(
        tt.lt(roughness, 0),
        tt.gt(roughness, 1)
    )
    
    def check_out_of_range(omega, eps=1e-10):
        conditions = [
            tt.lt(omega[0], -1),
            tt.gt(omega[0], 1 ),
            tt.lt(omega[1], -1),
            tt.gt(omega[1], 1)
        ]
        # Use tt.any to check if any condition in the list is True
        out_of_range = tt.any(conditions)
        return out_of_range
    
    out_of_range = tt.any(
        [check_out_of_range(omega_i),
        check_out_of_range(omega_o),
        out_of_roughness_range]
    )
    light_dir = disk_to_cart(omega_i)
    view_dir = disk_to_cart(omega_o)
    
    #Diffuse
    f_diffuse = 1 / np.pi 
    #Specular
    
    normal = tt.stack([0,0,1]) # standard normal
    
    half_vector = (light_dir + view_dir) / tt.sqrt(tt.sum((light_dir + view_dir)**2))
    NdotH = tt.dot(normal,half_vector)
    NdotL = tt.dot(normal,light_dir)
    NdotV = tt.dot(normal,view_dir)
    HdotV = tt.dot(half_vector,view_dir)
    # NDF GGX classic!
    
    D = NDF_GGX(NdotH,roughness)
    
    # Geometry
    
    G = G_SmithSchlick_GGX(NdotL,NdotV,roughness)
    
    # Fresnel
    
    # Their fresnel is too complex, I use Schlick's approximation
    
    F = fresnel_schlick(HdotV,F0)
    
    f_specular = (D * G * F) / (4 * NdotL * NdotV + 1e-10)
    costerm = NdotV
    result = f_diffuse * diffuse_prob + f_specular * (1 - diffuse_prob)
    return tt.switch(out_of_range, 0, result * costerm) 

def classic_shading_sampling(omega_i,omega_o,normal,roughness,F0=0.04):
    
    stan_model_code = """
    
    """

if __name__ == "__main__":
    Ndata = 2**16 // 8
    omega_i = np.array([-0.7,-0.7])

    with pm.Model() as model:
        X = pm.DensityDist('X', lambda value: pm.math.log((classic_shading_pdf(omega_i,value,0.5))), shape=(2,))
        trace = pm.sample(Ndata,step=pm.Metropolis(target_accept=0.85), tune=5000, chains=8, cores=8)
    
    x_fromnpy = trace['X']
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.hist2d(x_fromnpy[:,0],x_fromnpy[:,1],bins=100)
    plt.show()
    