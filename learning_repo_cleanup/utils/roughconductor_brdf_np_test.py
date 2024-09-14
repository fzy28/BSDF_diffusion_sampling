# This is some custom brdf, like classic GGX.
import numpy as np
import matplotlib.pyplot as plt

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
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    vectors = np.stack([x,y,z],axis=1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norm_vector = vectors / norms
    return norm_vector
def spher_to_cart0(theta,phi):
    x = np.sqrt(1 - theta**2) * np.cos(phi)
    y = np.sqrt(1 - theta**2) * np.sin(phi)
    z = theta
    vectors = np.stack([x,y,z],axis=1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norm_vector = vectors / norms
    return norm_vector
def classic_shading_pdf(omega_i,omega_o,roughness,F0=0.04, diffuse_prob=0):
    # copy from unreal engine's presentation
    # hnpps://de45xmedrsdbp.cloudfront.net/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf

    # omega_i and omega_o are spherical coordinates with theta and phi
    
        
    
    light_dir = spher_to_cart(omega_i[:,0],omega_i[:,1])
    view_dir = spher_to_cart(omega_o[:,0],omega_o[:,1])
    
    #Diffuse
    f_diffuse = 1 / np.pi
    #Specular
    
    normal = np.array([0,0,1]) # standard normal
    normal = normal.reshape(1,3)
    sum_dir = light_dir + view_dir
    norms = np.linalg.norm(sum_dir, axis=1, keepdims=True)
    half_vector = sum_dir / norms

    NdotH = np.einsum('ij,ij->i', normal, half_vector)  # 法线与半向量的点积
    NdotL = np.einsum('ij,ij->i', normal, light_dir)  # 法线与光源方向的点积
    NdotV = np.einsum('ij,ij->i', normal, view_dir)  # 法线与观察方向的点积
    VdotH = np.einsum('ij,ij->i', view_dir, half_vector)  # 观察方向与半向量的点积
    
    
    # NDF GGX classic!
    
    D = NDF_GGX(NdotH,roughness)
    
    # Geometry
    
    G = G_SmithSchlick_GGX(NdotL,NdotV,roughness)
    
    # Fresnel
    
    # Their fresnel is too complex, I use Schlick's approximation
    
    F = fresnel_schlick(VdotH,F0)
    
    f_specular = (D * G * F) / (4 * NdotL * NdotV + 1e-10)
    
    cosine_term = np.maximum(0,NdotV)
    
    result = (1-diffuse_prob) * f_specular * cosine_term   + diffuse_prob * f_diffuse * cosine_term
        
    
    return np.where(
    (omega_o[:,0] < np.pi/2) & (omega_o[:,0] > 0) & (omega_o[:,1] > -np.pi) & (omega_o[:,1] < np.pi),
    result,
    0
)

def classic_shading_sampling(albedo,omega_i,omega_o,normal,roughness,F0=0.04):
    
    stan_model_code = """
    
    """

if __name__ == "__main__":
    
    roughness = 0.5
    
    theta_l = np.linspace(0, np.pi / 2 , 150)
    phi_l = np.linspace(-np.pi,     np.pi, 150)

    theta, phi = np.meshgrid(theta_l, phi_l)
    tmp = theta.shape

    omega_i = np.array([np.pi/1.9,np.pi/4]).reshape(-1,2)
    omega_i = omega_i.reshape(-1,2)
    theta = theta.reshape(-1)

    phi = phi.reshape(-1)
    omega_o = np.stack([theta,phi],axis=1)
    omega_o = omega_o.reshape(-1,2)
    brdf = classic_shading_pdf(omega_i,omega_o,0.5) 
    brdf = brdf.reshape(tmp)
    theta = theta.reshape(tmp)
    phi = phi.reshape(tmp)

    plt.figure(figsize=(6, 6))
    plt.pcolormesh(theta, phi, brdf)
    plt.xlabel('theta_o')
    plt.ylabel('phi_o')
    def pi_formatter(x, pos):
        return f"{x/np.pi:.2f}π"
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(pi_formatter))
    # plt.gca().xaxis.set_major_locator(MultipleLocator(base=0.05*np.pi))
    # plt.gca().yaxis.set_major_locator(MultipleLocator(base=0.1*np.pi))
    eps = 1e-5
    #plt.xlim(0, np.pi / 2 + eps)
    #plt.ylim(0, np.pi + eps)
    plt.title('2D conditioned GGX BRDF')
    plt.colorbar()


    # omega_i = omega_i.reshape(-1,2)
    # theta = theta.reshape(-1)
    # phi = phi.reshape(-1)
    # omega_o = np.stack([theta,phi],axis=1)
    # omega_o = omega_o.reshape(-1,2)
    # brdf = classic_shading_pdf(omega_i,omega_o,roughness)
    # brdf = brdf.reshape(tmp)
    # theta = theta.reshape(tmp)
    # phi = phi.reshape(tmp)

    # plt.figure(figsize=(6, 6))
    # plt.pcolormesh(theta, phi, brdf)
    # plt.xlabel('theta_o')
    # plt.ylabel('phi_o')
    # def pi_formatter(x, pos):
    #     return f"{x/np.pi:.2f}π"
    # from matplotlib.ticker import MultipleLocator, FuncFormatter
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(pi_formatter))
    # plt.gca().xaxis.set_major_locator(MultipleLocator(base=0.05*np.pi))
    # plt.gca().yaxis.set_major_locator(MultipleLocator(base=0.1*np.pi))
    # eps = 1e-5
    # #plt.xlim(0, np.pi / 2 + eps)
    # #plt.ylim(0, np.pi + eps)
    # plt.title('2D conditioned GGX BRDF')
    plt.show()




    