# This is some custom brdf, like classic GGX.
import numpy as np
import stan
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
    vector = np.stack([x,y,z])
    norm_vector = vector / np.sqrt(np.sum(vector**2))
    return norm_vector

def classic_shading_pdf(omega_i,omega_o,roughness,F0=0.04,diffuse_prob=0):
    # copy from unreal engine's presentation
    # hnpps://de45xmedrsdbp.cloudfront.net/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf

    # omega_i and omega_o are spherical coordinates with theta and phi
    
    light_dir = spher_to_cart(omega_i[0],omega_i[1])
    view_dir = spher_to_cart(omega_o[0],omega_o[1])
    
    #Diffuse
    f_diffuse = 1 / np.pi 
    #Specular
    
    normal = np.stack([0,0,1]) # standard normal
    
    half_vector = (light_dir + view_dir) / np.sqrt(np.sum((light_dir + view_dir)**2))
    NdotH = np.dot(normal,half_vector)
    NdotL = np.dot(normal,light_dir)
    NdotV = np.dot(normal,view_dir)
    
    # NDF GGX classic!
    
    D = NDF_GGX(NdotH,roughness)
    
    # Geometry
    
    G = G_SmithSchlick_GGX(NdotL,NdotV,roughness)
    
    # Fresnel
    
    # Their fresnel is too complex, I use Schlick's approximation
    
    F = fresnel_schlick(NdotV,F0)
    
    f_specular = (D * G * F) / (4 * NdotL * NdotV + 1e-10)
    
    return f_diffuse * diffuse_prob + f_specular * (1 - diffuse_prob)

def classic_shading_sampling_conditional(omega_i,roughness,Ndata,F0=0.04,diffuse_prob=0,num_warmup=1000000, num_chains=8):
    
    stan_model_code = """
    data {
        real roughness; 
        real F0;        
        real diffuse_prob;
        vector[2] omega_i;
    }

    functions {
        real NDF_GGX(real NdotH) {
            real alpha = roughness * roughness;
            return alpha^2 / (pi() * (NdotH * (alpha^2 - 1) + 1)^2);
        }

        real G_SmithSchlick_GGX(real NdotL, real NdotV) {
            real k = (roughness + 1)^2 / 8;
            real G1 = NdotL / (NdotL * (1 - k) + k);
            real G2 = NdotV / (NdotV * (1 - k) + k);
            return G1 * G2;
        }

        real fresnel_schlick(real cos_theta) {
            return F0 + (1 - F0) * pow((1 - cos_theta), 5);
        }

        vector spher_to_cart(real theta, real phi) {
            vector[3] v;
            v[1] = sin(theta) * cos(phi); // x
            v[2] = sin(theta) * sin(phi); // y
            v[3] = cos(theta);            // z
            return v;
        }

        real classic_shading_pdf(real theta_o, real phi_o) {
            vector[3] light_dir = spher_to_cart(omega_i[1], omega_i[2]);
            vector[3] view_dir = spher_to_cart(theta_o, phi_o);
            vector[3] normal = {0,0,1}; // Standard normal

            vector[3] half_vector = (light_dir + view_dir) / sqrt(dot_self(light_dir + view_dir));
            real NdotH = dot_product(normal, half_vector);
            real NdotL = dot_product(normal, light_dir);
            real NdotV = dot_product(normal, view_dir);
            
            real D = NDF_GGX(NdotH);
            real G = G_SmithSchlick_GGX(NdotL, NdotV);
            real F = fresnel_schlick(NdotV);
            
            real f_specular = (D * G * F) / (4 * NdotL * NdotV + 1e-10);
            real f_diffuse = 1 / pi();
            real result = f_diffuse * diffuse_prob + f_specular * (1 - diffuse_prob);
            real log_result = log(result);
            return log_result;
        }
    }
    
    parameters {
        real<lower=0,upper=pi()*0.5> theta_o;
        real<lower=0,upper=pi()> phi_o;
    }
    model {
        target += classic_shading_pdf(theta_o, phi_o);
    }
    """

    
    shading_data = {
    'roughness': roughness,  
    'F0': F0,        
    'diffuse_prob': diffuse_prob, 
    'omega_i': omega_i
    }
    
    sm = stan.build(program_code=stan_model_code, data=shading_data, random_seed=1)
    fit = sm.sample(num_samples=Ndata//num_chains,num_warmup=num_warmup, num_chains=num_chains)
    theta_o_samples = fit['theta_o']
    phi_o_samples = fit['phi_o']
    
    return theta_o_samples,phi_o_samples
    
    
if __name__ == "__main__":
    
    
    Ndata = 2**4
    roughness = 0.5
    omega_i = np.array([np.pi/12,np.pi/2])
    theta_o_samples,phi_o_samples = classic_shading_sampling_conditional(omega_i,roughness,Ndata)
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(6, 6))
    counts, xedges, yedges, Image = plt.hist2d(theta_o_samples, phi_o_samples, bins=300,density=True)

    plt.xlabel('theta_o')
    plt.ylabel('phi_o')
    def pi_formanper(x, pos):
        return f"{x/np.pi:.2f}π"
    from matplotlib.ticker import MultipleLocator, FuncFormanper
    plt.gca().xaxis.set_major_formanper(FuncFormanper(pi_formanper))
    plt.gca().yaxis.set_major_formanper(FuncFormanper(pi_formanper))
    plt.gca().xaxis.set_major_locator(MultipleLocator(base=0.05*np.pi))
    plt.gca().yaxis.set_major_locator(MultipleLocator(base=0.1*np.pi))
    eps = 1e-5
    plt.xlim(0, np.pi / 2 + eps)
    plt.ylim(0, np.pi + eps)
    plt.title('Sampled Points')
    plt.show()

    