from utils.mitsuba_brdf_scalar import principle_bsdf as principle_bsdf_scalar
from utils.mitsuba_brdf_scalar import roughdielectric as roughdielectric_scalar

bsdf_materials = []

dict1_principled = {
    'metallic': 0.1,
'specular': 1.0,
'roughness': 0.2,
'spec_tint': 0.5,
'anisotropic': 0.5,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict2_principled = {
    'metallic': 0.3,
'specular': 0.7,
'roughness': 0.5,
'spec_tint': 0.5,
'anisotropic': 0.5,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict3_principled = {
    'metallic': 1.0,
'specular': 0.8,
'roughness': 0.1,
'spec_tint': 0.5,
'anisotropic': 0.5,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict4_principled = {
    'metallic': 0.1,
'specular': 0.9,
'roughness': 0.1,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict4_principled = {
    'metallic': 0.2,
'specular': 0.3,
'roughness': 0.3,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict5_principled = {
    'metallic': 0.1,
'specular': 0.8,
'roughness': 0.3,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict6_principled = {
    'metallic': 0.1,
'specular': 1.0,
'roughness': 0.1,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict7_principled = {
    'metallic': 0.9,
'specular': 0.7,
'roughness': 0.3,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict8_principled = {
    'metallic': 0.5,
'specular': 0.8,
'roughness': 0.3,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.3,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict9_principled = {
    'metallic': 0.1,
'specular': 0.8,
'roughness': 0.3,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}

dict10_principled = {
    'metallic': 0.3,
'specular': 0.2,
'roughness': 0.1,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict11_principled = {
    'metallic': 0.0,
'specular': 1.0,
'roughness': 0.1,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict12_principled = {
    'metallic': 0.8,
'specular': 0.2,
'roughness': 0.1,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict13_principled = {
    'metallic': 0.6,
'specular': 0.2,
'roughness': 0.3,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict14_principled = {
    'metallic': 0.3,
'specular': 0.2,
'roughness': 0.7,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict15_principled = {
    'metallic': 0.9,
'specular': 0.2,
'roughness': 0.5,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict16_principled = {
    'metallic': 0.9,
'specular': 0.2,
'roughness': 0.3,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict17_principled = {
    'metallic': 0.9,
'specular': 0.2,
'roughness': 0.6,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict18_principled = {
    'metallic': 0.9,
'specular': 0.2,
'roughness': 0.9,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict19_principled = {
    'metallic': 0.1,
'specular': 0.8,
'roughness': 0.1,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict20_principled = {
    'metallic': 0.1,
'specular': 0.5,
'roughness': 0.4,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict21_principled = {
    'metallic': 0.1,
'specular': 0.8,
'roughness': 0.3,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict22_principled = {
    'metallic': 0.1,
'specular': 0.5,
'roughness': 0.7,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
dict23_principled = {
    'metallic': 0.1,
'specular': 0.3,
'roughness': 0.8,
'spec_tint': 0.5,
'anisotropic': 0.7,
'sheen': 0.5,
'sheen_tint': 0.5,
'clearcoat': 0.5,
'clearcoat_gloss': 0.5,
'spec_trans': 0.9,
'flatness':1.0,
}
bsdf_materials.append(principle_bsdf_scalar(dict1_principled))
bsdf_materials.append(principle_bsdf_scalar(dict2_principled))
bsdf_materials.append(principle_bsdf_scalar(dict3_principled))
bsdf_materials.append(principle_bsdf_scalar(dict4_principled))
bsdf_materials.append(principle_bsdf_scalar(dict5_principled))
bsdf_materials.append(principle_bsdf_scalar(dict6_principled))
bsdf_materials.append(principle_bsdf_scalar(dict7_principled))
bsdf_materials.append(principle_bsdf_scalar(dict8_principled))
bsdf_materials.append(principle_bsdf_scalar(dict9_principled))
bsdf_materials.append(principle_bsdf_scalar(dict10_principled))
bsdf_materials.append(principle_bsdf_scalar(dict11_principled))
bsdf_materials.append(principle_bsdf_scalar(dict12_principled))
bsdf_materials.append(principle_bsdf_scalar(dict13_principled))
bsdf_materials.append(principle_bsdf_scalar(dict14_principled))
bsdf_materials.append(principle_bsdf_scalar(dict15_principled))
bsdf_materials.append(principle_bsdf_scalar(dict16_principled))
bsdf_materials.append(principle_bsdf_scalar(dict17_principled))
bsdf_materials.append(principle_bsdf_scalar(dict18_principled))
bsdf_materials.append(principle_bsdf_scalar(dict19_principled))
bsdf_materials.append(principle_bsdf_scalar(dict20_principled))
bsdf_materials.append(principle_bsdf_scalar(dict21_principled))
bsdf_materials.append(principle_bsdf_scalar(dict22_principled)) 
bsdf_materials.append(principle_bsdf_scalar(dict23_principled))
bsdf_materials.append(roughdielectric_scalar(0.2, int_ior = "bk7", ext_ior="air"))
bsdf_materials.append(roughdielectric_scalar(0.3, int_ior = "bk7", ext_ior="air"))
bsdf_materials.append(roughdielectric_scalar(0.5, int_ior = "bk7", ext_ior="air"))

