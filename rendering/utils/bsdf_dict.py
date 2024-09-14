import mitsuba as mi

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
def principle_bsdf(dict_principled):
    bsdf = mi.load_dict(
        {
            "type": "principled",
            'base_color': {
            'type': 'rgb',
            'value': [1.0, 1.0, 1.0]
            },
            "metallic": dict_principled['metallic'],
            "specular": dict_principled['specular'],
            "roughness": dict_principled['roughness'],
            "spec_tint": dict_principled['spec_tint'],
            "anisotropic": dict_principled['anisotropic'],
            "sheen": dict_principled['sheen'],
            "sheen_tint": dict_principled['sheen_tint'],
            "clearcoat": dict_principled['clearcoat'],
            "clearcoat_gloss": dict_principled['clearcoat_gloss'],
            "spec_trans": dict_principled['spec_trans'] ,
            "flatness": dict_principled['flatness']
        }
    )
    return bsdf
bsdf_materials.append(principle_bsdf(dict1_principled))
bsdf_materials.append(principle_bsdf(dict2_principled))
bsdf_materials.append(principle_bsdf(dict3_principled))
bsdf_materials.append(principle_bsdf(dict4_principled))
bsdf_materials.append(principle_bsdf(dict5_principled))
bsdf_materials.append(principle_bsdf(dict6_principled))
bsdf_materials.append(principle_bsdf(dict7_principled))
bsdf_materials.append(principle_bsdf(dict8_principled))
bsdf_materials.append(principle_bsdf(dict9_principled))


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
bsdf_materials.append(principle_bsdf(dict10_principled))
bsdf_materials.append(principle_bsdf(dict11_principled))
bsdf_materials.append(principle_bsdf(dict12_principled))
bsdf_materials.append(principle_bsdf(dict13_principled))
bsdf_materials.append(principle_bsdf(dict14_principled))
bsdf_materials.append(principle_bsdf(dict15_principled))
bsdf_materials.append(principle_bsdf(dict16_principled))
bsdf_materials.append(principle_bsdf(dict17_principled))
bsdf_materials.append(principle_bsdf(dict18_principled))
bsdf_materials.append(principle_bsdf(dict19_principled))
bsdf_materials.append(principle_bsdf(dict20_principled))
bsdf_materials.append(principle_bsdf(dict21_principled))
bsdf_materials.append(principle_bsdf(dict22_principled))
bsdf_materials.append(principle_bsdf(dict23_principled))

bsdf_materials.append(mi.load_dict(
            {
                "type": "roughdielectric",
                "distribution": "beckmann",
                "alpha": 0.2,
                "int_ior": "bk7",
                "ext_ior": "air",
                
            }
        ))
bsdf_materials.append(mi.load_dict(
            {
                "type": "roughdielectric",
                "distribution": "beckmann",
                "alpha": 0.3,
                "int_ior": "bk7",
                "ext_ior": "air",
                
            }
        ))
bsdf_materials.append(mi.load_dict(
            {
                "type": "roughdielectric",
                "distribution": "beckmann",
                "alpha": 0.5,
                "int_ior": "bk7",
                "ext_ior": "air",
                
            }
        ))