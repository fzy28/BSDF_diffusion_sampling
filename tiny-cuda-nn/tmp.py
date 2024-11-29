import torch
import tinycudann as tcnn
from torch import nn

batch_size = 128
features_in = 25
features_out = 2
hidden_dim = 32

# torch net, notice that there should be no bias, because there is no in the tcnn implementation
mlp_torch = nn.Sequential(
    nn.Linear(features_in, hidden_dim, bias=False),
    nn.SiLU(),
    nn.Linear(hidden_dim, hidden_dim, bias=False),
    nn.SiLU(),
    nn.Linear(hidden_dim, features_out, bias=False),
).cuda()

# same net, but in tcnn, should be faster on big batches
mlp_tcnn = tcnn.Network(
    n_input_dims=features_in,
    n_output_dims=features_out,
    network_config={
        "otype": "FullyfusedMLP",
        "activation": "Silu",
        "output_activation": "None",
        "n_neurons": hidden_dim,
        "n_hidden_layers": 2,
    }
)

input = torch.randn(batch_size, features_in).cuda()

# but the initialization is obviously different
output_torch = mlp_torch(input)
output_tcnn = mlp_tcnn(input)
print(torch.allclose(output_torch, output_tcnn.float(), rtol=0.01, atol=0.01)) # False

# in tcnn output layer's width always should be the multiple of 16, so need to pad last layer's params here
output_layer = mlp_torch[4].weight.data
output_layer = nn.functional.pad(output_layer,
                                 pad=(0, 0, 0, 16 - (features_out % 16)))
input_layer = mlp_torch[0].weight.data
input_layer = nn.functional.pad(input_layer,
                                 pad=(0, 16 - (features_in % 16), 0, 0))
# concatenate all flatten parameters
params = torch.cat([
    input_layer.flatten(),
    mlp_torch[2].weight.data.flatten(),
    output_layer.flatten()
]).half()

# assign their values to the tcnn net
mlp_tcnn.params.data[...] = params

# now both nets are the same (but there could be little differences due to half float usage)
output_torch = mlp_torch(input)
output_tcnn = mlp_tcnn(input)
print(torch.allclose(output_torch, output_tcnn.float(), rtol=0.01, atol=0.01)) # True