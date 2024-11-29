## Tiny-CUDA-NN

We modified Tiny-CUDA-NN to support the SiLU activation.

We use Tiny-CUDA-NN for much faster reflow step, since we need to do hundreds times of inference of a small network. Using Tiny-CUDA-NN can speed up the time by 5x-10x, really thanks for the contributors of this amazing projects!

To install just go:

```
tiny-cuda-nn$ cd bindings/torch
tiny-cuda-nn/bindings/torch$ python setup.py install
```

To validate the correctness, you may run the tmp.py, and see whether the output is False and True.