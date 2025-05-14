import numpy as np
import torch
from neural_ensemblers.model import NeuralEnsembler

input = torch.rand(( 5, 3, 2)) # Example input
print(input)
num_samples,  num_classes, num_base_functions= input.shape
model = NeuralEnsembler(num_base_functions=num_base_functions,
                        num_classes=num_classes)

x, w= model(input)

print(x.shape)
print(w.shape)  # Check the output values

print("x", x)
print("w", w)