import numpy as np
import torch
from neural_ensemblers.model import NeuralEnsembler
from neural_ensemblers.trainer import Trainer
from neural_ensemblers.trainer_args import TrainerArgs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input = torch.rand(( 100, 5, 15)).to(device) 
input = input / input.sum(dim=1, keepdim=True) # Normalize the input
weights = torch.rand((1, 1, 15)).to(device)
weights = weights / weights.sum(dim=2, keepdim=True) # Normalize the weights
target= (input * weights).sum(dim=2).argmax(dim=1).to(device) # Example target

num_samples,  num_classes, num_base_functions= input.shape
model = NeuralEnsembler(num_base_functions=num_base_functions,
                        num_classes=num_classes,
                        hidden_dim=512,
                        num_layers=3,
                        mode="model_averaging",
                        ).to(device)


x, w= model(input)

trainer_args = TrainerArgs(task_type="classification", batch_size=512, lr=0.0001, epochs=2000)
trainer = Trainer(model=model, trainer_args=trainer_args)
trainer.fit(input, target)

print("x", x)
print("w", w)
print("target", target)
print(model(input)[0].argmax(dim=1))
print("accuracy", (model(input)[0].argmax(dim=1) == target).float().mean())