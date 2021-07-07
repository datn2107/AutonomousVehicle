import os
import sys
#sys.path.append(os.path.dirname(os.path.basename(__file__)))

import torch

# from .models_pytorch import MultiBoxLoss


def train_one_epoch(model, optimizer, dataset, device, loss=None, print_freq=1000):
    model.train()
    for batch, (images, targets) in enumerate(dataset):
        print(batch)
        images = torch.stack(images, dim=0).to(device)
        # images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]


        # Compute prediction and loss
        prediction = model(images)
        print(prediction)
        return
        # total_loss = sum(loss for loss in loss_dict.values())
        #
        # # Backpropagation
        # optimizer.zero_grad()
        # total_loss.backward()
        # optimizer.step()
        #
        # if batch % print_freq == 0:
        #     loss = loss_dict.item()
        #     print(f"loss: {loss:>7f}")