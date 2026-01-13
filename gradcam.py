import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None

        if target_layer is None:
            target_layer = self._default_target_layer()

        layer = dict(self.model.named_modules()).get(target_layer, None)
        if layer is None:
            raise ValueError(f"Layer '{target_layer}' not found in the model.")

        layer.register_forward_hook(self._forward_hook)
        layer.register_full_backward_hook(self._backward_hook)

    def _default_target_layer(self):
        # set default target layers for Grad-CAM
        name = self.model.__class__.__name__.lower()
        if "vgg" in name:
            return "features.28"   # Last conv layer in VGG16
        elif "resnet" in name:
            return "layer4"        # Last conv block in ResNet
        else:
            raise NotImplementedError(f"Automatic target layer selection not implemented for model {name}")

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_cam(self, input_tensor, class_idx):
        # forward pass
        output = self.model(input_tensor)
        self.model.zero_grad() #make sure we aren't accumulating

        # backward pass wrt target class
        print("GradCAM output shape:", output.shape)
        print("class_idx:", class_idx)
        loss = output[0, class_idx]
        loss.backward()

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3]) 

        # weight channels by activations
        activations = self.activations[0]
        for i in range(pooled_grads.size(0)):
            activations[i, :, :] *= pooled_grads[i]

        #produce the heatmap overlay 
        heatmap = torch.sum(activations, dim=0)
        heatmap = torch.relu(heatmap)

        
        heatmap -= heatmap.min()
        if heatmap.max() != 0:
            heatmap /= heatmap.max()

        
        heatmap_np = heatmap.cpu().numpy()
        heatmap_np = cv2.resize(heatmap_np, (input_tensor.size(3), input_tensor.size(2))) #fit heatmap to original image

        return heatmap_np