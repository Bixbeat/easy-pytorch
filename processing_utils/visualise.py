# import visdom
import torch
from torchvision.transforms import Normalize, ToPILImage
import numpy as np
from PIL import Image, ImageOps
import datetime as dt
import time as t
import matplotlib as mpl
import matplotlib.pyplot as plt

from .analysis_utils import var_to_cpu

def plot_pairs(image, label):
    """Takes an image tensor and its reconstruction vars, and
    argmaxed softmax predictions to create a 1x2 comparison plot."""
    _, (imgplot, lblplot) = plt.subplots(1, 2, figsize=(12, 6))
    imgplot.imshow(image, aspect='auto')
    imgplot.grid(color='r', linestyle='dashed', alpha=0.75)

    lblplot.imshow(label, aspect='auto')
    lblplot.grid(color='r', linestyle='dashed', alpha=0.75)                

    plt.show(block=False)

def encoded_img_and_lbl_to_data(image, predictions, means, sdevs, label_colours):
    """For a given image and label pair, reconstruct both into
    images"""
    predictions = predictions.cpu()
    
    coloured_label = colour_lbl(predictions, label_colours)
    restored_img = decode_image(image, means, sdevs)
    return restored_img, coloured_label


def decode_image(tensor, mean, sdev):
    """For a given normalized image tensor, reconstructs
    the image by undoing the normalization and transforming
    the tensor to an image"""
    out_imgs = torch.FloatTensor()
    for i, img in enumerate(tensor):    
        transposed_tensor = img.permute((1, 2, 0)).cpu()
        unnormed_img = torch.Tensor(sdev) * transposed_tensor + torch.Tensor(mean)
        image = torch.clamp(unnormed_img, 0, 1)
        image = image.unsqueeze(0) # Add dim to cat along
        
        out_imgs = torch.cat((out_imgs, image))

    return out_imgs   
    
def colour_lbl(tensor, colours):
    """For a given RGB image, constructs a RGB image map
    using the defined image classes.
    Adapted to pure PT from SRC: https://github.com/bfortuner/pytorch_tiramisu/blob/master/tiramisu-pytorch.ipynb
    """
    out_imgs = torch.FloatTensor()
    for i, img in enumerate(tensor): 
        red = img.clone()
        green = img.clone()
        blue = img.clone()

        for i, key in enumerate(colours):
            # Colour along each dimension using axis-specific dict values
            red[img==i] = colours[key][0]
            green[img==i] = colours[key][1]
            blue[img==i] = colours[key][2]

        rgb = torch.zeros((img.shape[0], img.shape[1], 3))

        rgb[:,:,0] = (red/255.0)
        rgb[:,:,1] = (green/255.0)
        rgb[:,:,2] = (blue/255.0)
        rgb = rgb.unsqueeze(0) # Add dim to cat along

        out_imgs = torch.cat((out_imgs, rgb))

    return out_imgs

def _remove_microseconds(time_delta):
    return time_delta - dt.timedelta(microseconds=time_delta.microseconds)

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, input_img):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        features = input_img

        for module_pos, module in self.model._modules.items():
            if module_pos == 'fc':
                features = features.view(features.size(0), -1)            
            features = module(features)  # Forward
            if module_pos == self.target_layer:
                features.register_hook(self.save_gradient)
                conv_output = features  # Save the output on target layer

        model_output = features
        return conv_output, model_output

    def forward_pass(self, input_img):
        conv_output, model_output = self.forward_pass_on_convolutions(input_img)
        return conv_output, model_output

class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer, colorramp='inferno'):
        self.colorramp = colorramp
        self.model = model.eval()
        self.target_layer = target_layer
        self.extractor = CamExtractor(self.model, target_layer)       

        self.predicted_class = None
        self.layer_required_grad = None

    def generate_cam(self, input_image, means, sdevs, out_img_size=224, unnormalize=True, cam_transparency=0.5, target_class=None):
        self._enable_gradient()
        conv_output, model_output = self.extractor.forward_pass(input_image)
        self.predicted_class = np.argmax(model_output.data.numpy())
        if target_class is None:
            target_class = self.predicted_class
        self.model.zero_grad()

        original_img = var_to_cpu(input_image).data[0]
        if unnormalize:
            original_img = normalized_img_tensor_to_pil(original_img, means, sdevs)

        cam = self.results_to_cam(conv_output, model_output, target_class)
        cam_heatmap = self.cam_array_to_heatmap(cam, out_img_size)
        cam_overlaid = Image.blend(cam_heatmap, original_img, cam_transparency)

        self._disable_grad_if_required()
        return cam_overlaid

    def results_to_cam(self, conv_output, model_output, target_class):
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1 # Backprop target

        model_output.backward(gradient=one_hot_output, retain_graph=True)
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        target_conv = conv_output.data.numpy()[0]
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        
        cam = np.ones(target_conv.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target_conv[i, :, :] # Multiply weights with conv output and sum
        return cam

    def cam_array_to_heatmap(self, cam_array, out_img_size):
        cam_normalized = (cam_array - np.min(cam_array)) / (np.max(cam_array) - np.min(cam_array))
        cam_coloured = self.colourize_gradient(cam_normalized)[:, :, :3]
        cam_img = Image.fromarray(np.uint8(cam_coloured*255))
        cam_resized = ImageOps.fit(cam_img, (out_img_size, out_img_size))
        return cam_resized

    def create_gradcam_img(self, target_img, img_class, means, sdevs, input_size):
        cam_input_img = var_to_cpu(target_img)
        used_cuda = None
        if next(self.model.parameters()).is_cuda: #Most compact way to check if model is in cuda
            self.model = self.model.cpu()
            used_cuda = True

        cam_img = self.generate_cam(cam_input_img, means, sdevs, input_size, target_class=img_class)

        if used_cuda:
            self.model = self.model.cuda()
        return cam_img      

    def _enable_gradient(self):
        # Enable gradient to layers so that it can be hooked
        # method does NOT perform optimization
        layer = self.model._modules.get(self.target_layer)
        for param in layer.parameters():
            self.layer_required_grad = param.requires_grad
            param.requires_grad = True

    def _disable_grad_if_required(self):
        layer = self.model._modules.get(self.target_layer)
        if self.layer_required_grad is False:
            for param in layer.parameters():
                param.requires_grad = False   

    def colourize_gradient(self, img_array):
        colour = mpl.cm.get_cmap(self.colorramp)
        coloured_img = colour(img_array)
        return coloured_img

def normalized_img_tensor_to_pil(img_tensor, means, sdevs):
    bands = len(means)
    to_pil = ToPILImage()
    inverse_normalize = Normalize(
        mean =[-means[band]/sdevs[band] for band in range(bands)],
        std=[1/sdevs[band] for band in range(bands)]
    )
    inverse_tensor = inverse_normalize(img_tensor)      
    return to_pil(inverse_tensor)