import random

import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision.transforms import Normalize, ToPILImage
import numpy as np
from PIL import Image, ImageOps, ImageDraw
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

def draw_rectangles(image, bboxes, annotations, probabilities, gt=None):
    out_img = ImageDraw.Draw(image)
    for i,bbox in enumerate(bboxes):
        random_color = "#"+''.join(random.choice('0123456789ABCDEF'))
        out_img.Draw.rectangle(bbox, fill=None, outline=random_color)
        
        out_text = f"""{annotations[i]}\nP={probabilities[i]}:.1f"""
        if gt:
            out_text+=f"""\ntrue={gt}""" 
        top_left = bbox[0]
        out_img.Draw.multiline_text(top_left, out_text, fill=None, font='DejaVuSansMono.ttf')
    return out_img

def encoded_img_and_lbl_to_data(image, predictions, means, sdevs, label_colours):
    """For a given image and label pair, reconstruct both into
    images"""
    predictions = predictions.cpu()
    
    coloured_label = colour_lbl(predictions, label_colours)
    restored_img = decode_image(image, means, sdevs)
    return restored_img, coloured_label

def decode_image(in_img, mean, sdev):
    """For a given normalized image tensor, reconstructs
    the image by undoing the normalization"""
    transposed_tensor = in_img.permute((1, 2, 0)).cpu()
    unnormed_img = torch.Tensor(sdev) * transposed_tensor + torch.Tensor(mean)
    out_img = torch.clamp(unnormed_img, 0, 1)

    return out_img
    
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

def get_cam_img(in_img_tensor, model, target_layer_name, out_size=(224,224), target_class=None):
    """For a given (unbatched) image,
    generates a class attention map image from the target layer
    
    Adapted from https://github.com/metalbubble/CAM
    
    Arguments:
        in_img_tensor {Tensor} -- Unnormalized input image
        model {PyTorch model} -- Model which to get the CAM for
        target_layer_name {str} -- Name of the target convolutional layer
    
    Keyword Arguments:
        out_size {tuple} -- Output size after resizing (default: {(224,224)})
        target_class {int} -- Target class index (default: {None})
    """
    out_cam = []
    def hook_feature(module, image, output):
        out_cam.extend(output.data.cpu().numpy())
      
    model._modules.get(target_layer_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    img_variable = Variable(in_img_tensor.unsqueeze(0))
    if torch.cuda.is_available():
        img_variable.cuda()
        
    logit = model(img_variable)

    predictions = F.softmax(logit, dim=1).data.cpu().squeeze()
    _, idx = predictions.sort(0, True)
    idx = idx.numpy()

    if target_class:
        pred_class = target_class
        pred_class_probs = predictions[pred_class]
    else: # Use highest probability class
        pred_class = idx[0]
        pred_class_probs = predictions[pred_class]

    nc, h, w = out_cam[0].shape
    cam = weight_softmax[pred_class].dot(out_cam[0].reshape((nc, h*w)))

    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    try: # As per the source CV2 used because of its smooth interpolation resizing
        import cv2
        res_cam_img = cv2.resize(cam_img, out_size) 
    except ImportError: # Otherwise, resize to boring ol' squares
        res_cam_img = np.resize(cam_img, out_size)
    
    return res_cam_img, pred_class, pred_class_probs