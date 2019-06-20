import os
import datetime as dt

from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm
import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torchvision.transforms import Compose, Normalize
from torchvision.transforms import ToTensor
import torchvision.utils as vutils

from processing_utils import image_manipulations as i_manips
from processing_utils import visualise
from processing_utils import analysis_utils
from processing_utils.data_funcs import create_dir_if_not_exist
from processing_utils.analysis_utils import var_to_cpu

from model.retinanet.model import nms
from model.retinanet.losses import FocalLoss
from model.retinanet.utils import BBoxTransform, ClipBoxes

# import pdb; pdb.set_trace()

class ImageAnalysis():
    def __init__(self, model, classes, train_loader=None, val_loader=None, means=None, sdevs=None):
        ## Model components
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        ## Norm-parameters
        self.means = means
        self.sdevs = sdevs
        self.classes = classes

        # Tracking
        self.start_time = dt.datetime.now()
        self.epoch_now = 1
        
        self.loss_tracker = None
        self.writer = None
        self.visualiser = False

    def instantiate_loss_tracker(self, output_dir='outputs/'):
        self.loss_tracker = analysis_utils.LossRecorder(output_dir)

    def instantiate_visualiser(self, filepath=None):
        from tensorboardX import SummaryWriter
        self.visualiser = True

        if not filepath:
            self.writer = SummaryWriter('/tmp/log')
        else:
            self.writer = SummaryWriter(filepath)
        # self.writer.add_graph(self.model, next(iter(self.train_loader))[0]

    def _get_batch_loss_and_preds(self, images, labels, criterion):
        outputs = self.model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        outputs = None
        return loss, preds

    def _visualise_loss(self, settings, epoch, epoch_accuracy, split):
        if self.visualiser:
            self.writer.add_scalar(f'{split}/Loss', self.loss_tracker.all_loss[split][-1], epoch)
            self.writer.add_scalar(f'{split}/Accuracy', epoch_accuracy, epoch)

    def _save_if_best(self, avg_epoch_val_loss, model, out_name):
        if len(self.loss_tracker.all_loss['val']) > 0 and self.loss_tracker.store_models is True:
            if avg_epoch_val_loss == min(self.loss_tracker.all_loss['val']):
                self.loss_tracker.save_model(model, out_name)

    def _print_results(self, epoch, loss, accuracy, split):
        print(f"{split} {epoch} accuracy: {accuracy:.4f}")
        print(f"{split} {epoch} final loss: {loss:.4f}\n")

class AnnotatedImageAnalysis(ImageAnalysis):
    """Performs single-label scene annotation
    TODO: refactor repeated code (e.g. timekeeping)"""
    def __init__(self, model, classes, means, sdevs, train_loader=None, val_loader=None):    
        super().__init__(model, classes, train_loader, val_loader, means, sdevs)

    def run_singletask_model(self, settings, split, loader, optimizer=False):
        loss = 0
        accuracies = []
        conf_matrix = analysis_utils.ConfusionMatrix(len(self.classes))
        for i, batch in enumerate(loader):
            if optimizer:
                optimizer.zero_grad()

            images, labels = analysis_utils.imgs_labels_to_variables(batch[0], batch[1])
            batch_loss, preds = self._get_batch_loss_and_preds(images, labels, settings['criterion'])
            [conf_matrix.update(int(var_to_cpu(labels[i])), int(var_to_cpu(preds[i]))) for i in range(len(labels))]

            if optimizer:
                batch_loss.backward()
                optimizer.step()
            loss += batch_loss.item()

            if (i+1) % settings['report_interval'][split] == 0:
                print(f"{split}: [{i} out of {len(loader)}] : {loss/(i+1):.4f}")
            
            accuracies.append(analysis_utils.get_mean_acc(preds, labels))

            # Memory management - must be cleared else the output between train/val phase are both stored
            # Which leads to 2x memory use
            images = labels = batch_loss = None

        loss = loss/(i+1)
        accuracies = np.mean(accuracies)
        return loss, accuracies, conf_matrix.matrix
 
    def compute_cam_imgs(self, target_imgs, cam_layer, labels=None, target_class=None):
<<<<<<< HEAD
        img_size = (target_imgs.shape[2], target_imgs.shape[3])
=======
        img_size = (target_imgs.shape[0], target_imgs.shape[1])
>>>>>>> 0a23e16404f2ab798b8cb2150daaccd110e91f71
        
        cam_imgs = torch.ones(target_imgs.shape)
        for i,_ in enumerate(target_imgs):
            img = target_imgs[i]
<<<<<<< HEAD
            if labels is not None:
                label = labels[i].item()
=======
            if labels:
                label = labels[i] 
>>>>>>> 0a23e16404f2ab798b8cb2150daaccd110e91f71
    
            cam, pred_class, pred_class_probs = visualise.get_cam_img(img, self.model, cam_layer, img_size)

            cam_colored = np.uint8(cm.rainbow(cam)*255)
            out_cam = Image.fromarray(cam_colored[:,:,:3], 'RGB')
            unnormed_tensor = visualise.decode_image(img, self.means, self.sdevs)
            in_array = np.uint8(unnormed_tensor*255)
            
            original_img = Image.fromarray(in_array, 'RGB')
            overlaid_cam = Image.blend(original_img, out_cam, 0.3)

            if label:
<<<<<<< HEAD
                caption = f'pred: {self.classes[pred_class]} ({pred_class_probs*100:.2f}%), true: {self.classes[label]}'
            else:
                caption = f'pred: {self.classes[pred_class]} ({pred_class_probs*100:.2f}%)'
=======
                caption = f'pred: {self.classes[pred_class]} ({float(pred_class_probs*100):.2f}%), true: {self.classes[label[0]]}'
            else:
                caption = f'pred: {self.classes[pred_class]} ({float(pred_class_probs*100):.2f}%)'
>>>>>>> 0a23e16404f2ab798b8cb2150daaccd110e91f71
            
            draw = ImageDraw.Draw(overlaid_cam)
            font = ImageFont.load_default().font
            draw.text((10, 10), caption, fill=(255,0,125), font=font)

            to_tensor = ToTensor()
            cam_tensor = to_tensor(overlaid_cam)            

            cam_imgs[i:,:,:] = cam_tensor

        return cam_imgs

    def train(self, settings):
        """Performs model training"""
        if self.loss_tracker.store_loss is True:
            self.loss_tracker.set_loss_file('train')
        if 'lr_decay_patience' in settings:
            lr_scheduler = ReduceLROnPlateau(settings['optimizer'],
                                             'min',
                                             factor=settings['lr_decay'],
                                             patience=settings['lr_decay_patience'],
                                             verbose=True)

        for epoch in range(settings['n_epochs']):
            self.model = self.model.train()

            epoch_train_loss, epoch_train_accuracy, train_conf_matrix = self.run_singletask_model(settings, 'train', self.train_loader, optimizer=settings['optimizer'])
            
            self.epoch_now = len(self.loss_tracker.all_loss['train'])+1

            epoch_now = len(self.loss_tracker.all_loss['val'])+1
            self.loss_tracker.store_epoch_loss('train', epoch_now, epoch_train_loss, epoch_train_accuracy)
            self.loss_tracker.conf_matrix['train'].append(train_conf_matrix)
        
            if self.val_loader is not None:
                self.validate(settings)

            if epoch_now % settings['save_interval'] == 0 and self.loss_tracker.store_models is True:
                print("Checkpoint-saving model")
                self.loss_tracker.save_model(self.model, epoch)

            self._visualise_loss(settings, epoch_now, epoch_train_accuracy, 'train')
            self._print_results(epoch_now, epoch_train_loss, epoch_train_accuracy, 'train')
            print('Training confusion matrix:\n', train_conf_matrix)

            if 'lr_decay_epoch' in settings:
                if epoch in settings['lr_decay_epoch']:
                    analysis_utils.decay_optimizer_lr(settings['optimizer'], settings['lr_decay'])
                    print(f"\nlr decayed by {settings['lr_decay']}\n")
            elif 'lr_decay_patience' in settings:
                lr_scheduler.step(epoch_train_loss)

        if settings['shutdown'] is True:
            os.system("shutdown")        

    def validate(self, settings):
        """For a given model, evaluation criterion,
        and validation loader, performs a single evaluation
        pass."""

        self.model = self.model.eval()

        with torch.no_grad():
            if self.loss_tracker.store_loss is True:
                self.loss_tracker.set_loss_file('val')

            epoch_val_loss, epoch_val_accuracy, val_conf_matrix = self.run_singletask_model(settings,
                                                                        'val',
                                                                        self.val_loader)

            self.loss_tracker.store_epoch_loss('val', self.epoch_now, epoch_val_loss, epoch_val_accuracy)
            self.loss_tracker.conf_matrix['val'].append(val_conf_matrix)
            self._visualise_loss(settings, self.epoch_now, epoch_val_accuracy, 'val')

            self._save_if_best(epoch_val_loss, self.model, settings['run_name']+'_best')
            self._print_results(self.epoch_now, epoch_val_loss, epoch_val_accuracy, 'val')
        
            if settings['cam_layer'] is not None and self.visualiser:
                images, labels = next(iter(self.val_loader))
                cam_imgs = self.compute_cam_imgs(images, settings['cam_layer'], labels)

                cam_grid = vutils.make_grid(cam_imgs, nrow=len(cam_imgs), normalize=True, scale_each=True)
                self.writer.add_image("Validation batch CAMs", cam_grid, self.epoch_now, dataformats='CHW')                
        
        self._print_results(self.epoch_now, epoch_val_loss, epoch_val_accuracy, 'val')
        print('Validation confusion matrix:\n', val_conf_matrix)

    def infer(self, image, transforms, cam_layer=None, target_class=None):
        """Takes a single image and computes the most likely class
        """
        self.model = self.model.eval()
        image = transforms(image).unsqueeze(0)  
        if torch.cuda.is_available() and cam_layer is None:
            image = Variable(image.cuda())
        else:
            image = Variable(image)
        output = self.model(image)
        confidence = float(torch.max(nn.Softmax(output, dim=1)))
        _, predicted_class_index = torch.max(output, 1)
        predicted_class = self.classes[int(predicted_class_index)]
        
        if cam_layer:
            cam_imgs = self.compute_cam_imgs(self, image, cam_layer)
            return cam_imgs
        else:
            return output, confidence

class ObjectDetection(ImageAnalysis):
    def __init__(self, model, classes, means, sdevs, train_loader=None, val_loader=None, line_colors=None):
        super().__init__(model, classes, train_loader, val_loader, means, sdevs)
        self.line_col = line_colors

        self.epoch_now = 1

    def run_singletask_model(self, settings, split, loader, optimizer=False):
        loss = 0
        accuracies = []
        for i, batch in enumerate(loader):
            if optimizer:
                optimizer.zero_grad()
            images = Variable(batch[0])
            labels = Variable(batch[1])
            if torch.cuda.is_available:
                images = images.cuda()
                labels = labels.cuda()       

            losses, preds = self._get_batch_loss_and_preds(images, labels, settings['criterion'])
            batch_loss = losses[0] + losses[1]
            if optimizer:
                batch_loss.backward()
                optimizer.step()

            loss += batch_loss
            # accuracies.append(analysis_utils.get_mean_acc(preds, labels))
            if (i+1) % settings['report_interval'][split] == 0:
                print(f"{split}: [{i} out of {len(loader)}]\nClassification: {losses[0]/(i+1):.4f}\nRegression: {losses[1]/(i+1):.4f}")

            if self.visualiser and i+1==len(loader):
                visualise.draw_rectangles(images, preds['bboxes'], preds['pred_class'], preds['prob'], gt=labels)
                self._imgs_to_tensorboard(images, split)

            # Memory management - must be cleared else the output between train/val phase are both stored
            # Which leads to 2x memory use
            images = labels = batch_loss = None

        loss = loss/(i+1)
        return loss

    def _get_batch_loss_and_preds(self, images, labels, criterion):
        regression, classification, anchors = self.model([images, labels])
        criterion = FocalLoss()
        loss = criterion.forward(regression, classification, anchors, labels)
        preds = self._get_surpressed_boxes(images, regression, classification, anchors)
        return loss, preds

    def _get_surpressed_boxes(self, img_batch, regression, classification, anchors):
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

        transformed_anchors = self.regressBoxes.forward(anchors, regression)
        transformed_anchors = self.clipBoxes.forward(transformed_anchors, img_batch)

        scores = torch.max(classification, dim=2, keepdim=True)[0]

        scores_over_thresh = (scores>0.05)[0, :, 0]

        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just return
            return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

        classification = classification[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]

        anchors_nms_idx = nms(transformed_anchors, scores, overlap=0.5)

        nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
        
        outputs = { 'bboxes': transformed_anchors[0, anchors_nms_idx, :],
                    'pred_class': nms_class,
                    'prob': nms_scores }

        return outputs

    def _imgs_to_tensorboard(self, imgs, split):
        img = visualise.decode_image(imgs, self.means, self.sdevs)
        imag = torch.Tensor(img.permute(0,3,1,2))
        row_views = imag[:,:3,:,:] # Grab only colour bands on image

        side_view = vutils.make_grid(row_views, nrow=len(img), normalize=True, scale_each=True)
        self.writer.add_image(f'{split}_Predicted-from-Image', side_view, self.epoch_now, dataformats='CHW')
        
    def train(self, settings):
        """Performs model training"""
        if self.loss_tracker.store_loss is True:
            self.loss_tracker.set_loss_file('train')
        if 'lr_decay_patience' in settings:
            lr_scheduler = ReduceLROnPlateau(settings['optimizer'],
                                             'min',
                                             factor=settings['lr_decay'],
                                             patience=settings['lr_decay_patience'],
                                             verbose=True)

        for epoch in range(settings['n_epochs']):
            self.model = self.model.train()
            epoch_train_loss = self.run_singletask_model(settings,
                                                        'train',
                                                        self.train_loader,
                                                        optimizer=settings['optimizer'])

            self.epoch_now = len(self.loss_tracker.all_loss['train'])+1
            # self.loss_tracker.store_epoch_loss('train', self.epoch_now, epoch_train_loss, epoch_train_accuracy)
        
            if self.val_loader is not None:
                self.validate(settings)

            if self.epoch_now % settings['save_interval'] == 0 and self.loss_tracker.store_models is True:
                print("Checkpoint-saving model")
                self.loss_tracker.save_model(self.model, epoch)

            # self._visualise_loss(settings, self.epoch_now, epoch_train_accuracy, 'train')
            # self._print_results(self.epoch_now, epoch_train_loss, epoch_train_accuracy, 'train')

            if 'lr_decay_epoch' in settings:
                if epoch in settings['lr_decay_epoch']:
                    analysis_utils.decay_optimizer_lr(settings['optimizer'], settings['lr_decay'])
                    print(f"\nlr decayed by {settings['lr_decay']}\n")
            elif 'lr_decay_patience' in settings:
                lr_scheduler.step(epoch_train_loss)            
                
        if settings['shutdown'] is True:
            os.system("shutdown")

    def validate(self, settings):
        """For a given model, evaluation criterion,
        and validation loader, performs a single evaluation
        pass."""
        self.model = self.model.eval()

        with torch.no_grad():
            if self.loss_tracker.store_loss is True:
                self.loss_tracker.set_loss_file('val')

            epoch_val_loss, epoch_val_accuracy = self.run_singletask_model(settings,
                                                                        'val',
                                                                        self.val_loader)

            self.loss_tracker.store_epoch_loss('val', self.epoch_now, epoch_val_loss, epoch_val_accuracy)

            self._visualise_loss(settings, self.epoch_now, epoch_val_accuracy, 'val')
            self._save_if_best(epoch_val_loss, self.model, settings['run_name']+'_best')
            self._print_results(self.epoch_now, epoch_val_loss, epoch_val_accuracy, 'val')

class SemSegAnalysis(ImageAnalysis):
    """Performs semantic segmentation
    TODO: refactor repeated code (e.g. timekeeping)"""
    def __init__(self, model, classes, means, sdevs, train_loader=None, val_loader=None, label_colors=None):
        super().__init__(model, classes, train_loader, val_loader, means, sdevs)
        self.lab_col = label_colors
        self.store_probs = False

        self.epoch_now = 1

    def run_singletask_model(self, settings, split, loader, optimizer=False):
        loss = 0
        accuracies = []
        for i, batch in enumerate(loader):
            if optimizer:
                optimizer.zero_grad()
            images, labels = analysis_utils.imgs_labels_to_variables(batch[0], batch[1])
            batch_loss, preds = self._get_batch_loss_and_preds(images, labels, settings['criterion'])
            if optimizer:
                batch_loss.backward()
                optimizer.step()

            loss += batch_loss.item()
            accuracies.append(analysis_utils.get_mean_acc(preds, labels))
            if (i+1) % settings['report_interval'][split] == 0:
                print(f"{split}: [{i} out of {len(loader)}] : {loss/(i+1):.4f}")

            if self.visualiser and i+1==len(loader):
                self._imgs_to_tensorboard(images, preds, split)

            # Memory management - must be cleared else the output between train/val phase are both stored
            # Which leads to 2x memory use
            images = labels = batch_loss = preds = None

        loss = loss/(i+1)
        epoch_acc = np.mean(accuracies)
        return loss, epoch_acc

    def _imgs_to_tensorboard(self, imgs, preds, split):
        img, pred = visualise.encoded_img_and_lbl_to_data(imgs, preds, self.means, self.sdevs, self.lab_col)
        predi = torch.Tensor(pred.permute(0,3,1,2))
        imag = torch.Tensor(img.permute(0,3,1,2))
        row_views = torch.cat((imag[:,:3,:,:], predi)) # Grab only colour bands on image

        side_view = vutils.make_grid(row_views, nrow=len(img), normalize=True, scale_each=True)
        self.writer.add_image(f'{split}_Predicted-from-Image', side_view, self.epoch_now, dataformats='CHW')
        
    def train(self, settings):
        """Performs model training"""
        if self.loss_tracker.store_loss is True:
            self.loss_tracker.set_loss_file('train')
        if 'lr_decay_patience' in settings:
            lr_scheduler = ReduceLROnPlateau(settings['optimizer'],
                                             'min',
                                             factor=settings['lr_decay'],
                                             patience=settings['lr_decay_patience'],
                                             verbose=True)

        for epoch in range(settings['n_epochs']):
            self.model = self.model.train()
            epoch_train_loss, epoch_train_accuracy = self.run_singletask_model(settings,
                                                                               'train',
                                                                               self.train_loader,
                                                                               optimizer=settings['optimizer'])

            self.epoch_now = len(self.loss_tracker.all_loss['train'])+1
            self.loss_tracker.store_epoch_loss('train', self.epoch_now, epoch_train_loss, epoch_train_accuracy)
        
            if self.val_loader is not None:
                self.validate(settings)

            if self.epoch_now % settings['save_interval'] == 0 and self.loss_tracker.store_models is True:
                print("Checkpoint-saving model")
                self.loss_tracker.save_model(self.model, epoch)

            self._visualise_loss(settings, self.epoch_now, epoch_train_accuracy, 'train')
            self._print_results(self.epoch_now, epoch_train_loss, epoch_train_accuracy, 'train')

            if 'lr_decay_epoch' in settings:
                if epoch in settings['lr_decay_epoch']:
                    analysis_utils.decay_optimizer_lr(settings['optimizer'], settings['lr_decay'])
                    print(f"\nlr decayed by {settings['lr_decay']}\n")
            elif 'lr_decay_patience' in settings:
                lr_scheduler.step(epoch_train_loss)            
                
        if settings['shutdown'] is True:
            os.system("shutdown")

    def validate(self, settings):
        """For a given model, evaluation criterion,
        and validation loader, performs a single evaluation
        pass."""
        self.model = self.model.eval()

        with torch.no_grad():
            if self.loss_tracker.store_loss is True:
                self.loss_tracker.set_loss_file('val')

            epoch_val_loss, epoch_val_accuracy = self.run_singletask_model(settings,
                                                                        'val',
                                                                        self.val_loader)

            self.loss_tracker.store_epoch_loss('val', self.epoch_now, epoch_val_loss, epoch_val_accuracy)

            self._visualise_loss(settings, self.epoch_now, epoch_val_accuracy, 'val')
            self._save_if_best(epoch_val_loss, self.model, settings['run_name']+'_best')
            self._print_results(self.epoch_now, epoch_val_loss, epoch_val_accuracy, 'val')
    
    def mask(self, input_image, transforms, output_fname, class_index=None, conf_lower_bound=None):
        in_img = Image.open(input_image)
        in_tensor = transforms(in_img).unsqueeze(0)

        if torch.cuda.is_available():
            img = Variable(in_tensor.cuda())
        else:
            img = Variable(in_tensor)
            
        output = self.model(img)
        
        if self.store_probs == True:
            # Stores exponentiated probabilities (since model outputs log-odds)
            preds = output.squeeze(0).data.cpu().numpy()
            preds = np.exp(preds)               
        else:
            # Stores argmax labels
            preds = analysis_utils.get_predictions(output, class_index, conf_lower_bound) # t_band to -1 to fix indexing
            preds = preds.cpu().numpy()*255

        out_img = Image.fromarray(preds.squeeze(0))
        out_img_resized = out_img.resize(in_img.size, resample=Image.NEAREST).convert('1')
        in_img.paste(out_img_resized, mask=out_img_resized)
        in_img.save(output_fname)    
    
    def infer_geo(self, air_photo, transforms, output_dir, tile_size, nout_classes, target_band=None, conf_lower_bound=None):
        """With a deployed model and input directory, performs model evaluation
        on the image contents of that folder, then writes them to the output folder.
        """
        from osgeo import gdal
        self.model = self.model.eval()
        create_dir_if_not_exist(output_dir)
        raster = gdal.Open(air_photo)
        data = np.array(raster.ReadAsArray())

        width = data.shape[1]
        height = data.shape[2]

        out_raster = np.zeros([nout_classes, width, height])

        in_img = Image.open(air_photo)
        in_photo = transforms(in_img).unsqueeze(0)

        for w in range(0, width, tile_size):
            for h in range(0, height, tile_size):

                # Scales image extent back to max width/height where applicable
                if w+tile_size > width:
                    w = width - tile_size
                if h+tile_size > height:                    
                    h = height - tile_size

                tile = in_photo[:, :, w:w+tile_size, h:h+tile_size]
                
                if torch.cuda.is_available():
                    img = Variable(tile.cuda())
                else:
                    img = Variable(tile)
                    
                output = self.model(img)
                
                if self.store_probs == True:
                    # Stores exponentiated probabilities (since model outputs log-odds)
                    preds = output.squeeze(0).data.cpu().numpy()
                    preds = np.exp(preds)               
                else:
                    # Stores argmax labels
                    preds = analysis_utils.get_predictions(output, target_band-1, conf_lower_bound) # t_band to -1 to fix indexing
                    preds = preds.cpu().numpy()

                out_raster[:,w:w+tile_size, h:h+tile_size] = preds

        driver = gdal.GetDriverByName('GTiff')        
        bands = out_raster.shape[0]

        if bands == 1:
            datatype = gdal.GDT_Byte
        else:
            datatype = gdal.GDT_Float32

        out_file = os.path.join(output_dir, 'temp_raster.tif')
        raster_file = driver.Create(out_file, width, height, bands, datatype)
        raster_file.SetGeoTransform(raster.GetGeoTransform())
        raster_file.SetProjection(raster.GetProjection())
        
        for i, band in enumerate(out_raster):
            raster_file.GetRasterBand(i+1).WriteArray(band)
            raster_file.GetRasterBand(i+1).SetNoDataValue(0)
        
        raster_file.FlushCache()
               
    def store_probabilities(self):
        """Sets the model to store class probabilities per prediction rather
        than returning an argmax b&w image."""
        self.store_probs = True