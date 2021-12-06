"""
.. module:: segat
   :platform: Python
   :synopsis: An Adversarial Attack module for semantic segmentation neural networks in Pytorch. 
              segat (semantic segmentation attacks) requires that the input images are in the form of the network input.
              For testing a models robustness and adversarial training it advised to avoid the untargeted methods.
.. moduleauthor:: Lawrence Stewart <lawrence.stewart@valeo.com>
"""
from cocoproc import cocodata
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange
import os
from pycocotools.coco import COCO
from pycocotools import mask
from cocoproc import custom_transforms
from PIL import Image, ImageFile
from torchvision import transforms as torch_transform
from cocoproc.utils import decode_segmap, get_pascal_labels
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
import pylab
from torchvision.models.segmentation import deeplabv3_resnet101
import matplotlib.patches as mpatches
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim 

#TO DOS --- Work out what the default clipping value and alpha value should be 
# Implement each of the four attacks


class FGSM():
    """
    FGSM class containing all attacks and miscellaneous functions 
    """

    def __init__(self,model,loss,alpha=1,eps=1): 
        """ Creates an instance of the FGSM class.
        Args:
           model (torch.nn model):  The chosen model to be attacked,
                                    whose output layer returns the nonlinearlity of the pixel
                                    being in each of the possible classes.
           loss  (function):  The loss function to model was trained on
          
        Kwargs:
            alpha (float):  The learning rate of the attack
            eps   (float):  The clipping value for Iterated Attacks
        Returns:
           Instance of the FGSM class
        """
        self.model=model
        self.loss=loss
        self.eps=eps
        self.alpha=alpha
        self.predictor=None
        self.default_its=min(int(self.eps+4),int(1.25*self.eps))
        #set the model to evaluation mode for FGSM attacks
        self.model.eval()


    def untargeted(self,img,pred,labels):
        """Performs a single step untargeted FGSM attack
        Args:
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel 
                                for the whole image
            labels (torch.tensor):  The true labelelling of each pixel in the image
        Returns:
           adv (torch.tensor):  The pertubed image
           noise (torch.tensor): The adversarial noise added to the image during the attack
        """

        l=self.loss(pred,labels)
        img.retain_grad()
        torch.sum(l).backward()
        im_grad=img.grad
        noise=self.alpha*torch.sign(im_grad)
        adv=img+noise
        return adv, noise 



    def targeted(self,img,pred,target):
        """Performs a single step targeted FGSM attack
        Args:
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel 
                                 for the whole image
            target (torch.tensor):  The target labelling for each pixel
        Returns:
           adv (torch.tensor):  The pertubed image
           noise (torch.tensor): The adversarial noise added to the image during the attack
        """

        l=self.loss(pred,target)
        img.retain_grad()
        torch.sum(l).backward()
        im_grad=img.grad
        noise=-self.alpha*torch.sign(im_grad)
        adv=img+noise
        return adv, noise



    def iterated(self,img,pred,labels,its=None,targeted=False):
        """Performs iterated untargeted or targeted FGSM attack
        often referred to as FGSMI
        Args:
            model (torch.nn model): The pytorch model to be attacked
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel 
                                for the whole image
            labels (torch.tensor):  The true labelelling of each pixel in the image
                                    or the target labelling we wish the network to misclassify
                                    the network as (this should match the choice of the targeted 
                                    variable)
        Kwargs:
            its (int):  The number of iterations to attack
            targeted (boolean): False for untargeted attack and True for targeted
        Returns:
           adv (torch.tensor):  The pertubed image
           noise (torch.tensor): The adversarial noise added to the image during the attack
        """

        #set number of iterations to be the default value if not given
        its = self.default_its if its is None else its
        adv=img
        tbar=trange(its)
        for i in tbar:
            pred=self.predictor(adv)
            l=self.loss(pred,labels)
            img.retain_grad()
            torch.sum(l).backward()
            im_grad=img.grad

            #zero the gradients for the next iteration
            self.model.zero_grad()
            #Here the update is GD projected onto ball of radius clipping
            if targeted:
                noise=-self.alpha*torch.sign(im_grad).clamp(-self.eps,self.eps)
            else:
                noise=self.alpha*torch.sign(im_grad).clamp(-self.eps,self.eps)

            adv=adv+noise
            tbar.set_description('Iteration: {}/{} of iterated-FGSMI attack'.format(i, its))
        return adv, noise 



    def iterated_least_likely(self,img,pred,its=None):
        """Performs iterated untargeted FGSM attack towards the 
        least likely class, often referred to as FGSMII
        Args:
            model (torch.nn model): The pytorch model to be attacked
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel 
                                for the whole image
            labels (torch.tensor):  The true labelelling of each pixel in the image
        Kwargs:
            its (int):  The number of iterations to attack 
        Returns:
           adv (torch.tensor):  The pertubed image
           noise (torch.tensor): The adversarial noise added to the image during the attack
        """

        #set number of iterations to be the default value if not given
        its = self.default_its if its is None else its
        adv=img
        with torch.no_grad():
            pred=self.predictor(adv)
            targets=torch.argmax(pred[0],0)
            targets=targets.reshape(1,targets.size()[0],-1)
        tbar = trange(its)
        for i in tbar:
            pred=self.predictor(adv)
            l=self.loss(pred,targets)
            img.retain_grad()
            torch.sum(l).backward()
            im_grad=img.grad

            #zero the gradients for the next iteration
            self.model.zero_grad()
            #Here the update is GD projected onto ball of radius clipping
            noise=-self.alpha*torch.sign(im_grad).clamp(-self.eps,self.eps)
            adv=adv+noise
            tbar.set_description('Iteration: {}/{} of iterated-FGSMII attack'.format(i, its))
        return adv, noise 




    def ssim_iterated(self,img,pred,labels,its=None,targeted=False,threshold=0.99):
        """Performs iterated untargeted or targeted FGSM attack
        often referred to as FGSMI, halfing the current value of 
        alpha until the ssim value between the origional and perturbed image
        reaches the threshold value
        Args:
            model (torch.nn model): The pytorch model to be attacked
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel 
                                for the whole image
            labels (torch.tensor):  The true labelelling of each pixel in the image
                                    or the target labelling we wish the network to misclassify
                                    the network as (this should match the choice of the targeted 
                                    variable)
        Kwargs:
            its (int):  The number of iterations to attack
            targeted (boolean): False for untargeted attack and True for targeted
            threshold (float): Threshold ssi value to obtain
        Returns:
           adv (torch.tensor):  The pertubed image
           noise (torch.tensor): The adversarial noise added to the image during the attack
        """

        #set number of iterations to be the default value if not given


        ssim_val=0
        counter=0
        self.alpha=self.alpha*2
        its = self.default_its if its is None else its

        tbar=trange(its)
        
        img_ar=img[0].cpu().detach().numpy()
        img_ar=np.transpose(img_ar,(1,2,0))
        img_ar=self.denormalise(img_ar)

        while ssim_val<0.99:
            self.alpha = self.alpha/2
            adv=img
            counter+=1
            for i in tbar:
                pred=self.predictor(adv)
                l=self.loss(pred,labels)
                img.retain_grad()
                torch.sum(l).backward()
                im_grad=img.grad

                #zero the gradients for the next iteration
                self.model.zero_grad()
                #Here the update is GD projected onto ball of radius clipping
                if targeted:
                    noise=-self.alpha*torch.sign(im_grad).clamp(-self.eps,self.eps)
                else:
                    noise=self.alpha*torch.sign(im_grad).clamp(-self.eps,self.eps)

                adv=adv+noise
                tbar.set_description('Iteration: {}/{} of iterated-FGSMI attack- attempt {}'.format(i, its,counter))

            # convert to numpy array
            adv_ar=adv[0].cpu().detach().numpy()
            adv_ar=np.transpose(adv_ar,(1,2,0))
            adv_ar=self.denormalise(adv_ar)

            ssim_val=ssim(adv_ar,img_ar,multichannel=True)


        return adv, noise 




    def untargeted_varied_size(self,img,pred,labels,alphas=[0,0.005,0.01]):
        """Performs a single step untargeted FGSM attack for each of the given
        values of alpha.
        Args:
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel 
                                for the whole image
            labels (torch.tensor):  The true labelelling of each pixel in the image
        Kwargs:
            alphas (float list): The values of alpha to perform the attacks with
        Returns:
           adv_list (torch.tensor list):  The list of the pertubed images
           noise_list (torch.tensor list): The list of the adversarial noises created
        """
        if alphas==[]:
            raise Exception("alphas must be a non empty list")
        #create the output lists
        l=self.loss(pred,labels)
        img.retain_grad()
        torch.sum(l).backward()
        im_grad=img.grad
        noise_list=[alpha*torch.sign(im_grad) for alpha in alphas]
        adv_list=[img+noise for noise in noise_list]
        return adv_list, noise_list 
        

    def targeted_varied_size(self,img,pred,target,alphas=[0,0.005,0.01]):
        """Performs a single step targeted FGSM attack for each of the given
        values of alpha.
        Args:
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel 
                                for the whole image
            target (torch.tensor):  The target labelling that we wish the network
                                    to mixclassify the pixel as.
        Kwargs:
            alphas (float list): The values of alpha to perform the attacks with
        Returns:
           adv_list (torch.tensor list):  The list of the pertubed images
           noise_list (torch.tensor list): The list of the adversarial noises created
        """
        if alphas==[]:
            raise Exception("alphas must be a non empty list")
        #create the output lists
        l=self.loss(pred,target)
        img.retain_grad()
        torch.sum(l).backward()
        im_grad=img.grad
        noise_list=[-alpha*torch.sign(im_grad) for alpha in alphas]
        adv_list=[img+noise for noise in noise_list]
        
        return adv_list, noise_list 



    def DL3_pred(self,img):
        """Extractor function for deeplabv3 pretained: Please add your own
            to the self.predictor variable to suite your networks output 
        Args:
            img (torch.tensor):  The image to be pertubed to attack the network
        Returns: out (torch.tensor): Predicted semantic segmentation
           
        """
        out=self.model(img)['out']
        return out


    def denormalise(self,img):
        """
        Denormalises an image using the image net mean and 
        std deviation values.
        Args:
            img (numpy array): The image to be denormalised
        Returns:
            img (numpy array): The denormalised image
        """
        img *= (0.229, 0.224, 0.225)
        img += (0.485, 0.456, 0.406)
        img *= 255.000
        img=img.astype(np.uint8).clip(0,255)
        return img

