import torch
import torch.nn as nn

import torchvision.transforms as T

import numpy as np
import cv2
import os

#%%# Run command

#img_path = './surfer/'
#tracker = mosse(lr, sigma, num_pretrain, rotate, img_path)
#tracker.start_tracking([250, 100, 120, 180])

#%%# 

def linear_mapping(img):
    return (img - img.min()) / (img.max() - img.min())

def pre_process(img):
    # get the size of the img...
    height, width = img.shape
    # intensity normalization
    img = np.log(img + 1)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    # Apply the cosine window
    window = window_func_2d(height, width)
    img = img * window

    return img

# cosine window generation function
def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)

    win = mask_col * mask_row

    return win

def random_warp(img):
    a = -180 / 16
    b = 180 / 16
    r = a + (b - a) * np.random.uniform()
    # rotate the image...
    matrix_rot = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), r, 1)
    img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0]))
    img_rot = img_rot.astype(np.float32) / 255
    return img_rot

#%%#

# alpha value
lr = 0.125
# variance for the target Gaussian response
sigma = 100
# number of frames to be used for the pre-training
num_pretrain =  1
# rotation augmentation?
rotate = True


#%%# 

class mosse:
    def __init__(self, lr, sigma, num_pretrain, rotate, img_path, include_rgb=False):
        # get arguments..
        self.lr = lr
        self.sigma = sigma
        self.num_pretrain = num_pretrain
        self.rotate = rotate

        self.img_path = img_path
        # get the img lists...
        self.frame_lists = self._get_img_lists(self.img_path)
        self.frame_lists.sort()

        self.window_list = []
        
        self.gi_list= []
        
        self.include = include_rgb
        
        # VGG Network
        self.vgg = nn.Sequential(
        			# encode 1-1
        			nn.Conv2d(3, 3, kernel_size=(1,1), stride= (1, 1)),
        			nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        			nn.ReLU(inplace=True), # relu 1-1
        			# encode 2-1
        			nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        			nn.ReLU(inplace=True),
        			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

        			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        			nn.ReLU(inplace=True), # relu 2-1
        			# encoder 3-1
        			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        			nn.ReLU(inplace=True),

        			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        			nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        			nn.ReLU(inplace=True), # relu 3-1
        			# encoder 4-1
        			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        			nn.ReLU(inplace=True),
        			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        			nn.ReLU(inplace=True),
        			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        			nn.ReLU(inplace=True),
        			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

        			nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        			nn.ReLU(inplace=True), # relu 4-1
        			# rest of vgg not used
        			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        			nn.ReLU(inplace=True),
        			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        			nn.ReLU(inplace=True),
        			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        			nn.ReLU(inplace=True),
        			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

        			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        			nn.ReLU(inplace=True),
        			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        			nn.ReLU(inplace=True),
        			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        			nn.ReLU(inplace=True),
        			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        			nn.ReLU(inplace=True)
        			)
        self.vgg.load_state_dict(torch.load("vgg_normalized.pth"))
        
        # Result
        self.result_list_org =  []
        self.result_list_l1 =   []
        self.result_list_l2 =   []
        self.result_list_int =  []

    def _LbFS(self, img, features, init_gt, top_n=4):
        
        img_gray = T.ToTensor()(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                
        l1_list = []
        l2_list = []
        
        for feature in features:
            
            i = img_gray[0, init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
            f = feature[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
            
            l1_list.append(nn.L1Loss()(i, f).item())
            l2_list.append(nn.MSELoss()(i, f).item())
        
        l1_sort = sorted(range(len(l1_list)), key=lambda k: l1_list[k])
        l2_sort = sorted(range(len(l2_list)), key=lambda k: l2_list[k])
        
        return l1_sort[:top_n], l2_sort[:top_n]

    # pre train the filter on the first frame...
    def _pre_training(self, init_frame, G):
        height, width = G.shape
        fi = cv2.resize(init_frame, (width, height))
        # pre-process img..
        fi = pre_process(fi)
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))

        # pre-training sequence
        for _ in range(self.num_pretrain):
            if self.rotate:
                fi = pre_process(random_warp(init_frame))
            else:
                fi = pre_process(init_frame)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        
        return Ai, Bi
    
    # start to do the object tracking...
    def start_tracking(self, init_gt):

        #%%# Stage 1 : Filter Initialization
        #init_frame = cv2.imread(self.frame_lists[0], cv2.IMREAD_GRAYSCALE)
        init_img = cv2.imread(self.frame_lists[0])
        init_frame = cv2.cvtColor(init_img.copy(), cv2.COLOR_BGR2GRAY)
        init_frame = init_frame.astype(np.float32)

        # Loss-based Feature Selection
        features = self.vgg[:3](T.ToTensor()(init_img)).detach()
        l1_idx, l2_idx = self._LbFS(init_img, features, init_gt)
        
        
        init_feat1 = features[l1_idx]
        init_feat2 = features[l2_idx]

        response_map = self._get_gauss_response(init_frame, init_gt)
        g = response_map[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
        G = np.fft.fft2(g)

        # Origin       
        fi = init_frame[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
        # L1-Loss
        fi1= init_feat1[:, init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]].numpy()
        # L2-Loss
        fi2= init_feat2[:, init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]].numpy()

        # Pretrain Origin
        Ai, Bi = self._pre_training(fi, G)
        # Pretrain L1
        l1_Ai1, l1_Bi1 = self._pre_training(linear_mapping(fi1[0])*1, G)
        l1_Ai2, l1_Bi2 = self._pre_training(linear_mapping(fi1[1])*1, G)
        l1_Ai3, l1_Bi3 = self._pre_training(linear_mapping(fi1[2])*1, G)
        l1_Ai4, l1_Bi4 = self._pre_training(linear_mapping(fi1[3])*1, G)
        # Pretrain L2
        l2_Ai1, l2_Bi1 = self._pre_training(linear_mapping(fi2[0])*1, G)
        l2_Ai2, l2_Bi2 = self._pre_training(linear_mapping(fi2[1])*1, G)
        l2_Ai3, l2_Bi3 = self._pre_training(linear_mapping(fi2[2])*1, G)
        l2_Ai4, l2_Bi4 = self._pre_training(linear_mapping(fi2[3])*1, G)
        
        

        # start the tracking...
        for idx in range(len(self.frame_lists)):
            # Origin
            current_frame = cv2.imread(self.frame_lists[idx])
            frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray.astype(np.float32)
            
            
            features = self.vgg[:3](T.ToTensor()(current_frame)).detach()
            current_l1 = features[l1_idx].numpy()
            current_l2 = features[l2_idx].numpy()
            
            if idx == 0:
                Ai, Bi = self.lr * Ai, self.lr * Bi
                
                l1_Ai1, l1_Bi1 = self.lr * l1_Ai1, self.lr * l1_Bi1 
                l1_Ai2, l1_Bi2 = self.lr * l1_Ai2, self.lr * l1_Bi2 
                l1_Ai3, l1_Bi3 = self.lr * l1_Ai3, self.lr * l1_Bi3 
                l1_Ai4, l1_Bi4 = self.lr * l1_Ai4, self.lr * l1_Bi4 
                
                l2_Ai1, l2_Bi1 = self.lr * l2_Ai1, self.lr * l2_Bi1 
                l2_Ai2, l2_Bi2 = self.lr * l2_Ai2, self.lr * l2_Bi2 
                l2_Ai3, l2_Bi3 = self.lr * l2_Ai3, self.lr * l2_Bi3 
                l2_Ai4, l2_Bi4 = self.lr * l2_Ai4, self.lr * l2_Bi4 
                
                pos = init_gt.copy()
                pos_l1 = init_gt.copy()
                pos_l2 = init_gt.copy()
                
                clip_pos = np.array([pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3]]).astype(np.int64)
                clip_pos_l1 = clip_pos.copy()
                clip_pos_l2 = clip_pos.copy()
            else:
                #%%# Stage 2 : Tracking
                
                # Origin
                Hi = Ai / Bi
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
                
                # L1
                l1_Hi1 = l1_Ai1 / l1_Bi1
                l1_Hi2 = l1_Ai2 / l1_Bi2
                l1_Hi3 = l1_Ai3 / l1_Bi3
                l1_Hi4 = l1_Ai4 / l1_Bi4
                fi_l1 = current_l1[:, clip_pos_l1[1]:clip_pos_l1[3], clip_pos_l1[0]:clip_pos_l1[2]]
                fi1_l1 = pre_process(cv2.resize(linear_mapping(fi_l1[0])*1, (init_gt[2], init_gt[3])))
                fi2_l1 = pre_process(cv2.resize(linear_mapping(fi_l1[1])*1, (init_gt[2], init_gt[3])))
                fi3_l1 = pre_process(cv2.resize(linear_mapping(fi_l1[2])*1, (init_gt[2], init_gt[3])))
                fi4_l1 = pre_process(cv2.resize(linear_mapping(fi_l1[3])*1, (init_gt[2], init_gt[3])))
                
                # L2
                l2_Hi1 = l2_Ai1 / l2_Bi1
                l2_Hi2 = l2_Ai2 / l2_Bi2
                l2_Hi3 = l2_Ai3 / l2_Bi3
                l2_Hi4 = l2_Ai4 / l2_Bi4
                fi_l2 = current_l2[:, clip_pos_l2[1]:clip_pos_l2[3], clip_pos_l2[0]:clip_pos_l2[2]]
                fi1_l2 = pre_process(cv2.resize(linear_mapping(fi_l2[0])*1, (init_gt[2], init_gt[3])))
                fi2_l2 = pre_process(cv2.resize(linear_mapping(fi_l2[1])*1, (init_gt[2], init_gt[3])))
                fi3_l2 = pre_process(cv2.resize(linear_mapping(fi_l2[2])*1, (init_gt[2], init_gt[3])))
                fi4_l2 = pre_process(cv2.resize(linear_mapping(fi_l2[3])*1, (init_gt[2], init_gt[3])))
                
                self.window_list.append(fi)
                
                #%%# Origin
                Gi = Hi * np.fft.fft2(fi)
                gi = linear_mapping(np.fft.ifft2(Gi))
                
                self.gi_list.append(gi)
                dx, dy = self._get_diff(gi)
                pos[0] += dx
                pos[1] += dy

                # L1
                l1_Gi1 = l1_Hi1 * np.fft.fft2(fi1_l1)
                l1_gi1 = linear_mapping(np.fft.ifft2(l1_Gi1))
                l1_dx1, l1_dy1 = self._get_diff(l1_gi1)
                
                l1_Gi2 = l1_Hi2 * np.fft.fft2(fi2_l1)
                l1_gi2 = linear_mapping(np.fft.ifft2(l1_Gi2))
                l1_dx2, l1_dy2 = self._get_diff(l1_gi2)
                
                l1_Gi3 = l1_Hi3 * np.fft.fft2(fi3_l1)
                l1_gi3 = linear_mapping(np.fft.ifft2(l1_Gi3))
                l1_dx3, l1_dy3 = self._get_diff(l1_gi3)
                
                l1_Gi4 = l1_Hi4 * np.fft.fft2(fi4_l1)
                l1_gi4 = linear_mapping(np.fft.ifft2(l1_Gi4))
                l1_dx4, l1_dy4 = self._get_diff(l1_gi4)
                
                if self.include:
                    pos_l1[0] += int((dx + l1_dx1 + l1_dx2 + l1_dx3 + l1_dx4)/5)
                    pos_l1[1] += int((dy + l1_dy1 + l1_dy2 + l1_dy3 + l1_dy4)/5)
                else:
                    pos_l1[0] += int((l1_dx1 + l1_dx2 + l1_dx3 + l1_dx4)/4)
                    pos_l1[1] += int((l1_dy1 + l1_dy2 + l1_dy3 + l1_dy4)/4)

                # L2
                l2_Gi1 = l2_Hi1 * np.fft.fft2(fi1_l2)
                l2_gi1 = linear_mapping(np.fft.ifft2(l2_Gi1))
                l2_dx1, l2_dy1 = self._get_diff(l2_gi1)
                
                l2_Gi2 = l2_Hi2 * np.fft.fft2(fi2_l2)
                l2_gi2 = linear_mapping(np.fft.ifft2(l2_Gi2))
                l2_dx2, l2_dy2 = self._get_diff(l2_gi2)
                
                l2_Gi3 = l2_Hi3 * np.fft.fft2(fi3_l2)
                l2_gi3 = linear_mapping(np.fft.ifft2(l2_Gi3))
                l2_dx3, l2_dy3 = self._get_diff(l2_gi3)
                
                l2_Gi4 = l2_Hi4 * np.fft.fft2(fi4_l2)
                l2_gi4 = linear_mapping(np.fft.ifft2(l2_Gi4))
                l2_dx4, l2_dy4 = self._get_diff(l2_gi4)
                
                if self.include:
                    pos_l2[0] += int((dx + l2_dx1 + l2_dx2 + l2_dx3 + l2_dx4)/5)
                    pos_l2[1] += int((dy + l2_dy1 + l2_dy2 + l2_dy3 + l2_dy4)/5)
                else:
                    pos_l2[0] += int((l2_dx1 + l2_dx2 + l2_dx3 + l2_dx4)/4)
                    pos_l2[1] += int((l2_dy1 + l2_dy2 + l2_dy3 + l2_dy4)/4)


                #%%#

                # Position Clipping
                clip_pos = self._clip(clip_pos, pos, current_frame)
                clip_pos_l1 = self._clip(clip_pos_l1, pos_l1, current_frame)
                clip_pos_l2 = self._clip(clip_pos_l2, pos_l2, current_frame)

                # Filter Update - Origin
                fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
                Ai = self.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.lr) * Ai
                Bi = self.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.lr) * Bi
                
                # Filter Update - L1
                fi_l1 = current_l1[:, clip_pos_l1[1]:clip_pos_l1[3], clip_pos_l1[0]:clip_pos_l1[2]]
                fi1_l1 = pre_process(cv2.resize(fi_l1[0], (init_gt[2], init_gt[3])))
                fi2_l1 = pre_process(cv2.resize(fi_l1[1], (init_gt[2], init_gt[3])))
                fi3_l1 = pre_process(cv2.resize(fi_l1[2], (init_gt[2], init_gt[3])))
                fi4_l1 = pre_process(cv2.resize(fi_l1[3], (init_gt[2], init_gt[3])))
                
                l1_Ai1 = self.lr * (G * np.conjugate(np.fft.fft2(fi1_l1))) + (1 - self.lr) * l1_Ai1
                l1_Bi1 = self.lr * (np.fft.fft2(fi1_l1) * np.conjugate(np.fft.fft2(fi1_l1))) + (1 - self.lr) * l1_Bi1
                
                l1_Ai2 = self.lr * (G * np.conjugate(np.fft.fft2(fi2_l1))) + (1 - self.lr) * l1_Ai2
                l1_Bi2 = self.lr * (np.fft.fft2(fi2_l1) * np.conjugate(np.fft.fft2(fi2_l1))) + (1 - self.lr) * l1_Bi2
                
                l1_Ai3 = self.lr * (G * np.conjugate(np.fft.fft2(fi3_l1))) + (1 - self.lr) * l1_Ai3
                l1_Bi3 = self.lr * (np.fft.fft2(fi3_l1) * np.conjugate(np.fft.fft2(fi3_l1))) + (1 - self.lr) * l1_Bi3
                
                l1_Ai4 = self.lr * (G * np.conjugate(np.fft.fft2(fi4_l1))) + (1 - self.lr) * l1_Ai4
                l1_Bi4 = self.lr * (np.fft.fft2(fi4_l1) * np.conjugate(np.fft.fft2(fi4_l1))) + (1 - self.lr) * l1_Bi4
                        
                # Filter Update - L2
                fi_l2 = current_l2[:, clip_pos_l2[1]:clip_pos_l2[3], clip_pos_l2[0]:clip_pos_l2[2]]
                fi1_l2 = pre_process(cv2.resize(fi_l2[0], (init_gt[2], init_gt[3])))
                fi2_l2 = pre_process(cv2.resize(fi_l2[1], (init_gt[2], init_gt[3])))
                fi3_l2 = pre_process(cv2.resize(fi_l2[2], (init_gt[2], init_gt[3])))
                fi4_l2 = pre_process(cv2.resize(fi_l2[3], (init_gt[2], init_gt[3])))
                
                l2_Ai1 = self.lr * (G * np.conjugate(np.fft.fft2(fi1_l2))) + (1 - self.lr) * l2_Ai1
                l2_Bi1 = self.lr * (np.fft.fft2(fi1_l2) * np.conjugate(np.fft.fft2(fi1_l2))) + (1 - self.lr) * l2_Bi1
                
                l2_Ai2 = self.lr * (G * np.conjugate(np.fft.fft2(fi2_l2))) + (1 - self.lr) * l2_Ai2
                l2_Bi2 = self.lr * (np.fft.fft2(fi2_l2) * np.conjugate(np.fft.fft2(fi2_l2))) + (1 - self.lr) * l2_Bi2
                
                l2_Ai3 = self.lr * (G * np.conjugate(np.fft.fft2(fi3_l2))) + (1 - self.lr) * l2_Ai3
                l2_Bi3 = self.lr * (np.fft.fft2(fi3_l2) * np.conjugate(np.fft.fft2(fi3_l2))) + (1 - self.lr) * l2_Bi3
                
                l2_Ai4 = self.lr * (G * np.conjugate(np.fft.fft2(fi4_l2))) + (1 - self.lr) * l2_Ai4
                l2_Bi4 = self.lr * (np.fft.fft2(fi4_l2) * np.conjugate(np.fft.fft2(fi4_l2))) + (1 - self.lr) * l2_Bi4
                
            # Append Org/L1/L2/Integrated
            self.result_list_org.append(cv2.rectangle(current_frame.copy(), (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), (255, 0, 0), 2))
            self.result_list_l1.append(cv2.rectangle(current_frame.copy(), (pos_l1[0], pos_l1[1]), (pos_l1[0]+pos_l1[2], pos_l1[1]+pos_l1[3]), (0, 255, 0), 2))
            self.result_list_l2.append(cv2.rectangle(current_frame.copy(), (pos_l2[0], pos_l2[1]), (pos_l2[0]+pos_l2[2], pos_l2[1]+pos_l2[3]), (0, 0, 255), 2))
            
            integ = current_frame.copy()
            cv2.rectangle(integ, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), (255, 0, 0), 2)
            cv2.rectangle(integ, (pos_l1[0], pos_l1[1]), (pos_l1[0]+pos_l1[2], pos_l1[1]+pos_l1[3]), (0, 255, 0), 2)
            cv2.rectangle(integ, (pos_l2[0], pos_l2[1]), (pos_l2[0]+pos_l2[2], pos_l2[1]+pos_l2[3]), (0, 0, 255), 2)
            
            self.result_list_int.append(integ)
            
            
        #return self.result_list, self.window_list, self.gi_list
        return None
             

    def _get_diff(self, gi):
        max_value = np.max(gi)
        max_pos = np.where(gi == max_value)
        dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
        dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)

        return dx, dy

    def make_video(self, fps=30):
        
        h, w, _ = self.result_list_org[0].shape
        
        # Make Video : Origin
        out1 = cv2.VideoWriter("./result_org.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                              fps, (w, h))
        out2 = cv2.VideoWriter("./result_l1.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                              fps, (w, h))
        out3 = cv2.VideoWriter("./result_l2.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                              fps, (w, h))
        out4 = cv2.VideoWriter("./result_integrate.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                              fps, (w, h))
        
        for i in range(len(self.result_list_org)):
            out1.write(self.result_list_org[i])
            out2.write(self.result_list_l1[i])
            out3.write(self.result_list_l2[i])
            out4.write(self.result_list_int[i])
            
        out1.release(), out2.release(), out3.release(), out3.release()
        
        return None

    def _clip(self, clip_pos, pos, frame):
        clip_pos[0] = np.clip(pos[0], 0, frame.shape[1])
        clip_pos[1] = np.clip(pos[1], 0, frame.shape[0])
        clip_pos[2] = np.clip(pos[0]+pos[2], 0, frame.shape[1])
        clip_pos[3] = np.clip(pos[1]+pos[3], 0, frame.shape[0])
        
        return clip_pos.astype(np.int64)

    # get the ground-truth gaussian reponse...
    def _get_gauss_response(self, img, gt):
        # get the shape of the image..
        height, width = img.shape
        # get the mesh grid...
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        # get the center of the object...
        center_x = gt[0] + 0.5 * gt[2]
        center_y = gt[1] + 0.5 * gt[3]
        # cal the distance...
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.sigma)
        # get the response map...
        response = np.exp(-dist)
        # normalize...
        response = linear_mapping(response)
        return response

    # it will extract the image list 
    def _get_img_lists(self, img_path):
        frame_list = []
        for frame in os.listdir(img_path):
            if os.path.splitext(frame)[1] == '.jpg':
                frame_list.append(os.path.join(img_path, frame)) 
        return frame_list
    
