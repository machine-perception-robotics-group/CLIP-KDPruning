'''
 * Copyright (c) 2023, Dachuan Shi.
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * For full license text, see LICENSE.txt file in the repo root
 * By Dachuan Shi
'''
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from clip import clip
import utils
from utils import cosine_lr_schedule, print_params_and_flops
from data import create_dataset, create_sampler, create_loader

import io
# from petrel_client.client import Client
import math

from torch.cuda.amp import autocast as autocast

#可視化のため
import matplotlib.pyplot as plt
import seaborn as sns



def update_alpha_parameters(model, vision_layers, transformer_layers, p, pi, print_info=True):

    standarlization = lambda x, mean, std : (x - mean) / std

    alpha_grad_attn_vision = torch.stack([getattr(model.module.visual.transformer.resblocks, str(i)).attn.alpha.grad for i in range(vision_layers)])
    alpha_grad_attn_language = torch.stack([getattr(model.module.transformer.resblocks, str(i)).attn.alpha.grad for i in range(transformer_layers)])
    alpha_grad_attn = torch.cat([alpha_grad_attn_vision.view(-1), alpha_grad_attn_language.view(-1)])
    mean, std = torch.mean(alpha_grad_attn), torch.std(alpha_grad_attn)
    alpha_grad_attn_vision, alpha_grad_attn_language = standarlization(alpha_grad_attn_vision, mean, std), standarlization(alpha_grad_attn_language, mean, std)

    alpha_grad_mlp_vision = torch.stack([getattr(model.module.visual.transformer.resblocks, str(i)).alpha.grad for i in range(vision_layers)])
    alpha_grad_mlp_language = torch.stack([getattr(model.module.transformer.resblocks, str(i)).alpha.grad for i in range(transformer_layers)])
    alpha_grad_mlp = torch.cat([alpha_grad_mlp_vision.view(-1), alpha_grad_mlp_language.view(-1)])
    mean, std = torch.mean(alpha_grad_mlp), torch.std(alpha_grad_mlp)
    alpha_grad_mlp_vision, alpha_grad_mlp_language = standarlization(alpha_grad_mlp_vision, mean, std), standarlization(alpha_grad_mlp_language, mean, std)
    
    alpha_grad = torch.cat([alpha_grad_attn_vision.view(-1), alpha_grad_attn_language.view(-1), alpha_grad_mlp_vision.view(-1), alpha_grad_mlp_language.view(-1)])
    sorted_alpha_grad, indices = torch.sort(alpha_grad, descending=True)
    compression_weight = torch.ones_like(indices)
    compression_weight[indices < alpha_grad_attn.numel()] = 36 # 36 = 12 (number of heads) * [1 (weights of query) + 1 (weights of key) + 1 (weights of value)]
    threshold = sorted_alpha_grad[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*pi))]
    
    def update(module, grad):
        mask = ((grad <= threshold) | (grad <= torch.min(grad)))
        module.data.copy_(mask + (~mask)*(1 - pi/p))

    for i in range(vision_layers):
        update(getattr(model.module.visual.transformer.resblocks, str(i)).attn.alpha, alpha_grad_attn_vision[i])
        update(getattr(model.module.visual.transformer.resblocks, str(i)).alpha, alpha_grad_mlp_vision[i])
    for i in range(transformer_layers):
        update(getattr(model.module.transformer.resblocks, str(i)).attn.alpha, alpha_grad_attn_language[i])
        update(getattr(model.module.transformer.resblocks, str(i)).alpha, alpha_grad_mlp_language[i])

    if print_info:
        attn, mlp = [], []
        for i in range(vision_layers):
            attn.append(getattr(model.module.visual.transformer.resblocks, str(i)).attn.alpha.flatten())
            mlp.append(getattr(model.module.visual.transformer.resblocks, str(i)).alpha.flatten())
        for i in range(transformer_layers):
            attn.append(getattr(model.module.transformer.resblocks, str(i)).attn.alpha.flatten())
            mlp.append(getattr(model.module.transformer.resblocks, str(i)).alpha.flatten())
        print('Current compression ratio of attn: ', 1-torch.mean(torch.cat(attn)))
        print('Current compression ratio of mlp: ', 1-torch.mean(torch.cat(mlp)))
        print('Current compression ratio: ', pi)  

def same_modal_KD(image_Fs_teacher,text_Fs_teacher,image_Fs,text_Fs):
    mse_image = F.mse_loss(image_Fs_teacher,image_Fs)
    mse_text = F.mse_loss(text_Fs_teacher,text_Fs)
    L_same = 0.5*(mse_image + mse_text)
    return L_same
def different_modal_KD(i2t_sim_teacher,t2i_sim_teacher,i2t_sim,t2i_sim):
    mse_i2t = F.mse_loss(i2t_sim_teacher,i2t_sim)
    mse_t2i = F.mse_loss(t2i_sim_teacher,t2i_sim)
    L_diff = 0.5*(mse_i2t + mse_t2i)
    return L_diff

def train(model,teacher_model, data_loader, optimizer, epoch, device, config, search=False, interval=50, scaler=None, KD=False):

    vision_layers, transformer_layers = model.module.vision_layers, model.module.transformer_layers
    # train
    model.train()  
    teacher_model.eval()
    print(f"KD:{KD}")

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if search:
        metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_sp_attn', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_sp_mlp', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    """
    if KD:
        metric_logger.add_meter('loss_same', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_diff', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_relation', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    """
    header = 'Train Epoch: [{}]'.format(epoch) if not search else 'Search Epoch: [{}]'.format(epoch)
    #print_freq = 50
    print_freq = 5000
    len_data_loader = len(data_loader)
    total_steps = len_data_loader*config['max_epoch']

    for i,(image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)   
        idx = idx.to(device,non_blocking=True)   

        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))
        
        if scaler is not None:
            print("scaler is not None")
            with autocast():
                loss = model(image, caption, alpha=alpha, idx=idx)       
                if search:
                    sparsity_loss_attn, sparsity_loss_mlp = model.module.get_sparsity_loss()
                    metric_logger.update(loss_ita=loss.item()) 
                    metric_logger.update(loss_sp_attn=config['w_sp_attn'] * sparsity_loss_attn.item()) 
                    metric_logger.update(loss_sp_mlp=config['w_sp_mlp'] * sparsity_loss_mlp.item()) 
                    loss += config['w_sp_attn'] * sparsity_loss_attn + config['w_sp_mlp'] * sparsity_loss_mlp
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            #知識蒸留が有効な場合
            if KD:
                
                #学習ステップからマスクを更新するタイミングを測定
                setp_switch = epoch*len_data_loader+i
                mask_update_step = search and (setp_switch % interval == 0 or setp_switch == total_steps - 1)
                """損失，画像特徴，テキスト特徴，画像テキスト間の類似度，テキスト画像間の類似度を取得"""
                loss_teacher, image_Fs_teacher,text_Fs_teacher,i2t_sim_teacher,t2i_sim_teacher = teacher_model(image, caption, alpha=alpha, idx=idx,KD=KD, mask_update_step=mask_update_step) 
                i2t_sim_teacher_targets = i2t_sim_teacher
                t2i_sim_teacher_targets = t2i_sim_teacher
                i2t_sim_teacher_targets = F.softmax(i2t_sim_teacher_targets, dim=1)
                t2i_sim_teacher_targets = F.softmax(t2i_sim_teacher_targets, dim=1)
                #"""異なるモーダル間の知識蒸留（順位関係のみ）はforward文で計算"""
                loss, image_Fs,text_Fs,i2t_sim,t2i_sim= model(image, caption, alpha=alpha, idx=idx, KD=KD, mask_update_step=mask_update_step,sim_i2t_targets_log=i2t_sim_teacher_targets, sim_t2i_targets_log=t2i_sim_teacher_targets) 
                #metric_logger.update(loss_relation=loss.item())
                #"""同一モーダル間の知識蒸留（画像とテキストのMSE）"""
                L_same = same_modal_KD(image_Fs_teacher,text_Fs_teacher,image_Fs,text_Fs)
                #metric_logger.update(loss_same=L_same.item())
                #"""異なるモーダル間の知識蒸留（画像テキスト間のMSEとテキスト画像間のMSE）"""
                L_diff = different_modal_KD(i2t_sim_teacher,t2i_sim_teacher,i2t_sim,t2i_sim)
                #metric_logger.update(loss_diff=L_diff.item())
                #マスク更新ステップはlossに追加しない
                if mask_update_step==True:
                    print("mask_update_step")
                    
                #マスク更新ステップでない場合はlossに追加
                else:
                    #print("mask_update_step is False")
                    loss += L_same + L_diff
            #知識蒸留が無効な場合通常のlossを使用
            else:
                if i==11:    
                    print("KD is False")
                loss,image_Fs,text_Fs,i2t_sim,t2i_sim = model(image, caption, alpha=alpha, idx=idx, KD=KD)   

            if search:
                #print("add_sparsity_loss")
                sparsity_loss_attn, sparsity_loss_mlp = model.module.get_sparsity_loss()
                metric_logger.update(loss_ita=loss.item()) 
                metric_logger.update(loss_sp_attn=config['w_sp_attn'] * sparsity_loss_attn.item()) 
                metric_logger.update(loss_sp_mlp=config['w_sp_mlp'] * sparsity_loss_mlp.item()) 
                loss += config['w_sp_attn'] * sparsity_loss_attn + config['w_sp_mlp'] * sparsity_loss_mlp
            
            #print("optimizer.zero_grad")
            optimizer.zero_grad()
            #print("loss.backward")
            loss.backward()
            #print("optimizer.step")
            optimizer.step()    
        if args.debug:
            if i==11:
                break
            else:
                print(i)
        

        step = epoch*len_data_loader+i
        if search and (step % interval == 0 or step == total_steps - 1):
            pi = config['p']*((1-math.cos(math.pi*(step+1)/total_steps))/2)**(1/2)
            update_alpha_parameters(model, vision_layers, transformer_layers, config['p'], pi)
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


@torch.no_grad()
def evaluation(model, data_loader, device, config,search=False, ax_epoch=0, make_sim_matrix=False,teacher_model_env=0, min_val=0, max_val=1):
    # test
    model.eval() 
    ax_epoch_100_500 = ax_epoch
    ax_epoch_50_250 = ax_epoch
    ax_epoch_10_50 = ax_epoch
    
    print('Computing features for evaluation...')

    texts = data_loader.dataset.text   
    num_text = len(texts)
    print(f"num_text {num_text}")
    #print(f"texts {texts}")
    print(f"texts type {type(texts)}")
    text_bs = 256
    text_embeds = []  
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        if search:
            text_input = model.module.tokenize(text).to(device) 
            text_output = model.module.encode_text(text_input)
        else:
            text_input = model.tokenize(text).to(device) 
            text_output = model.encode_text(text_input)
        text_embed = text_output / text_output.norm(dim=1, keepdim=True)
        text_embeds.append(text_embed)   
    text_embeds = torch.cat(text_embeds,dim=0)
    text_embeds_log = text_embeds.clone().detach().float().cpu()

    image_embeds = []
    num_images = 0
    
    for image, img_id in data_loader: 
        image = image.to(device) 
        if search:
            image_feat = model.module.encode_image(image)
        else:
            image_feat = model.encode_image(image)
        image_embed = image_feat / image_feat.norm(dim=1, keepdim=True)
        image_embeds.append(image_embed)
        num_images += 1
    print(f"num_images {num_images}")
    image_embeds = torch.cat(image_embeds,dim=0)
    image_embeds_log = image_embeds.clone().detach().float().cpu()

    sims_matrix = image_embeds @ text_embeds.t()
    sims_matrix_log = sims_matrix.clone().detach().float().cpu()

    if make_sim_matrix:
        sims_matrix_log_copy = sims_matrix_log
        num_sims_matrix_log_copy = sims_matrix_log_copy.numpy()
        print(f"num_sims_matrix_log_copy {num_sims_matrix_log_copy.dtype}")
        torch_sims_matrix_log_copy = torch.from_numpy(num_sims_matrix_log_copy)
        plot_sims_matrix_log_copy = F.softmax(torch_sims_matrix_log_copy, dim=1)
        
        print(f"plot_sims_matrix_log_copy shape {plot_sims_matrix_log_copy.shape}")
        max_index = torch.argmax(plot_sims_matrix_log_copy, dim=1)
        if teacher_model_env == 1:
            min_val = plot_sims_matrix_log_copy.min()
            max_val = plot_sims_matrix_log_copy.max()
        else:
            min_val = min_val
            max_val = max_val
        print(f"min_val {min_val}")
        print(f"max_val {max_val}")
        plt.figure(figsize=(250,50))
        ax = sns.heatmap(plot_sims_matrix_log_copy,cmap='coolwarm', cbar=True,vmin=min_val,vmax=max_val)
        ax.set_xticks([])
        ax.set_yticks([])
        """
        for img_idx_ax,text_idx_ax in enumerate(max_index):
            ax.plot(text_idx_ax+0.5,img_idx_ax+0.5,'o',color='green',markersize=30)
        """
        if search:
            save_path = os.path.join(args.output_dir,f'cosine_similarity_matrix_epoch_{ax_epoch}.png')
            ax_epoch +=1
        else:
            if not search and teacher_model_env == 1:
                save_path = os.path.join(args.output_dir,'cosine_similarity_matrix_teacher.png')
                Flag_save_cosine_similarity_matrix = 0
            else:
                save_path = os.path.join(args.output_dir,f'cosine_similarity_matrix_finetuned_{ax_epoch}.png')
                ax_epoch +=1
        plt.savefig(save_path)
        plt.close()
       
       #100:500サンプルに限定
        limited_sims_matrix_log_copy_100_500 = plot_sims_matrix_log_copy[:100,:500]
        print(f"limited_sims_matrix_log_copy shape {limited_sims_matrix_log_copy_100_500.shape}")
        max_index = torch.argmax(limited_sims_matrix_log_copy_100_500, dim=1)
        if teacher_model_env == 1:
            min_val = limited_sims_matrix_log_copy_100_500.min()
            max_val = limited_sims_matrix_log_copy_100_500.max()
        else:
            min_val = min_val
            max_val = max_val
        print(f"min_val {min_val}")
        print(f"max_val {max_val}")
        plt.figure(figsize=(250,50))
        ax = sns.heatmap(limited_sims_matrix_log_copy_100_500,cmap='coolwarm', cbar=True,vmin=min_val,vmax=max_val)
        ax.set_xticks([])
        ax.set_yticks([])
        for img_idx_ax,text_idx_ax in enumerate(max_index):
            ax.plot(text_idx_ax+0.5,img_idx_ax+0.5,'o',color='green',markersize=15)
        if search:
            save_path = os.path.join(args.output_dir,f'cosine_similarity_matrix100_500_epoch_{ax_epoch}.png')
            ax_epoch_100_500 +=1
        else:
            if not search and teacher_model_env == 1:
                save_path = os.path.join(args.output_dir,'cosine_similarity_matrix100_500_teacher.png')
                Flag_save_cosine_similarity_matrix = 0
            else:
                save_path = os.path.join(args.output_dir,f'cosine_similarity_matrix100_500_finetuned_{ax_epoch}.png')
                ax_epoch_100_500 +=1
        plt.savefig(save_path)
        plt.close() 
    
        #50:250サンプルに限定
        limited_sims_matrix_log_copy_50_250 = plot_sims_matrix_log_copy[:50,:250]
        print(f"limited_sims_matrix_log_copy shape {limited_sims_matrix_log_copy_50_250.shape}")
        max_index = torch.argmax(limited_sims_matrix_log_copy_50_250, dim=1)
        #print(f"max_index {max_index}")
        if teacher_model_env == 1:
            min_val = limited_sims_matrix_log_copy_50_250.min()
            max_val = limited_sims_matrix_log_copy_50_250.max()
        else:
            min_val = min_val
            max_val = max_val
        print(f"min_val {min_val}")
        print(f"max_val {max_val}")


        plt.figure(figsize=(250,50))
        ax = sns.heatmap(limited_sims_matrix_log_copy_50_250,cmap='coolwarm', cbar=True,vmin=min_val,vmax=max_val)
        ax.set_xticks([])
        ax.set_yticks([])
        for img_idx_ax,text_idx_ax in enumerate(max_index):
            #print(f"text_idx_ax {text_idx_ax}")
            #print(f"img_idx_ax {img_idx_ax}")
            #print("do plot max_index")
            ax.plot(text_idx_ax+0.5,img_idx_ax+0.5,'o',color='green',markersize=30)
        """
        plt.xlabel('Text Embeddings')
        plt.ylabel('Image Embeddings')
        plt.title('Cosine Similarity Matrix')
        """
        if search:
            save_path = os.path.join(args.output_dir,f'cosine_similarity_matrix50_250_epoch_{ax_epoch}.png')
            ax_epoch_50_250 +=1
        else:
            if not search and teacher_model_env == 1:
                save_path = os.path.join(args.output_dir,'cosine_similarity_matrix50_250_teacher.png')
                Flag_save_cosine_similarity_matrix = 0
            else:
                save_path = os.path.join(args.output_dir,f'cosine_similarity_matrix50_250_finetuned_{ax_epoch}.png')
                ax_epoch_50_250 +=1
        plt.savefig(save_path)
        plt.close()

        #10:50サンプルに設定
        limited_sims_matrix_log_copy_10_50 = plot_sims_matrix_log_copy[:10,:50]
        print(f"limited_sims_matrix_log_copy shape {limited_sims_matrix_log_copy_10_50.shape}")
        max_index = torch.argmax(limited_sims_matrix_log_copy_10_50, dim=1)
        #print(f"max_index {max_index}")
        if teacher_model_env == 1:
            min_val = limited_sims_matrix_log_copy_10_50.min()
            max_val = limited_sims_matrix_log_copy_10_50.max()
        else:
            min_val = min_val
            max_val = max_val
        print(f"min_val {min_val}")
        print(f"max_val {max_val}")


        plt.figure(figsize=(250,50))
        ax = sns.heatmap(limited_sims_matrix_log_copy_10_50,cmap='coolwarm', cbar=True,vmin=min_val,vmax=max_val)
        ax.set_xticks([])
        ax.set_yticks([])
        for img_idx_ax,text_idx_ax in enumerate(max_index):
            #print(f"text_idx_ax {text_idx_ax}")
            #print(f"img_idx_ax {img_idx_ax}")
            #print("do plot max_index")
            ax.plot(text_idx_ax+0.5,img_idx_ax+0.5,'o',color='green',markersize=60)
        """
        plt.xlabel('Text Embeddings')
        plt.ylabel('Image Embeddings')
        plt.title('Cosine Similarity Matrix')
        """
        if search:
            save_path = os.path.join(args.output_dir,f'cosine_similarity_matrix10_50_epoch_{ax_epoch}.png')
            ax_epoch_10_50 +=1
        else:
            if not search and teacher_model_env == 1:
                save_path = os.path.join(args.output_dir,'cosine_similarity_matrix10_50_teacher.png')
                Flag_save_cosine_similarity_matrix = 0
            else:
                save_path = os.path.join(args.output_dir,f'cosine_similarity_matrix10_50_finetuned_{ax_epoch}.png')
                ax_epoch_10_50 +=1
        plt.savefig(save_path)
        plt.close()
    
    else:
        min_val = min_val
        max_val = max_val
    
    """コサイン類似度の関係性を定量的評価"""
    print("check shape cosine similarity matrix")
    print(f"sims_matrix shape {sims_matrix.shape}")
    print(f"sims_matrix_log shape {sims_matrix_log.shape}")
    #sims_matrix shape torch.Size([5000, 25010])
    #sims_matrix_log shape torch.Size([5000, 25010])
    

    return sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy(), text_embeds_log, image_embeds_log, sims_matrix_log, min_val, max_val ,ax_epoch

            
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result
"""===類似度を計算する関数==="""
def compute_similarity_metrics(teacher_embeds,student_embeds):
    #コサイン類似度の計算
    cosine_sim = F.cosine_similarity(teacher_embeds,student_embeds, dim=1)
    cosine_sim_mean = cosine_sim.mean().item()
    #CKAの計算
    cka_score = compute_cka(teacher_embeds, student_embeds)
    
    return cosine_sim_mean, cka_score
def compute_cka(features_x, features_y):
    K = features_x @ features_x.t()
    L = features_y @ features_y.t()
    H = torch.eye(K.size(0)).to(K.device) - (1 / K.size(0)) * torch.ones(K.size(0), K.size(0)).to(K.device)
    K_c = H @ K @ H
    L_c = H @ L @ H
    hsic = (K_c * L_c).sum()
    norm_x = (K_c * K_c).sum().sqrt()
    norm_y = (L_c * L_c).sum().sqrt()
    cka_score = hsic / (norm_x * norm_y)
    return cka_score

def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity_func(a - a.mean(1).unsqueeze(1),
                            b - b.mean(1).unsqueeze(1), eps)
def cosine_similarity_func(a, b, eps=1e-8):
    return (a*b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)

def feature_similarity_check(teacher_text_embeds, teacher_image_embeds, teacher_sims_matrix, text_embeds_log, image_embeds_log, sims_matrix_log):
    print("check shape")
    print(f"teacher_text_embeds shape {teacher_text_embeds.shape}")
    print(f"teacher_image_embeds shape {teacher_image_embeds.shape}")
    print(f"teacher_sims_matrix shape {teacher_sims_matrix.shape}")
    print(f"text_embeds_log shape {text_embeds_log.shape}")
    print(f"image_embeds_log shape {image_embeds_log.shape}")
    print(f"sims_matrix_log shape {sims_matrix_log.shape}")

    text_cosine_sim,text_cka_sim = compute_similarity_metrics(teacher_text_embeds,text_embeds_log)
    print(f"Text_Cosine_Similarity: {text_cosine_sim:.4f}")
    print(f"Text_CKA_Similarity: {text_cka_sim:.4f}")
    image_cosine_sim, image_cka_sim = compute_similarity_metrics(teacher_image_embeds,image_embeds_log)
    print(f"Image_Cosine_Similarity: {image_cosine_sim:.4f}")
    print(f"Image_CKA_Similarity: {image_cka_sim:.4f}")
    
    if sims_matrix_log.shape[0] == teacher_sims_matrix.shape[0]:
        cos_pearson_corr = pearson_correlation(teacher_sims_matrix,sims_matrix_log)
        cos_pearson_corr = cos_pearson_corr.mean()
        print(f"sim_matrix_pearson_correlation: {cos_pearson_corr:.4f}")
    else:
        print("sims_matrix_log and teacher_sims_matrix have different shapes")
    

def main(args, config, client):
    utils.init_distributed_mode(args)    

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    config['pretrained'] = args.pretrained
    config['w_sp_attn'] = args.w_sp_attn / args.world_size
    config['w_sp_mlp'] = args.w_sp_mlp  /args.world_size
    config['max_epoch'] = args.epoch
    config['init_lr'] = args.lr
    config['p'] = args.p
    ax_epoch = 0
    ax_epoch_100_500 = 0
    ax_epoch_50_250 = 0
    ax_epoch_10_50 = 0
    if not args.evaluate:
        print('Target compression ratio: {}%'.format(config['p']*100))

    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config, client)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   


    if not args.evaluate:
        """====生徒モデルと教師モデルを定義====="""
        print("Creating model for searching")
        if client is not None:
            search_model, preprocess = clip.load_from_client(name=config['pretrained'], device=device, search=True, client=client)
        else:
            #clipディレクトリのclip.pyにある関数を使用してモデルを読み込む
            search_model, preprocess = clip.load(name=config['pretrained'], device=device, search=True)
            """
            if args.debug:
                print("search_model",search_model)
                for name, param in search_model.named_parameters():
                    if "alpha" in name:
                        print(name,param.shape)
            """
            #教師モデルを定義
            teacher_model, teacher_preprocess = clip.load(name=config['pretrained'], device=device, search=True)
            """
            if args.debug:
                print("teacher_model",teacher_model)
            """
        search_model.tokenize = clip.tokenize
        search_model.copy_params()
        teacher_model.tokenize = clip.tokenize
        teacher_model.copy_params()
        #print_params_and_flops('retrieval_clip', search_model, device, config)
        search_model_without_ddp = search_model
        
        if args.distributed:
            search_model = torch.nn.parallel.DistributedDataParallel(search_model, device_ids=[args.gpu])
            search_model_without_ddp = search_model.module  
            
        if not args.amp:
            optimizer = torch.optim.AdamW(
                    params=[{'params':[param for name, param in list(search_model.named_parameters()) if not ('alpha' in name)]}], 
                    lr=config['init_lr'], 
                    weight_decay=config['weight_decay']
                    )
            
        else:
            optimizer = torch.optim.AdamW(
                    [{'params':[param for name, param in list(search_model.named_parameters()) if not ('alpha' in name)], 
                      'lr': config['init_lr'], 'weight_decay': config['weight_decay']},
                     {'params':[param for name, param in list(search_model.named_parameters()) if ('alpha' in name)], 
                      'lr': 0, 'weight_decay': 0}]
                    )
            
        make_sim_matrix = args.make_sim_matrix
        teacher_model_env = 1

        if not args.debug:
            print("teacher_model evaluation")
            score_test_i2t_teacher, score_test_t2i_teacher, teacher_text_embeds, teacher_image_embeds, teacher_sims_matrix, teacher_min_val, teacher_max_val ,ax_epoch = evaluation(teacher_model,test_loader,device,config,ax_epoch=ax_epoch,make_sim_matrix=make_sim_matrix,teacher_model_env=teacher_model_env)
            print("do itm_eval")
            test_result_teacher = itm_eval(score_test_i2t_teacher, score_test_t2i_teacher, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
            print(f"test_result_teacher {test_result_teacher}")
            torch.cuda.empty_cache()
            
        KD = args.KD

        print(f"KD {KD}")
        print("Start searching")
        scaler = torch.cuda.amp.GradScaler() if args.amp else None
        for epoch in range(0, config['max_epoch']):
            if args.evaluate:
                break
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train(search_model,teacher_model, train_loader, optimizer, epoch, device, config, search=True, interval=50 if config['dataset']=='flickr' else 200, scaler=scaler,KD=KD) 
            #枝刈り中のモデル性能を評価
            score_search_i2t, score_search_t2i, text_embeds_log, image_embeds_log, sims_matrix_log ,min_val, max_val,ax_epoch = evaluation(search_model, test_loader, device, config,search=True, ax_epoch=ax_epoch, make_sim_matrix=make_sim_matrix,min_val=teacher_min_val, max_val=teacher_max_val)
            print(f"min_val {min_val}")
            print(f"max_val {max_val}")
            print("do itm_eval")
            search_result = itm_eval(score_search_i2t, score_search_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
            print(f"search_result {search_result}")
            torch.cuda.empty_cache()
            #特徴表現の維持度を評価
            feature_similarity_check(teacher_text_embeds, teacher_image_embeds, teacher_sims_matrix, text_embeds_log, image_embeds_log, sims_matrix_log)

            #枝刈り中のモデルを保存
            if not args.distributed or dist.get_rank() == 0:
                save_obj_search_model = {
                    'model':search_model.module.state_dict(),
                    'epoch':epoch,
                }
                save_path = os.path.join(args.output_dir,f"model_search_{epoch}.pth")
                torch.save(save_obj_search_model, save_path)
                print(f"model_search_{epoch}.pth saved")
            
        dist.barrier()   
        search_model.module.print_compression_statistics()

        print("Creating model for training")
        if client is not None:
            model, preprocess = clip.load_from_client(name=config['pretrained'], device=device, client=client)
        else:
            model, preprocess = clip.load(name=config['pretrained'], device=device)
        msg = model.load_state_dict(search_model_without_ddp.state_dict(), strict=False)
        model.tokenize = clip.tokenize
        model.compress(search_model_without_ddp)
    else:
        print("Creating model for evaluation")
        if client is not None:
            model, preprocess = clip.load_from_client(name=config['pretrained'], device=device, client=client, evaluate=True)
        else:
            model, preprocess = clip.load(name=config['pretrained'], device=device, evaluate=True)
        model.tokenize = clip.tokenize
        model.prune_if_compressed(client, config['pretrained'])
        model = model.to(device)  

    model.copy_params()
    #print_params_and_flops('retrieval_clip', model, device, config)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay']) 
    best = 0
    best_epoch = 0

    print("Start training")
    scaler = torch.cuda.amp.GradScaler() if (not args.evaluate and args.amp) else None
    for epoch in range(0, config['max_epoch']):    
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            
            train_stats = train(model, teacher_model, train_loader, optimizer , epoch, device, config, scaler=scaler )

        score_val_i2t, score_val_t2i, text_embeds_val, image_embeds_val, sims_matrix_val, min_val, max_val,ax_epoch = evaluation(model_without_ddp, val_loader, device, config, min_val=teacher_min_val, max_val=teacher_max_val)
        score_test_i2t, score_test_t2i, text_embeds_test, image_embeds_test, sims_matrix_test, min_val, max_val,ax_epoch = evaluation(model_without_ddp, test_loader, device, config, make_sim_matrix=make_sim_matrix, min_val=teacher_min_val, max_val=teacher_max_val)
        print(f"min_val {min_val}")
        print(f"max_val {max_val}")
        #特徴表現を評価
        feature_similarity_check(teacher_text_embeds, teacher_image_embeds, teacher_sims_matrix, text_embeds_test, image_embeds_test, sims_matrix_test)
    
        if utils.is_main_process():  
      
            val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
            print(val_result)
                                
            if val_result['r_mean']>best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    # 'config': config,
                    # 'epoch': epoch,
                }
                if client is not None:
                    with io.BytesIO() as f:
                        torch.save(save_obj, f)
                        f.seek(0)
                        client.put(os.path.join('s3://BucketName/ProjectName', args.output_dir, 'checkpoint_best.pth'), f)
                else:
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                best = val_result['r_mean']        
                best_epoch = epoch  
                
                test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
                print(test_result)
            
            if args.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},                  
                            }
                # with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                #     f.write(json.dumps(log_stats) + "\n")     
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},  
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                # with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                #     f.write(json.dumps(log_stats) + "\n")   
            print("LOG: ", log_stats)
                    
        if args.evaluate: 
            break

        dist.barrier()     
        torch.cuda.empty_cache()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')        
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--use_ceph', action='store_true')  
    parser.add_argument('--pretrained', default='pretrained/clip_large_retrieval_flickr.pth', type=str)
    parser.add_argument('--w_sp_attn', default=(22/15)*8e-3, type=float, help='regularization coefficient for attn')
    parser.add_argument('--w_sp_mlp', default=2e-4, type=float, help='regularization coefficient for mlp')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--epoch', default=12, type=int, help='number of epoches')
    parser.add_argument('--p', default=0.5, type=float, help='total compression ratio')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--KD', action='store_true')
    parser.add_argument('--make_sim_matrix', action='store_true')
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    if not args.use_ceph:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
        client=None
    else:
        client = Client('~/petreloss.conf', enable_mc=True)
        client.put(os.path.join('s3://BucketName/ProjectName', args.output_dir, 'config.yaml'), yaml.dump(config))
    client=None
    
    main(args, config, client)