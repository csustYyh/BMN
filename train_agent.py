import os
from os import path
import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network import STCN
from dataset.davis_test_dataset import DAVISTestDataset, DAVISTrainAgentDataset
from util.tensor_util import unpad
from inference_core import InferenceCore
from progressbar import progressbar
from model.agent import Agent 
from model.repaly_buffer import ReplayBuffer
import datetime
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("tensorboard")
""" 
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='') # VOS model path
parser.add_argument('--davis_path', default='')
parser.add_argument('--output', default='./output')
parser.add_argument('--split', help='val/testdev', default='train_agent')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument("-total_iter", type=int, help="total iter num",default=200000)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true') 
args = parser.parse_args()

davis_path = args.davis_path
out_path = os.path.join(args.output, args.split)

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(davis_path + '/trainval/Annotations/480p/blackswan/00000.png')).getpalette()

torch.autograd.set_grad_enabled(False)

# Setup Dataset
if args.split == 'val':
    test_dataset = DAVISTestDataset(davis_path+'/trainval', imset='2017/val.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
elif args.split == 'testdev':
    test_dataset = DAVISTestDataset(davis_path+'/test-dev', imset='2017/test-dev.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
elif args.split == 'train_agent':
    test_dataset = DAVISTrainAgentDataset(davis_path+'/trainval', imset='2017/train.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
else:
    raise NotImplementedError

# Load our checkpoint
top_k = args.top
prop_model = STCN().cuda().eval()

# Performs input mapping such that stage 0 model can be loaded
prop_saved = torch.load(args.model)
for k in list(prop_saved.keys()):
    if k == 'value_encoder.conv1.weight':
        if prop_saved[k].shape[1] == 4:
            pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
            prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
prop_model.load_state_dict(prop_saved)


# agent intialization
# device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
agent = Agent(device='cuda')
replay_buffer = ReplayBuffer(capacity = 4000)

loader_iter = iter(test_loader)
count = 1
episode_return = 0
# Start train agent
for iter_ in range(args.total_iter):
    try:
        data = next(loader_iter)
    except:
        loader_iter = iter(test_loader)
        data = next(loader_iter)
    with torch.cuda.amp.autocast(enabled=args.amp): 
        rgb = data['rgb'].cuda()
        msk = data['gt'][0].cuda()
        if (msk.shape[0] == 0):
            continue
        info = data['info']
        name = info['name'][0]
        k = len(info['labels'][0])
        size = info['size_480p']
        torch.cuda.synchronize()
        process_begin = time.time()

        processor = InferenceCore(agent, prop_model, rgb, k, top_k=top_k, 
                        mem_every=args.mem_every, include_last=args.include_last)
        processor.interact(msk[:,0], 0, rgb.shape[1])

        # Do unpad -> upsample to original size 
        out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
        out_masks_compare = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
        out_masks_final = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
        out_masks_full = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
        
        for ti in range(processor.t):
            if ti == 10: # only the last frame in trainning clip
                prob = unpad(processor.prob[:,ti], processor.pad)
                prob_compare = unpad(processor.prob_compare[:,ti], processor.pad)
                prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
                prob_compare = F.interpolate(prob_compare, size, mode='bilinear', align_corners=False)
                
                prob_final = unpad(processor.prob_final[:,ti], processor.pad)
                prob_full = unpad(processor.prob_full[:,ti], processor.pad)
                prob_final = F.interpolate(prob_final, size, mode='bilinear', align_corners=False)
                prob_full = F.interpolate(prob_full, size, mode='bilinear', align_corners=False)
                
                out_masks[ti] = torch.argmax(prob, dim=0)
                out_masks_compare[ti] = torch.argmax(prob_compare, dim=0)
                out_masks_final[ti] = torch.argmax(prob_final, dim=0)
                out_masks_full[ti] = torch.argmax(prob_full, dim=0)
                
        from util.tensor_util import compute_tensor_iu, compute_tensor_iou, compute_rl_reward
        iou = compute_tensor_iou(out_masks[10]>0.5, msk[:,10]>0.5)
        iou_compare = compute_tensor_iou(out_masks_compare[10]>0.5, msk[:,10]>0.5)
        
        iou_final = compute_tensor_iou(out_masks_final[10]>0.5, msk[:,10]>0.5)
        iou_full = compute_tensor_iou(out_masks_full[10]>0.5, msk[:,10]>0.5)
        
        reward_first = compute_rl_reward(iou, iou_compare)     # reward
        reward_second = compute_rl_reward(iou_final, iou_full) # reward_f
        
        episode_return += (reward_first + reward_second)

        replay_buffer.add(processor.state[:, :, :30, :54], processor.action, reward_first, processor.next_state[:, :, :30, :54], 0)
        replay_buffer.add(processor.next_state[:, :, :30, :54], processor.next_action, reward_second, None, 1)

        
        if count > 10:
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(8)
            transition_dict = {'states': b_s, 'actions': b_a, 'rewards': b_r, 'next_states': b_ns, 'done': b_d}
            loss = agent.rl_update(transition_dict, count)

        if count % 60 == 0:
            print('episode_return:[{}]\t'.format(episode_return))
            writer.add_scalar("episode_return", episode_return, count // 60)
            episode_return = 0
        if count % 100 == 0:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("train steps:[{}] - Current Time: {}\t".format(count, current_time))
        count += 1
        del rgb
        del msk
        del processor


writer.close()