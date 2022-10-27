import torch 
from collections import OrderedDict

new_state_dict = OrderedDict()
state_dict = torch.load('pretrained/swin_base_patch244_window877_kinetics400_22k.pth')
for k, v in state_dict['state_dict'].items():
    k = k.replace('backbone.', '')   # remove prefix backbone.
    #k = k.replace('attn.qkv.weight', 'attn.attn.in_proj_weight')
    #k = k.replace('attn.qkv.bias', 'attn.attn.in_proj_bias')
    #k = k.replace('attn.proj.weight', 'attn.attn.out_proj.weight')
    #k = k.replace('attn.proj.bias', 'attn.attn.out_proj.bias')
    new_state_dict[k] = v

torch.save(new_state_dict, 'pretrained/swin_base_patch244_window877_kinetics400_22k_fixed.pth')