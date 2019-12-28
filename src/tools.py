import os
import torch
import json
import torch.nn as nn
def load_checkpoint_optimizer(model, optimizer, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        step = checkpoint['step']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (step {})".format(filename, step))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    

def load_checkpoint(model,filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model


def save_checkpoints(model,optimizer,step,filename):
    state = {'step': step , 'state_dict': model.state_dict(),'optimizer': optimizer.state_dict(),}
    torch.save(state, filename)
    #print("saving model {} at step {}".format(filename,step))

def save_config(config, path, verbose=True):
    with open(path, 'w') as outfile:
        json.dump(config, outfile, indent=2)
    if verbose:
        print("Config saved to file {}".format(path))
    return config

def get_optimizer(name,params,lr):
    if name=="sgd":
        return torch.optim.SGD(params=params,lr=lr)
    elif name=="adam":
        return torch.optim.Adam(params=params,lr=lr)
    else:
        print("wrong optimizer name.")

class DicToObj:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def generate_mask(X,X_len):

    X_len = X_len.view(-1,1)
    max_len = X.size(-1)
    if X_len.is_cuda:
        mask = (torch.arange(max_len)[None,:]).cuda()<X_len[:,None]
    else:
        mask = (torch.arange(max_len)[None, :])< X_len[:, None]
    mask = mask.view(X.size())
    return mask

def kl_lambda_annealing(current_step,start_step,annealing_rate):
    if current_step<start_step:
        return 0
    else:
        return min(1,annealing_rate*current_step)

def batched_index_select(input,dim,index):
    if len(index.shape)==1:
        index = index.unsqueeze(-1)
    for ii in range(1,len(input.shape)):
        if ii!=dim:
            index  = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0]=-1
    expanse[dim]=-1
    index_e = index.expand(expanse)
    return torch.gather(input,dim,index_e)


def gussian_kl_divergence(mu1,logstd1,mu2,logstd2):
    var1 = torch.exp(2*logstd1)
    var2 = torch.exp(2*logstd2)
    kl = torch.sum(logstd2-logstd1+(var1+torch.mul(mu1-mu2,mu1-mu2))/(2*var2)-0.5,-1)
    return kl


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)





if __name__ =="__main__":
    import numpy as np
    X = np.random.random([2,4,6]).round(1) * 2 + 3
    X = torch.from_numpy(X)
    X_len = torch.sum(X<3.4,-1)
    print(X_len)
    mask = generate_mask(X,X_len)
    print(mask)