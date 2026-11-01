"""Coalesce tiny pose tensors into one transfer in each direction."""
import numpy as np
import torch


def filter_parameters(body,flame,parameter_filter,timestamp):
    groups=(body,flame)
    entries=[(group,key,value) for group,values in enumerate(groups) for key,value in values.items() if torch.is_tensor(value)]
    if not entries:
        filtered=parameter_filter.update({}, {},float(timestamp))
        return {},{},*filtered,dict(body),dict(flame)
    first=entries[0][2]
    packed=all(v.dtype==torch.float32 and v.device==first.device for _,_,v in entries)
    raw=[{},{}]
    if packed:
        values=torch.cat([v.reshape(-1) for _,_,v in entries]).detach().cpu().numpy()
        offset=0
        for group,key,value in entries:
            size=value.numel();raw[group][key]=values[offset:offset+size].reshape(tuple(value.shape));offset+=size
    else:
        # Preserve the generic path if an upstream output's dtype/device differs.
        for group,key,value in entries:raw[group][key]=value.detach().cpu().numpy()
    filtered=list(parameter_filter.update(*raw,float(timestamp)))
    result=[dict(body),dict(flame)]
    if packed:
        uploaded=torch.from_numpy(np.concatenate([filtered[g][k].reshape(-1) for g,k,_ in entries])).to(first.device)
        offset=0
        for group,key,value in entries:
            size=value.numel();result[group][key]=uploaded[offset:offset+size].reshape(value.shape);offset+=size
    else:
        for group,key,value in entries:result[group][key]=torch.from_numpy(filtered[group][key]).to(value.device)
    return (*raw,*filtered,*result)
