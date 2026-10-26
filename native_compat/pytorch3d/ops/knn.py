from collections import namedtuple

import torch

_KNN = namedtuple("_KNN", "dists idx knn")


def knn_points(p1, p2, lengths1=None, lengths2=None, norm=2, K=1, version=-1, return_nn=False, return_sorted=True):
    if lengths1 is not None or lengths2 is not None:
        raise NotImplementedError("The Windows inference KNN path requires dense inputs.")
    if K > p2.shape[1] or norm not in (1, 2):
        raise ValueError("Invalid KNN parameters")
    # Chunk query points to avoid constructing an unbounded pairwise tensor.
    distances, indices = [], []
    for chunk in p1.split(1024, dim=1):
        distance = torch.cdist(chunk, p2, p=float(norm))
        if norm == 2:
            distance = distance.square()
        values, index = distance.topk(K, dim=-1, largest=False, sorted=return_sorted)
        distances.append(values)
        indices.append(index)
    dists, idx = torch.cat(distances, 1), torch.cat(indices, 1)
    nearest = None
    if return_nn:
        nearest = p2[torch.arange(p2.shape[0], device=p2.device)[:, None, None], idx]
    return _KNN(dists, idx, nearest)
