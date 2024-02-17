import numpy as np
import torch
import torch.nn.functional as F

# ALL THESE LOSSES ARE WORKING WITH A SINGLE IMAGE (192 x 256). BATCH SIZE IS 1.

def contrastive_loss(embedding, num_planes, segmentation, device):
    """Uses the center of each plane as an anchor, yielding 1 positive and num_planes-1 negative anchors per pixel.
    """
    b, c, h, w = embedding.size() # 1 x 2 x 192 x 256

    # Get rid of first dimension (batch)
    num_planes = num_planes.numpy()[0]
    embedding = embedding[0]
    segmentation = segmentation[0]
    embeddings = []

    # Select embedding with segmentation
    for i in range(num_planes): # avoid non-planar region
        feature = torch.transpose(torch.masked_select(embedding, segmentation[i, :, :].view(1, h, w)).view(c, -1), 0, 1) # num_pixels_i x 2 
        embeddings.append(feature)

    # Compute the center of each plane
    centers = []
    for feature in embeddings:
        center = torch.mean(feature, dim=0).view(1, c)
        centers.append(center)
    centers = torch.cat(centers) # num_planes x 2

    # Inner product of each pixel with each plane center
    centers = centers.unsqueeze(1)
    embedding = embedding.view(-1, c).unsqueeze(0)
    logits = embedding * centers
    logits = logits.sum(2) # num_planes x h*w

    segmentation = segmentation[:num_planes, :, :].view(-1, h*w) # mask each pixel w.r.t. segmentation

    # Only take the dot product of the corresponding center
    positive = logits * segmentation.to(torch.float)

    # Consider only planar pixels
    indices = segmentation.sum(dim=0).nonzero()
    positive = torch.index_select(positive, 1, indices.squeeze())  # num_planes x num_planar
    logits = torch.index_select(logits, 1, indices.squeeze()) # num_planes x num_planar

    exp_logits = torch.exp(logits)

    # positive.sum(0) has a single non-zero entry per sum
    log_prob = positive.sum(0) - torch.log(exp_logits.sum(0))

    loss = - log_prob
    return torch.mean(loss.to(device)), torch.mean(loss.to(device)), torch.mean(loss.to(device))


def contrastive_loss_anchors(embedding, num_planes, segmentation, device):
    """Uses num_samples (m) + 1 (centers) anchors per plane. Anchors are obtained by subsampling each plane (feature) 
    m times and computing its subcenter. The length of the subsample is size times the whole plane. This has m+1 positive
    anchors and (m+1)*(num_planes-1) negative ones.
    """
    b, c, h, w = embedding.size() # 1 x 2 x 192 x 256

    # Get rid of first dimension (batch)
    num_planes = num_planes.numpy()[0]
    embedding = embedding[0]
    segmentation = segmentation[0]

    # Anchor parameters
    num_samples = 20 # additional anchors per plane (m)
    size = 0.2 # proportion of the plane to subsample for the new center

    anchors = []

    # Select embedding with segmentation
    # Compute centers and subcenters
    for i in range(num_planes): # avoid non-planar region
        feature = torch.transpose(torch.masked_select(embedding, segmentation[i, :, :].view(1, h, w)).view(c, -1), 0, 1)
        length = feature.shape[0]

        center = torch.mean(feature, dim=0).view(1, c)
        subcenters = torch.cat([torch.mean(feature[torch.randperm(length)[:int(length*size)], :], dim=0).view(1, c) for _ in range(num_samples)])

        anchors.append(torch.cat((center, subcenters)))

    anchors = torch.stack(anchors) # num_planes x m+1 x c

    # Inner product of each pixel with each of the centers
    embedding = embedding.view(-1,c).unsqueeze(1).unsqueeze(1)
    anchors = anchors.unsqueeze(0).unsqueeze(0)
    logits = embedding*anchors # 1 x h*w x num_planes x m+1 x c
    logits = logits.sum(-1).squeeze(0) # h*w x num_planes x m+1

    segmentation = torch.transpose(segmentation[:num_planes, :, :].view(-1, h*w), 0, 1) # mask each pixel w.r.t. segmentation

    # Only take the dot product of the corresponding positive centers
    positive = logits * segmentation.to(torch.float).unsqueeze(-1)
    
    # Consider only planar pixels
    indices = segmentation.sum(dim=1).nonzero().squeeze()
    positive = positive[indices] # num_planar x num_planes x m+1
    logits = logits[indices].view(indices.shape[0], -1).unsqueeze(-1) # num_planar x num_planes*m+1 x 1
    
    exp_logits = torch.exp(logits)

    
    # positive.sum(1) has a single non-zero entry per sum
    log_prob = positive.sum(1) - torch.log(exp_logits.sum(1)) # num_planar x m+1
    mean_pos_log_prob = log_prob.mean(1)

    loss = - mean_pos_log_prob

    return torch.mean(loss.to(device)), torch.mean(loss.to(device)), torch.mean(loss.to(device))




def contrastive_loss_anchors_neg(embedding, num_planes, segmentation, device):
    """Uses num_samples (m) + 1 (centers) anchors per plane, but only for the negative. Anchors are obtained by subsampling 
    each plane (feature) m times and computing its subcenter. The length of the subsample is size times the whole plane. 
    This has 1 positive anchor (the true plane center) and (m+1)*(num_planes-1) negative ones.
    """
    b, c, h, w = embedding.size() # 1 x 2 x 192 x 256

    # Get rid of first dimension (batch)
    num_planes = num_planes.numpy()[0]
    embedding = embedding[0]
    segmentation = segmentation[0]

    # Anchor parameters
    num_samples = 20 # additional anchors per plane (m)
    size = 0.2 # proportion of the plane to subsample for the new center

    anchors = []


    # Select embedding with segmentation
    # Compute centers and subcenters
    for i in range(num_planes): # avoid non-planar region
        feature = torch.transpose(torch.masked_select(embedding, segmentation[i, :, :].view(1, h, w)).view(c, -1), 0, 1)
        length = feature.shape[0]

        center = torch.mean(feature, dim=0).view(1, c)
        subcenters = torch.cat([torch.mean(feature[torch.randperm(length)[:int(length*size)], :], dim=0).view(1, c) for _ in range(num_samples)])

        anchors.append(torch.cat((center, subcenters)))

    anchors = torch.stack(anchors) # num_planes x m+1 x c   

    
    # Inner product of each pixel with each of the centers
    embedding = embedding.view(-1,c).unsqueeze(1).unsqueeze(1)
    anchors = anchors.unsqueeze(0).unsqueeze(0)
    logits = embedding*anchors # 1 x h*w x num_planes x m+1 x c
    logits = logits.sum(-1).squeeze(0) # h*w x num_planes x m+1

    segmentation = torch.transpose(segmentation[:num_planes, :, :].view(-1, h*w), 0, 1) # mask each pixel w.r.t. segmentation

    # Only take the dot product of the actual positive center
    positive = logits[:,:,0] * segmentation.to(torch.float)  


    # Consider only planar pixels
    indices = segmentation.sum(dim=1).nonzero().squeeze()
    positive = positive[indices] # num_planar x num_planes
    logits = logits[indices].view(indices.shape[0], -1) # num_planar x num_planes*m+1 
    
    exp_logits = torch.exp(logits)

    log_prob = positive.sum(1) - torch.log(exp_logits.sum(1))
    loss = - log_prob

    return torch.mean(loss.to(device)), torch.mean(loss.to(device)), torch.mean(loss.to(device))
