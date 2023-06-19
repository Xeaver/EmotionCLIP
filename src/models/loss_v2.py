import sys

import torch
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
if sys.platform == 'linux':
    import torch.distributed.nn


def gather_features(
        image_features,
        text_features,
        sentiment_score,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1
):
    all_sentiment_score = None
    # We gather tensors from all gpus
    if gather_with_grad and sys.platform == 'linux':
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        if sentiment_score is not None:
            all_sentiment_score = torch.cat(torch.distributed.nn.all_gather(sentiment_score), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
        if sentiment_score is not None:
            gathered_sentiment_score = [torch.zeros_like(sentiment_score) for _ in range(world_size)]
            dist.all_gather(gathered_sentiment_score, sentiment_score)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_sentiment_score[rank] = sentiment_score
            all_sentiment_score = torch.cat(gathered_sentiment_score, dim=0)
    return all_image_features, all_text_features, all_sentiment_score

def get_sentiment_weight(input_score,target_score):
    input_log_prob = F.log_softmax(input_score,dim=-1)
    target_log_score = F.log_softmax(target_score,dim=-1)
    # input_prob = input_log_prob.exp()
    target_prob = target_log_score.exp()
    # fix numerical problem using abs
    sentiment_weight = 1./(pairwise_kl_div(input_log_prob,target_prob).abs()) 
    sentiment_weight.fill_diagonal_(0)
    return sentiment_weight

def pairwise_kl_div(x,y):

    """https://discuss.pytorch.org/t/calculate-p-pair-wise-kl-divergence/131424
    (y * y.log()).sum (dim = 1) - torch.einsum ('ik, jk -> ij', x, y)
    x: input log_prob
    y: target prob
    
    """
    log_y = y.log()
    return (y * log_y).sum(dim = 1) - torch.einsum('ik, jk -> ij', x, y)

class ReweightedClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            sentiment_scale=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.sentiment_scale = sentiment_scale

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale, sentiment_score):
        device = image_features.device
        sentiment_weight = None
        if self.world_size > 1:
            all_image_features, all_text_features, all_sentiment_score = gather_features(
                image_features, text_features, sentiment_score,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
                if sentiment_score is not None:
                    sentiment_weight = get_sentiment_weight(sentiment_score,all_sentiment_score)
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
                if sentiment_score is not None:
                    sentiment_weight = get_sentiment_weight(all_sentiment_score,all_sentiment_score)
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
            if sentiment_score is not None:
                sentiment_weight = get_sentiment_weight(sentiment_score,sentiment_score)
        if sentiment_weight is not None:
            logits_per_image = logits_per_image - self.sentiment_scale * sentiment_weight
            logits_per_text = logits_per_text - self.sentiment_scale * sentiment_weight

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss


class DisimilarityLoss(nn.Module):

    def __init__(
            self,
    ):
        super().__init__()

    def forward(self, token_features):
        # token_features: [*,num_tokens,feat_dim]
        # device = image_features.device
        disimilarity = 1. - token_features[:,0:1,:] @ token_features[:,1:,:].T

        
        return disimilarity.mean()