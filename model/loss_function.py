import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# 动态二分类交叉熵的focal loss
class DynamicFocalLossBCE(nn.Module):
    '''
    二分类交叉熵的focal loss
    '''
    def __init__(self, alpha: float=0.3, gamma: float=1.0, device: str='cuda'):
        '''
        alpha： 正样本被正确划分时的权重，此时，负样本的权重为1-alpha
        gamma: 损失放大系数
        '''
        super(DynamicFocalLossBCE, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
      
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs: Tensor, targets: Tensor):
        '''
        '''
        # 计算标准交叉熵
        bce_loss = self.BCELoss(inputs, targets)
       
        # 计算alpha，一行中1的个数除以行的总长度, 正样本的权重
        alpha = (1.0 - (torch.sum(targets, dim=-1) / targets.shape[-1])) * self.alpha
       
        for _ in range(len(targets.shape) - len(alpha.shape)):
            alpha = alpha.unsqueeze(dim=-1)

        
        # 计算正样本的权重
        pos_weight = torch.ones(targets.shape).to(self.device) * alpha

        # 控制正负样本的权重,aplha_t
        alpha_weight = torch.where(torch.eq(targets, 1), pos_weight, 1.0 - pos_weight)  #  1.0 - pos_weight 为负样本的权重

        # with_logits输出没有激活函数，要重新计算一遍
        inputs = torch.sigmoid(inputs)

        #  aplha_t *  (1-pt)^gamma)
        focal_weight  = torch.where(torch.eq(targets, 1), 1.0 - inputs, inputs)
        focal_weight = alpha_weight * torch.pow(focal_weight, self.gamma)

        # 根据公式计算最后的focal loss,
        # fl(pt) = - aplha_t * (1-pt)^gamma) * log(pt), 其中 -log(pt)=bce_loss
        loss =  focal_weight * bce_loss
        
        return loss

#  二分类交叉熵的focal loss
class FocalLossBCE(nn.Module):
    '''
    二分类交叉熵的focal loss
    '''
    def __init__(self, alpha: float=0.25, gamma: float=1.0, with_logits: bool=True, device: str='cuda'):
        '''
        alpha： 正样本被正确划分时的权重，此时，负样本的权重为1-alpha
        gamma: 损失放大系数
        with_logits：输出没有用激活函数
        '''
        super(FocalLossBCE, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.with_logits = with_logits

        if with_logits:
            self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.BCELoss = nn.BCELoss(reduction='none')

    def forward(self, inputs: Tensor, targets: Tensor):
        
        # 计算标准交叉熵
        bce_loss = self.BCELoss(inputs, targets)

        # 计算正负样本的权重
        pos_weight = torch.ones(targets.shape).to(self.device) * self.alpha     # 正样本的权重

        # 控制正负样本的权重,aplha_t
        alpha_weight = torch.where(torch.eq(targets, 1), pos_weight, 1.0 - pos_weight)  #  1.0 - pos_weight 为负样本的权重

        # with_logits输出没有激活函数，要重新计算一遍
        if self.with_logits:
            inputs = torch.sigmoid(inputs)

        #  aplha_t *  (1-pt)^gamma)
        focal_weight  = torch.where(torch.eq(targets, 1), 1.0 - inputs, inputs)
        focal_weight = alpha_weight * torch.pow(focal_weight, self.gamma)

        # 根据公式计算最后的focal loss,
        # fl(pt) = - aplha_t * (1-pt)^gamma) * log(pt), 其中 -log(pt)=bce_loss
        loss =  focal_weight * bce_loss
        
        return loss 

# 多分类交叉熵FocalLoss
class FocalLossCrossEntropy(nn.Module):
    '''
    多分类交叉熵FocalLoss
    '''
    def __init__(self, num_class: int, device: str, alpha: float=0.25, gamma: float=1.0, reduce: str='sum'):
        '''
        inputs: [batch_size, num_class, max_seq_len]
        tragets: [batch_size, max_seq_len]

        reduce: ['sum', 'mean','none']
                'sum' return: [batch_size, seq_len]
                'mean' return: mean loss of batch
                'none' return: [batch_size, num_class, seq_len]
        '''
        super(FocalLossCrossEntropy, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.num_class = num_class
        self.device = device
        self.reduce = reduce

    def forward(self, inputs: Tensor, targets: Tensor):
        '''
        inputs: [batch_size, num_class, max_seq_len]
        tragets: [batch_size, max_seq_len]
        '''
        prob = F.softmax(inputs, dim=1)

        # 生成target的on_hot向量
        traget_onehot = torch.zeros_like(inputs).to(self.device)
        traget_onehot = traget_onehot.scatter_(1, targets.unsqueeze(1), 1)
        
        pt = torch.sum(traget_onehot * prob, dim=1) + 1e-10
        log_pt = torch.log(pt)

        loss = - self.alpha * torch.pow(1 - pt, self.gamma) * log_pt

        if self.reduce == 'sum':
            loss = torch.sum(loss, dim=0)   # [batch_size, seq_len]
        elif self.reduce == 'mean':
            loss = torch.mean(loss)
        
        return loss


if __name__ == "__main__":
    device = 'cpu'
    # bs=2, class=[0,1,2,3], seq_len=5
    targets  = torch.FloatTensor([[1, 0, 0, 0], [1,1,0,0]]).to(device)
    outputs = torch.FloatTensor([[0.95, 0.05, 0.4, 0.6],[0.7,0.7,0.2,0.2]]).to(device)

    f_loss = DynamicFocalLossBCE(device=device, gamma=2.0)
    loss = f_loss(outputs, targets)
    print(loss)


    exit()
    outputs  = torch.randn((3, 4, 5))
    tragets = torch.LongTensor([[0,1,2,1,0],[0,3,0,1,2],[0,1,2,3,1]])
    FCELoss = FocalLossCrossEntropy(num_class=4, device=device, alpha=0.8, gamma=1.0, reduce='none')
    loss = FCELoss(outputs, tragets)
    print(loss)
    print(loss.mean())
    exit()

    targets  = torch.FloatTensor([1, 0, 1, 0]).cuda()
    outputs = torch.FloatTensor([0.95, 0.05, 0.4, 0.6]).cuda()

    f_loss =FocalLossBCE(device=device, alpha=0.25, gamma=1.0, with_logits=True)
    loss = f_loss(outputs, targets)
    print(loss)
      