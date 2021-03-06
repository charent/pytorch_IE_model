import torch
import torch.nn as nn

class EMA(object):
    '''
    对模型的权重进行滑动平均
    usage:
        model = resnet()
        model.train()
        ema = EMA(model, 0.9999)
        ....
        for img, lb in dataloader:
            loss = ...
            loss.backward()
            optim.step()
            ema.update_params() # apply ema
        evaluate(model)  # evaluate with original model as usual
        ema.apply_shadow() # copy ema status to the model
        evaluate(model) # evaluate the model with ema paramters
            ema.restore() # resume the model parameters
    '''
    def __init__(self, model: nn.Module, decay: float=0.999):
        '''
        decay: each parameter p should be computed as p_hat = decay * p + (1. - decay) * p_hat
        '''
        super().__init__()
        self.step = 0
        self.model = model
        self.decay = decay

        self.params = {}
        self.params_backup = {}

        self.best_params = {}
        
        # 注册
        self.shadow = self.get_model_params()

    def get_model_params(self):
        params = dict()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                params[n] = p.data.clone().detach()
        return params

    def update_params(self):
        decay = min(self.decay, (self.step + 1) / (self.step + 10))
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    new_average = decay * self.shadow[name] + (1.0 - decay) * param.data
                    self.shadow[name] = new_average.clone().detach()

        self.step += 1
    
    def apply_shadow(self):
        '''
        应用滑动平均
        '''
        for name, param in self.model.named_parameters():
           if param.requires_grad:
               self.params_backup[name] = param.data
               param.data = self.shadow[name]
    
    def save_best_params(self):
        self.best_params = self.get_model_params()
    
    def restore_best_params(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.best_params[name]
        
    def restore(self):
        '''
        恢复旧的权重
        '''
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.params_backup[name]
        self.params_backup = {}

if __name__ == '__main__':
    model = nn.Linear(5,5)
    ema = EMA(model, decay=0.9)
    
    inputs = torch.randn((2, 5), requires_grad=True)
    outs = model(inputs)
    # print(outs)

    print(model.state_dict())
    print(ema.shadow)
    ema.update_params()
    ema.apply_shadow()
    print(model.state_dict())
    