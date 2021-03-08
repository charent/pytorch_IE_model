import torch
import numpy as np
from config import Config

from model.p_so_model import Trainer, load_model_and_evalute
# from model.sp_o_model import Trainer, load_model_and_evalute

seed = 233
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    torch.set_default_tensor_type(torch.FloatTensor)
    
    # 加载配置文件
    config = Config()
    
    # 指定训练设备
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.cuda_device_number))
        torch.backends.cudnn.benchmark = True

    print('device: {}'.format(device))

    trainer = Trainer()
    trainer.train(config, device)

    # load_model_and_evalute(config, device)