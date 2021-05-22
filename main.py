import torch
import numpy as np
from config import Config
import fire


# 设置随机数种子
seed = 233
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# from model.p_so_model import Trainer, load_model_and_test
# from model.s_model import Trainer, load_model_and_test

class PytorchIE(object):
    def __init__(self, config: Config=None):
        super().__init__()

        # 加载配置文件
        if config is None:
            config = Config()

        self.config = config

        # 指定训练设备
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(config.cuda_device_number))
            torch.backends.cudnn.benchmark = True

        print('device: {}'.format(self.device))

    # =====================SP_O_MODEL===================
    def train_sp_o(self):
        from model.sp_o_model import Trainer

        print('train sp_o model')

        trainer = Trainer()
        trainer.train(self.config, self.device)

    def test_sp_o(self):
        from model.sp_o_model import load_model_and_test

        print('test sp_o model')

        torch.backends.cudnn.benchmark = False
        load_model_and_test(self.config, self.device)


    # =====================P_SO_MODEL===================
    def train_p_so(self):
        from model.p_so_model import Trainer

        print('train p_so model')

        trainer = Trainer()
        trainer.train(self.config, self.device)

    def test_p_so(self):
        from model.p_so_model import load_model_and_test

        print('test p_so model')
        
        torch.backends.cudnn.benchmark = False
        load_model_and_test(self.config, self.device)

if __name__ == "__main__":

    # 设置默认为FloatTensor
    torch.set_default_tensor_type(torch.FloatTensor)

    # 解析命令行参数，执行指定函数
    fire.Fire(component=PytorchIE())