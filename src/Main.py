#!/usr/local/bin/python
from Executor import Executor
import os,random,torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ConfigLoader import config_model,path_parser
from tools import DicToObj,xavier_init
from Log import Tensorboard_Logger,Logger
import numpy as np
def generate_name_path(config,path):
    # model name
    name_pattern_max_n = 'days-{0}.msgs-{1}-words-{2}'
    name_max_n = name_pattern_max_n.format(config.max_n_days, config.max_n_msgs, config.max_n_words)

    name_pattern_input_type = 'word_embed-{0}.vmd_in-{1}'
    name_input_type = name_pattern_input_type.format(config.word_embed_type, config.variant_type)

    name_pattern_key = 'alpha-{0}.anneal-{1}.rec-{2}'
    name_key = name_pattern_key.format(config.alpha, config.kl_lambda_anneal_rate, config.vmd_rec)

    name_pattern_train = 'batch-{0}.opt-{1}.lr-{2}-drop-{3}-cell-{4}_v2'
    name_train = name_pattern_train.format(config.batch_size, config.opt, config.lr, config.dropout_mel_in, config.mel_cell_type)

    name_tuple = (config.mode, name_max_n, name_input_type, name_key, name_train)
    model_name = '_'.join(name_tuple)
    saved_path = os.path.join(path.checkpoints, model_name)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    config.saved_path = saved_path
    return config

if __name__ == '__main__':
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_gpu = torch.cuda.device_count()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if n_gpu>0:
        torch.cuda.manual_seed(seed)

    silence_step = 0 # 从哪步开始可以进行evaluate
    skip_step = 20 # 每隔多少步做evaluate
    config = DicToObj(**config_model)
    formatter = '%(asctime)s %(levelname)s %(message)s'
    config = generate_name_path(config,path_parser)

    train_logger = Logger(filename = config.saved_path+ '/train.log', fmt=formatter).logger
    tb_logger = Tensorboard_Logger(config.saved_path)
    exe = Executor(model_name="stocknet",config=config,silence_step=silence_step, skip_step=skip_step,train_logger=train_logger,tb_logger=tb_logger)
    exe.apply(xavier_init)
    exe.train_and_dev(do_continue=False)
    exe.restore_and_test()

