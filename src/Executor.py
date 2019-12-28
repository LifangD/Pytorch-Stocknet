#!/usr/local/bin/python
import os
import Metrics as metrics
from DataPipe import DataPipe
from tools import save_checkpoints,load_checkpoint,load_checkpoint_optimizer,get_optimizer,xavier_init
from Model import StockNET
import torch.nn as nn
import time,torch

class Executor(nn.Module):

    def __init__(self, model_name, config,silence_step=200, skip_step=20,train_logger=None,tb_logger=None):
        super(Executor,self).__init__()
        model = {"stocknet":StockNET}
        self.config = config
        self.silence_step = silence_step
        self.skip_step = skip_step
        self.pipe = DataPipe()
        word_table = self.pipe.init_word_table()
        self.model = model[model_name](config,word_table)
        self.train_logger = train_logger
        self.tb_logger = tb_logger
        self.optimizer = get_optimizer(name=self.config.opt, params=self.model.parameters(), lr=self.config.lr)

    def train_and_dev(self,do_continue=False):
        device = "cuda" if self.config.use_cuda else "cpu"
        self.model.to(device)

        if do_continue:
            load_checkpoint_optimizer(self.model,self.optimizer,os.path.join(self.config.saved_path,"best_model.pt"))
        else:
            files = os.listdir(self.config.saved_path)
            for f in files:
                name = os.path.join(self.config.saved_path,f)
                os.remove(name)
                self.train_logger.info("remove file {}".format(f))

        step = 0
        score_history = [0.0]
        #t1 = time.time()
        epoch_size, epoch_n_acc = 0, 0.0

        for epoch in range(self.config.n_epochs):
            self.train_logger.info('------------------Epoch: {0}/{1} start------------------'.format(epoch+1, self.config.n_epochs))
            train_batch_loss_list = list()
            train_batch_gen = self.pipe.batch_gen(phase='train')
            for train_batch_dict in train_batch_gen:
                self.optimizer.zero_grad()
                train_result= self.model(train_batch_dict,current_step = step,phase="train")
                #self.train_logger.info("cost time after one batch:{:.2f} secs".format(time.time()-t1))
                #t1 = time.time()
                train_batch_y, train_batch_y_, train_batch_loss = train_result["y"],train_result["y_"],train_result["loss"]
                y = train_batch_y.detach().cpu().numpy()
                y_ = train_batch_y_.detach().cpu().numpy()
                tp, fp, tn, fn = metrics.create_confusion_matrix(y,y_, True)
                acc = (metrics.n_accurate(train_batch_y, train_batch_y_))/train_batch_dict["batch_size"]
                mcc = metrics.eval_mcc(tp, fp, tn, fn)
                try:
                    self.train_logger.info("step {}: tp = {}, fp = {}, tn = {}, fn = {}, acc = {:.4f}, mcc = {:.6f} ".format(step,tp, fp, tn, fn,acc,mcc))
                except Exception as e:
                    print(step,tp, fp, tn, fn,acc,mcc)
                    print(e)

                train_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config.clip)
                self.optimizer.step()
                epoch_size += train_batch_dict['batch_size']
                train_batch_loss_list.append(train_batch_loss)
                train_batch_n_acc = metrics.n_accurate(y=train_batch_y, y_=train_batch_y_)  # float
                epoch_n_acc += float(train_batch_n_acc)
                loss_val = train_batch_loss.item()
                if step%self.skip_step==0 and step>self.silence_step:
                    epoch_acc = epoch_n_acc / epoch_size
                    try:
                        self.train_logger.info("step/epoch: {}/{}, training samples: {}, train_loss = {:.4f} train_acc = {:.6f}".format(step,epoch+1, epoch_size, loss_val,epoch_acc))
                    except Exception as e:
                        print(e)
                        print(step,epoch+1, epoch_size, loss_val,epoch_acc)
                step += 1
            t2  = time.time()
            self.model.eval()
            eval_res = self.do_eval(phase="dev")
            self.model.train()
            eval_acc,eval_mcc,eval_size = eval_res["acc"],eval_res["mcc"],eval_res["size"]
            eval_time = time.time()-t2
            self.train_logger.info("epoch {},eval_time: {:.2f} secs, eval_samples: {}\n eval_acc = {:.4f}, eval_mcc = {:.6f} ".format(epoch,eval_time,eval_size,eval_acc,eval_mcc))

            if eval_acc >score_history[-1]:
                score_history.append(eval_acc)
                self.train_logger.info("new best model saved")
                save_checkpoints(self.model,self.optimizer,step,filename=os.path.join(self.config.saved_path,"best_model.pt"))


    def do_eval(self,phase):
        eval_batch_gen = self.pipe.batch_gen(phase=phase)
        eval_size,eval_n_acc = 0,0.0
        y_list = []
        y_list_= []
        #eval_step=0
        with torch.no_grad():
            for eval_batch_dict in eval_batch_gen:
                # eval_step+=1
                # if eval_step>3:
                #     break
                eval_result = self.model(eval_batch_dict,0,phase)
                eval_batch_y,eval_batch_y_ = eval_result["y"],eval_result["y_"]
                eval_batch_n_acc = metrics.n_accurate(eval_batch_y,eval_batch_y_)
                eval_n_acc += eval_batch_n_acc
                eval_size += float(eval_batch_dict["batch_size"])
                y_list.extend(eval_batch_y)
                y_list_.extend(eval_batch_y_)

        y_list = torch.stack(y_list,0).detach().cpu().numpy()
        y_list_ = torch.stack(y_list_,0).detach().cpu().numpy()
        acc = metrics.eval_acc(eval_n_acc,eval_size)
        tp,fp,tn,fn = metrics.create_confusion_matrix(y_list,y_list_,True)
        self.train_logger.info("eval tp = {}, fp = {}, tn = {}, fn = {}".format(tp,fp,tn,fn))
        mcc = metrics.eval_mcc(tp,fp,tn,fn)
        res = {"acc":acc,"mcc":mcc,"size":eval_size}
        return res

    def restore_and_test(self):
        device = "cuda" if self.config.use_cuda else "cpu"
        self.model.to(device)
        self.train_logger.info("begin evaluating on test set")
        load_checkpoint(self.model, os.path.join(self.config.saved_path, "best_model.pt"))
        eval_res  = self.do_eval(phase="test")
        acc,mcc,size = eval_res["acc"],eval_res["mcc"],eval_res["size"]
        try:
            self.train_logger.info("eval result on test set:\n size:{}, eval_acc: {:.4f}, eval_mcc: {:.6f}".format(size,acc,mcc))
        except Exception as e:
            print(size,acc,mcc)
            print(e)


