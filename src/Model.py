from __future__ import print_function
import torch.nn as nn
from Module import Embedding,MsgEncoder,MsgPriceAgg,VariationalMovDecoder,TemporalAttDecoder,CalculateJointLoss
import torch
from tools import batched_index_select
class StockNET(nn.Module):
    def __init__(self,config,word_table):
        super(StockNET,self).__init__()
        self.config= config
        self.emb= Embedding(word_table)
        self.msg_encoder = MsgEncoder(config)
        self.msg_price_aggregator = MsgPriceAgg(config)
        self.variational_decoder = VariationalMovDecoder(config)
        self.temporal_att_decoder = TemporalAttDecoder(config)
        self.loss_builder = CalculateJointLoss(config)

    def forward(self,batch_dict,current_step,phase):
        device = "cuda" if self.config.use_cuda else "cpu"
        T = torch.LongTensor(batch_dict["T_batch"]).to(device)
        n_words = torch.LongTensor(batch_dict['n_words_batch']).to(device)
        n_msgs = torch.LongTensor(batch_dict['n_msgs_batch']).to(device)
        y_ = torch.FloatTensor(batch_dict['y_batch']).to(device)
        price = torch.FloatTensor(batch_dict["price_batch"]).to(device)
        word = torch.LongTensor(batch_dict["word_batch"]).to(device)
        ss_index = torch.LongTensor(batch_dict["ss_index_batch"]).to(device)

        emb_x = self.emb(word)
        ht_s = self.msg_encoder(emb_x,n_words,ss_index)
        ht_price_s = self.msg_price_aggregator(ht_s,n_msgs,price)
        g,y,g_T,kl_s = self.variational_decoder(ht_price_s,T,y_,phase)
        y_T,v_star = self.temporal_att_decoder(g,y,g_T,T)
        y_T_ = batched_index_select(y_,  dim=1,index=T - 1).squeeze()
        loss = self.loss_builder(v_star,y_,y,y_T,y_T_,kl_s,current_step=current_step,n_day=T,phase=phase)
        res =  {"y":y_T,"y_":y_T_,"loss":loss}
        return res


