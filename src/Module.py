#!/usr/local/bin/python
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
from tools import batched_index_select,gussian_kl_divergence
import torch.nn.functional as F

from tools import DicToObj, generate_mask, kl_lambda_annealing


class Embedding(nn.Module):
    def __init__(self, word_table):
        super(Embedding, self).__init__()
        self.word_table = nn.Embedding.from_pretrained(torch.FloatTensor(word_table),freeze=True)

    def forward(self, x):
        return self.word_table(x)


class MsgEncoder(nn.Module):
    def __init__(self, config):
        super(MsgEncoder, self).__init__()
        self.config = config
        if self.config.use_in_bn:
            self.bn_layer = nn.BatchNorm1d(self.config.max_n_words)  #
        self.msg_in_drop_layer = nn.Dropout(self.config.dropout_mel_in)
        self.msg_rnn_encoder = nn.GRU(input_size=self.config.word_embed_size,
                                      hidden_size=self.config.mel_h_size,
                                      batch_first=True, bidirectional=True)

    def encode_with_rnn_bidirection(self, x, n_words,ss_index):
        # x: -1, max_n_words, word_embed_size
        # n_words: -1,1
        _, sort_idx = torch.sort(n_words, dim=0, descending=True)
        x = x.index_select(0, sort_idx)
        _,unsort_idx = torch.sort(sort_idx,dim=0)
        lengths = list(n_words[sort_idx])  # this should be same with "_"
        for i in range(len(lengths)):
            if lengths[i]==0:
                lengths[i] =1
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(x,lengths,True)


        #h_size = (2, x.size(0),self.config.mel_h_size)  # num_layers*direction,batch_size,hidden_size
        # h0 = Variable(torch.zeros(*h_size), requires_grad=False)
        # if self.config.use_cuda:
        #     h0 = h0.cuda()
        rnn_outputs, ht = self.msg_rnn_encoder(rnn_inputs)

        # ht: -1,2,hidden_size
        rnn_outputs,_ = nn.utils.rnn.pad_packed_sequence(rnn_outputs,batch_first=True)
        rnn_outputs = rnn_outputs.index_select(0, unsort_idx)  #batch_size*max_n_days*max_n_msgs,seq_len,hidden_size*2
        bi_h = batched_index_select(rnn_outputs,1,ss_index).squeeze()
        bi_h = bi_h.view(bi_h.size(0),2,self.config.mel_h_size)
        h = (bi_h[:,0,:]+bi_h[:,0,:])/2
        return h

    def forward(self, x, n_words,ss_index):
        # x: batch_size*max_n_days*max_n_msgs*max_n_words*word_embed_size
        # n_words: batch_size*max_n_days*n_msgs，
        x_3d = x.view(-1, self.config.max_n_words, self.config.word_embed_size)
        n_words = n_words.view(-1)
        ss_index = ss_index.view(-1)
        if self.config.use_in_bn:
            x_3d = self.bn_layer(x_3d)
        x_3d_in = self.msg_in_drop_layer(x_3d)
        msg_h_3d = self.encode_with_rnn_bidirection(x_3d_in, n_words,ss_index)
        msg_h = msg_h_3d.view(self.config.batch_size, self.config.max_n_days, self.config.max_n_msgs,self.config.mel_h_size)
        return msg_h


class MsgPriceAgg(nn.Module):
    # attention operation to aggregate everydays' msgs_h
    def __init__(self, config):
        self.config = config
        super(MsgPriceAgg, self).__init__()
        self.reduce4att_layer = nn.Sequential(nn.Linear(self.config.mel_h_size, self.config.mel_h_size, bias=False),
                                               nn.Tanh(),
                                               nn.Linear(self.config.mel_h_size, 1))

    def forward(self, x, n_msgs, price_mv):
        # x: batch_size*max_n_days*max_n_messages*hidden_size
        att_weight = self.reduce4att_layer(x).squeeze()  # x: batch_size*max_n_days*max_n_messages
        n_msg_mask = generate_mask(att_weight, n_msgs)
        att_weight[~n_msg_mask] = -1e9
        att_score = F.softmax(att_weight, -1).unsqueeze(-2)  # att_socre: batch_size*max_n_days*1*max_n_messages
        agg_msg_by_day = torch.matmul(att_score, x).squeeze()  # batch_size*max_n_days*hidden_size
        msg_price = torch.cat([agg_msg_by_day, price_mv], -1)
        return msg_price


class Z(nn.Module):
    def __init__(self, config, input_dim, z_dim):
        super(Z, self).__init__()
        self.config = config
        self.h_z_linear = nn.Sequential(nn.Linear(input_dim, z_dim), nn.Tanh())
        self.mean_linear = nn.Linear(z_dim, z_dim)
        self.stddev_linear = nn.Linear(z_dim, z_dim)

    def forward(self, x, is_prior):
        hz = self.h_z_linear(x)
        mean = self.mean_linear(hz)
        logstd = self.stddev_linear(hz)
        stddev = torch.sqrt(torch.exp(logstd))
        epsilon = torch.randn([self.config.batch_size, self.config.z_size])
        if self.config.use_cuda:
            epsilon = epsilon.cuda()
        z = mean if is_prior else mean + stddev * epsilon

        return z, mean,logstd


class VariationalMovDecoder(nn.Module):
    def __init__(self, config):
        super(VariationalMovDecoder, self).__init__()
        self.config = config
        self.vmd_in_drop_layer = nn.Dropout(self.config.dropout_vmd_in)
        self.day_rnn_encoder = nn.GRU(input_size=self.config.mel_h_size+3,
                                      hidden_size=self.config.h_size,
                                      batch_first=True, bidirectional=False)
        self.prior_z_layer = Z(config=config, z_dim=self.config.z_size,
                               input_dim=self.config.mel_h_size + 3 + self.config.h_size+self.config.z_size)
        self.post_z_layer = Z(config=config, z_dim=self.config.h_size,
                              input_dim=self.config.mel_h_size + 3 + self.config.h_size+self.config.z_size + 2)
        # z_size is same with h_size
        self.g_mlp = nn.Sequential(nn.Linear(self.config.h_size * 2, self.config.g_size), nn.Tanh())
        self.y_mlp = nn.Sequential(nn.Linear(self.config.g_size, self.config.y_size), nn.Softmax(dim=-1))

    def encode_dayinfo_with_rnn(self, x, n_day):
        # x: batch_size,max_n_day,hidden_size
        # n_day: -1,1
        _, sort_idx = torch.sort(n_day, dim=0, descending=True)
        x = x.index_select(0, sort_idx)
        _,unsort_idx = torch.sort(sort_idx)
        lengths = list(n_day[sort_idx])  # this should be same with "_"
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(x, lengths,True)
        # h_size = (1, x.size(0), x.size(-1))  # num_layers*direction,batch_size,hidden_size
        rnn_outputs, ht = self.day_rnn_encoder(rnn_inputs)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True,total_length=self.config.max_n_days)
        # rnn_outputs: batch_size,max_n_day,hidden_size
        rnn_outputs = rnn_outputs.index_select(0, unsort_idx)
        return rnn_outputs

    def forward(self, x, n_day, y_,phase):
        # 1. h_t = GRU(x,h_{t-1})
        x = self.vmd_in_drop_layer(x)
        h_s = self.encode_dayinfo_with_rnn(x, n_day)

        # 2. forward max_n_day
        x = x.permute(1, 0, 2)
        h_s = h_s.permute(1, 0, 2)
        y_ = y_.permute(1, 0, 2)
        z_shape = [self.config.batch_size, self.config.z_size]
        z_post_t_1 = Variable(torch.randn(*z_shape), requires_grad=False)
        if self.config.use_cuda:
            z_post_t_1 = z_post_t_1.cuda()
        z_prior_s = []
        z_post_s = []
        kl_s = []
        for t in range(self.config.max_n_days):
            if t > 0: z_post_t_1 = z_post_s[-1]
            z_prior_t, mean_prior,logstd_prior = self.prior_z_layer(torch.cat([x[t, :, :], h_s[t, :, :], z_post_t_1],-1),
                                                          is_prior=True)
            z_post_t, mean_post,logstd_post = self.post_z_layer(torch.cat([x[t, :, :], h_s[t, :, :], y_[t, :, :], z_post_t_1],-1),
                                                       is_prior=False)
            z_prior_s.append(z_prior_t)
            z_post_s.append(z_post_t)
            kl_t = gussian_kl_divergence(mean_prior,logstd_prior,mean_post,logstd_post)
            kl_s.append(kl_t)

        h_s = h_s.permute(1, 0, 2)
        z_prior_s = torch.stack(z_prior_s).permute(1, 0, 2)
        z_post_s = torch.stack(z_post_s).permute(1, 0, 2)
        kl_s = torch.stack(kl_s).permute(1, 0)
        g = self.g_mlp(torch.cat([h_s, z_prior_s], -1)) # prior_z or post_z
        y = self.y_mlp(g)

        sample_index = n_day - 1
        if phase=="train":
            # 训练阶段返回用多天的信息，但是测试阶段只用一天？
            g_T = batched_index_select(g, dim=1, index=sample_index)  # 训练阶段用后验的Z
        else:
            # 测试阶段用先验的Z
            z_prior_T = batched_index_select(z_prior_s, dim=1, index=sample_index)
            h_s_T = batched_index_select(h_s, dim=1, index=sample_index)
            g_T = self.g_mlp(torch.cat([z_prior_T, h_s_T], -1))
        return g, y, g_T, kl_s


class TemporalAttDecoder(nn.Module):
    def __init__(self, config):
        super(TemporalAttDecoder,self).__init__()
        self.config = config
        self.vi_reduce4att_layer = nn.Sequential(nn.Linear(self.config.g_size, self.config.g_size, bias=False),
                                                  nn.Tanh(),
                                                  nn.Linear(self.config.g_size, 1))
        self.projd_layer = nn.Sequential(nn.Linear(self.config.g_size, self.config.g_size, bias=False),
                                          nn.Tanh())
        self.target_T_predict_layer = nn.Sequential(
            nn.Linear(self.config.g_size * 2, self.config.y_size), nn.Softmax(dim=-1))

    def forward(self, g, y, g_T, n_day):
        vi = self.vi_reduce4att_layer(g).squeeze()  # batch_size * max_n_day
        projd = self.projd_layer(g)  # batch_size *max_n_day*g_size
        g_T = g_T.permute(0,2,1) # batch_size*g_size*1
        vd = torch.matmul(projd, g_T).squeeze()  # batch_size *max_n_day
        aux_score = torch.mul(vi, vd)  # 辅助日的重要性取决于自身给出的一个分数，以及目标日对其做attention的分数
        aux_mask = generate_mask(aux_score, n_day)
        aux_score[~aux_mask] = -1e9
        v_stared = torch.softmax(aux_score, -1)  # batch_size*n_day

        att_c = torch.matmul(v_stared.unsqueeze(1), g) # batch_size*1*g_size
        y_T = self.target_T_predict_layer(torch.cat([att_c.squeeze(), g_T.squeeze()], -1))
        return y_T, v_stared


class CalculateJointLoss(nn.Module):
    def __init__(self, config):
        super(CalculateJointLoss, self).__init__()
        self.config = config

    def forward(self, v_stared, y_, y, y_T,y_T_,kl_s, current_step,n_day,phase):
        v_aux = self.config.alpha * v_stared
        likelyhood_aux =torch.sum( y_ * torch.log(y),-1)  # batch_size*max_n_day
        if phase =="train":
            kl_lambda = kl_lambda_annealing(current_step, self.config.kl_lambda_start_step, self.config.kl_lambda_anneal_rate)
        else:
            kl_lambda = 1.0
        obj_aux = likelyhood_aux - kl_lambda * kl_s  # batch_size*max_n_day

        likelyhood_T = torch.sum(y_T_.squeeze() * torch.log(y_T + 1e-7),-1)
        kl_T = batched_index_select(kl_s, dim = 1,index=n_day - 1)  # 单独一天的loss由两部分组成
        obj_T = likelyhood_T - kl_lambda * (kl_T.squeeze())

        obj = obj_T + torch.sum(v_aux * obj_aux, -1)  # batch_size
        loss = torch.mean(-obj)
        return loss
