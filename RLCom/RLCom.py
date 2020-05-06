import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from bleu import *

class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder,self).__init__()
        self.config = config
        for i in range(config.NUM_LAYER):
            self.__setattr__("layer_{}".format(i),nn.LSTM(config.ENC_SIZE,config.ENC_SIZE))
        
    def forward(self,inputs):
        config = self.config
        lengths = [x.shape[0] for x in inputs]
        tensor = rnn_utils.pad_sequence(inputs)
        for i in range(config.NUM_LAYER):
            skip = tensor
            tensor = rnn_utils.pack_padded_sequence(tensor,lengths,enforce_sorted=False)
            tensor, (h,c) = getattr(self,"layer_{}".format(i))(tensor)
            tensor = rnn_utils.pad_packed_sequence(tensor)[0]
            tensor = tensor + skip
        
        tensor = [x[:i] for x,i in zip(torch.unbind(tensor,axis=1),lengths)]
        return tensor,(h,c)

class Attn(nn.Module):
    def __init__(self,config):
        super(Attn,self).__init__()
        self.config = config
        self.Q = nn.Linear(config.ENC_SIZE,config.ATTN_SIZE)
        self.K = nn.Linear(config.ENC_SIZE,config.ATTN_SIZE)
        self.V = nn.Linear(config.ENC_SIZE,config.ATTN_SIZE)
        self.W = nn.Linear(config.ATTN_SIZE,1)
    
    def forward(self,q,k,v,mask):
        config = self.config
        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)
        q = q.unsqueeze(1)
        k = k.unsqueeze(0)
        attn_weight = self.W(torch.tanh(q+k))
        attn_weight = attn_weight.squeeze(-1)
        _nan = torch.tensor(-1e6).to(q.device)
        attn_weight = torch.where(mask,attn_weight,_nan)
        attn_weight = attn_weight.softmax(1)
        attn_weight = attn_weight.unsqueeze(-1)
        context = attn_weight*v
        context = context.sum(1)
        return context

class Decoder(nn.Module):
    def __init__(self,config):
        super(Decoder,self).__init__()
        self.config = config
        self.attn = Attn(config)
        for i in range(config.NUM_LAYER-1):
            self.__setattr__("layer_{}".format(i),nn.LSTM(config.DEC_SIZE,config.DEC_SIZE))
        self.rnn = nn.LSTM(2*config.DEC_SIZE,config.DEC_SIZE)
        
    def forward(self,inputs,l_states,enc):
        config = self.config
        lengths_enc = [x.shape[0] for x in enc]
        enc = rnn_utils.pad_sequence(enc)
        mask = [torch.ones(x).to(enc.device) for x in lengths_enc]
        mask = rnn_utils.pad_sequence(mask)
        mask = mask.unsqueeze(0)
        mask = mask.eq(1)
        lengths_input = [x.shape[0] for x in inputs]
        tensor = rnn_utils.pad_sequence(inputs)
        for i in range(config.NUM_LAYER-1):
            skip = tensor
            tensor = rnn_utils.pack_padded_sequence(tensor,lengths_input,enforce_sorted=False)
            tensor,l_states[i] = getattr(self,"layer_{}".format(i))(tensor,l_states[i])
            tensor = rnn_utils.pad_packed_sequence(tensor)[0]
            tensor = tensor + skip
        
        context = self.attn(tensor,enc,enc,mask)
        tensor = torch.cat([tensor,context],-1)
        tensor = rnn_utils.pack_padded_sequence(tensor,lengths_input,enforce_sorted=False)
        tensor, l_states[-1] = self.rnn(tensor,l_states[-1])
        tensor = rnn_utils.pad_packed_sequence(tensor)[0]
        return tensor,l_states

class Actor(nn.Module):
    def __init__(self,config):
        super(Actor,self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.embedding_enc = nn.Embedding(config.DICT_CODE,config.EMB_SIZE)
        self.embedding_dec = nn.Embedding(config.DICT_WORD,config.EMB_SIZE)
        self.dropout_enc = nn.Dropout(config.DROPOUT)
        self.dropout_h = nn.Dropout(config.DROPOUT)
        self.dropout_c = nn.Dropout(config.DROPOUT)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.fc = nn.Linear(config.DEC_SIZE,config.DICT_WORD)

    def forward(self,inputs):
        device = self.device
        config = self.config
        inputs = [torch.tensor(x).to(device) for x in inputs]
        inputs = [self.embedding_enc(x) for x in inputs]
        enc,(h,c) = self.encoder(inputs)
        enc = [self.dropout_enc(x) for x in enc]
        h = self.dropout_h(h)
        c = self.dropout_c(c)
        l_states = [(h,c) for _ in range(config.NUM_LAYER)]
        batch_size = len(inputs)
        start_token = config.START_TOKEN
        end_token = config.END_TOKEN
        preds = [[start_token] for _ in range(batch_size)]
        tensor = torch.tensor([start_token for _ in range(batch_size)]).to(device)
        outputs = torch.tensor([]).to(device)
        for i in range(config.MAX_COMMENT_LEN):
            tensor = tensor.view(1,-1)
            tensor = self.embedding_dec(tensor)
            tensor = tensor.unbind(1)
            tensor,l_states = self.decoder(tensor,l_states,enc)
            tensor = self.fc(tensor)
            tensor = tensor.softmax(-1)
            outputs = torch.cat([outputs,tensor],0)
            for j in range(batch_size):
                if preds[j][-1]!=end_token:
                    next_action = torch.multinomial(tensor[0,j],1)[0]
                    preds[j].append(int(next_action))
            tensor = torch.tensor([preds[j][-1] for j in range(batch_size)]).to(device)
        preds = [x[1:] for x in preds]
        return preds,outputs

    def pretrain(self,inputs,targets):
        device = self.device
        config = self.config
        inputs = [torch.tensor(x).to(device) for x in inputs]
        inputs = [self.embedding_enc(x) for x in inputs]
        enc,(h,c) = self.encoder(inputs)
        enc = [self.dropout_enc(x) for x in enc]
        h = self.dropout_h(h)
        c = self.dropout_c(c)
        l_states = [(h,c) for _ in range(config.NUM_LAYER)]
        tensor = [torch.tensor([config.START_TOKEN]+x).to(device) for x in targets]
        tensor = [self.embedding_dec(x) for x in tensor]
        tensor,l_states = self.decoder(tensor,l_states,enc)
        tensor = self.fc(tensor)
        return targets,tensor

    def translate(self,inputs):
        device = self.device
        config = self.config
        inputs = [torch.tensor(x).to(device) for x in inputs]
        inputs = [self.embedding_enc(x) for x in inputs]
        enc,(h,c) = self.encoder(inputs)
        l_states = [(h,c) for _ in range(config.NUM_LAYER)]
        batch_size = len(inputs)
        start_token = config.START_TOKEN
        end_token = config.END_TOKEN
        preds = [[start_token] for _ in range(batch_size)]
        tensor = torch.tensor([start_token for _ in range(batch_size)]).to(device)
        tensor = tensor.view(1,-1)
        for i in range(config.MAX_COMMENT_LEN):
            tensor = self.embedding_dec(tensor)
            tensor = tensor.unbind(1)
            tensor,l_states = self.decoder(tensor,l_states,enc)
            tensor = self.fc(tensor)
            tensor = tensor.argmax(-1).detach()
            for j in range(batch_size):
                if preds[j][-1]!=end_token:
                    next_action = tensor[0,j]
                    preds[j].append(int(next_action))
        preds = [x[1:] for x in preds]
        return preds

class Critic(nn.Module):
    def __init__(self,config):
        super(Critic,self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.embedding_enc = nn.Embedding(config.DICT_CODE,config.EMB_SIZE)
        self.embedding_dec = nn.Embedding(config.DICT_WORD,config.EMB_SIZE)
        self.dropout_enc = nn.Dropout(config.DROPOUT)
        self.dropout_h = nn.Dropout(config.DROPOUT)
        self.dropout_c = nn.Dropout(config.DROPOUT)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.fc = nn.Linear(config.DEC_SIZE,1)

    def forward(self,inputs,targets):
        device = self.device
        config = self.config
        inputs = [torch.tensor(x).to(device) for x in inputs]
        inputs = [self.embedding_enc(x) for x in inputs]
        enc,(h,c) = self.encoder(inputs)
        enc = [self.dropout_enc(x) for x in enc]
        h = self.dropout_h(h)
        c = self.dropout_c(c)
        l_states = [(h,c) for _ in range(config.NUM_LAYER)]
        tensor = [torch.tensor([config.START_TOKEN]+x).to(device) for x in targets]
        tensor = [self.embedding_dec(x) for x in tensor]
        tensor,l_states = self.decoder(tensor,l_states,enc)
        tensor = self.fc(tensor)
        tensor = torch.relu(tensor)
        tensor = tensor.squeeze(-1)
        return targets,tensor

    def pretrain(self,inputs,targets):
        return self.forward(inputs,targets)

class RLCom(nn.Module):
    def __init__(self,config):
        super(RLCom,self).__init__()
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.actor = Actor(config)
        self.critic = Critic(config)
        self.set_trainer()
        self.to(self.device)

    def forward(self,inputs,mode,targets=None):
        if mode:
            return self.train_on_batch(inputs,targets)
        else:
            return self.translate(inputs)

    def set_trainer(self):
        config = self.config
        self.optimizer_actor = optim.Adam(params=[
            {"params":self.actor.parameters()}
        ],lr=config.LR)
        self.optimizer_critic = optim.Adam(params=[
            {"params":self.critic.parameters()}
        ],lr=config.LR)
        self.loss_function_pretrain_actor = nn.CrossEntropyLoss(reduction='sum')

    def train_on_batch(self,inputs,targets):
        self.actor.train()
        self.critic.eval()
        device = self.device
        config = self.config
        batch_size = len(inputs)
        end_token = config.END_TOKEN
        preds,prob_action = self.actor(inputs)
        lengths = [len(x) for x in preds]
        prob_action = prob_action[:max(lengths)]
        _,pre_points = self.critic(inputs,[x[:-1] for x in preds])
        ref_points = torch.tensor([score_sentence(preds[i][:-1] if preds[i][-1]==end_token else preds[i],targets[i],4,1)[-1] for i in range(batch_size)]).to(device)
        ref_points = ref_points.view(1,-1)
        ref_points = ref_points.repeat(pre_points.shape[0],1)
        pre_points = pre_points.detach()
        mask = torch.zeros_like(prob_action).to(device)
        for i in range(batch_size):
            for j in range(lengths[i]):
                mask[j,i,preds[i][j]] = 1
        mask = mask.eq(1)
        tensor = torch.where(mask,prob_action,torch.zeros_like(prob_action).to(device))
        tensor = tensor.sum(-1)
        mask = [torch.ones(x).to(device) for x in lengths]
        mask = rnn_utils.pad_sequence(mask)
        mask = mask.eq(1)
        tensor = torch.log(tensor)
        tensor = torch.where(mask,tensor,torch.zeros_like(tensor).to(device))
        tensor = tensor*(pre_points-ref_points)
        loss_actor = tensor.sum()
        loss_actor = loss_actor / sum(lengths)
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        self.actor.eval()
        self.critic.train()
        preds,_ = self.actor(inputs)
        lengths = [len(x) for x in preds]
        _,pre_points = self.critic(inputs,[x[:-1] for x in preds])
        ref_points = torch.tensor([score_sentence(preds[i][:-1] if preds[i][-1]==end_token else preds[i],targets[i],4,1)[-1] for i in range(batch_size)]).to(device)
        ref_points = ref_points.view(1,-1)
        ref_points = ref_points.repeat(pre_points.shape[0],1)
        mask = [torch.ones(x).to(device) for x in lengths]
        mask = rnn_utils.pad_sequence(mask)
        loss_critic = (mask*((pre_points-ref_points)**2)).sum()
        loss_critic = loss_critic / sum(lengths)
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        score = [float(x) for x in ref_points[0]]
        return sum(score)/len(score), float(loss_actor), float(loss_critic)
        
    def translate(self,inputs):
        self.eval()
        config = self.config
        end_token = config.END_TOKEN
        preds = self.actor.translate(inputs)
        preds = [x[:-1] if x[-1]==end_token else x for x in preds]
        return preds

    def pretrain(self,inputs,targets,mode):
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        device = self.device
        config = self.config
        batch_size = len(inputs)
        if mode=="actor":
            self.actor.train()
            lengths = [len(x)+1 for x in targets]
            _,prob_action = self.actor.pretrain(inputs,targets)
            targets = [torch.tensor(x+[config.END_TOKEN]).to(device) for x in targets]
            loss_actor = 0.
            for i in range(batch_size):
                loss_actor = loss_actor + self.loss_function_pretrain_actor(prob_action[:lengths[i],i],targets[i])
            loss_actor = loss_actor / sum(lengths)
            loss_actor.backward()
            self.optimizer_actor.step()
            return float(loss_actor)
        elif mode=="critic":
            self.actor.eval()
            self.critic.train()
            end_token = config.END_TOKEN
            preds,_ = self.actor(inputs)
            lengths = [len(x) for x in preds]
            _,pre_points = self.critic.pretrain(inputs,[x[:-1] for x in preds])
            ref_points = torch.tensor([score_sentence(preds[i][:-1] if preds[i][-1]==end_token else preds[i],targets[i],4,1)[-1] for i in range(batch_size)]).to(device)
            ref_points = ref_points.view(1,-1)
            ref_points = ref_points.repeat(pre_points.shape[0],1)
            mask = [torch.ones(x).to(device) for x in lengths]
            mask = rnn_utils.pad_sequence(mask)
            loss_critic = (mask*((pre_points-ref_points)**2)).sum()
            loss_critic = loss_critic / sum(lengths)
            loss_critic.backward()
            self.optimizer_critic.step()
            return float(loss_critic)

    def save(self,path):
        checkpoint = {
            'config':self.config,
            'actor':self.actor,
            'critic':self.critic,
            'optimizer_actor':self.optimizer_actor,
            'optimizer_critic':self.optimizer_critic
        }
        torch.save(checkpoint,path)

    def load(self,path):
        checkpoint = torch.load(path)
        #self.config = checkpoint['config']
        self.actor = checkpoint['actor']
        self.critic = checkpoint['critic']
        self.optimizer_actor = checkpoint['optimizer_actor']
        self.optimizer_critic = checkpoint['optimizer_critic']