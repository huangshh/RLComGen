import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils

class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder,self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.embedding = nn.Embedding(config.DICT_CODE,config.EMB_SIZE)
        for i in range(config.NUM_LAYER):
            self.__setattr__("layer_{}".format(i),nn.GRU(config.ENC_SIZE,config.ENC_SIZE))
    
    def forward(self,inputs):
        device = self.device
        config = self.config
        lengths = [len(x) for x in inputs]
        inputs = [torch.tensor(x).to(device) for x in inputs]
        inputs = rnn_utils.pad_sequence(inputs)
        tensor = self.embedding(inputs)
        for i in range(config.NUM_LAYER):
            skip = tensor
            tensor = rnn_utils.pack_padded_sequence(tensor,lengths,enforce_sorted=False)
            tensor, h = getattr(self,"layer_{}".format(i))(tensor)
            tensor = rnn_utils.pad_packed_sequence(tensor)[0]
            tensor = tensor + skip
        
        tensor = [x[:i] for x,i in zip(torch.unbind(tensor,axis=1),lengths)]
        return tensor, h

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
        _inf = torch.tensor(-1e6).to(q.device)
        attn_weight = torch.where(mask,attn_weight,_inf)
        attn_weight = attn_weight.softmax(1)
        attn_weight = attn_weight.unsqueeze(-1)
        context = attn_weight*v
        context = context.sum(1)
        return context

class Decoder(nn.Module):
    def __init__(self,config):
        super(Decoder,self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config       
        self.ast_attn = Attn(config)
        self.api_attn = Attn(config)
        self.ast_enc_dropout = nn.Dropout(config.DROPOUT)
        self.ast_h_dropout = nn.Dropout(config.DROPOUT)
        self.api_enc_dropout = nn.Dropout(config.DROPOUT)
        self.api_h_dropout = nn.Dropout(config.DROPOUT)
        self.embedding = nn.Embedding(config.DICT_WORD,config.EMB_SIZE)
        for i in range(config.NUM_LAYER-1):
            self.__setattr__("layer_{}".format(i),nn.GRU(config.DEC_SIZE,config.DEC_SIZE))
        self.rnn = nn.GRU(2*config.DEC_SIZE,config.DEC_SIZE)
        self.fc = nn.Linear(config.DEC_SIZE,config.DICT_WORD)

        self.loss_function = nn.CrossEntropyLoss(reduction='sum')

    def forward(self,inputs,l_states,ast_enc,ast_mask,api_enc,api_mask):
        config = self.config
        device = self.device
        lengths = [len(x) for x in inputs]
        inputs = [torch.tensor(x).to(device) for x in inputs]
        inputs = rnn_utils.pad_sequence(inputs)
        tensor = self.embedding(inputs)
        for i in range(config.NUM_LAYER-1):
            skip = tensor
            tensor = rnn_utils.pack_padded_sequence(tensor,lengths,enforce_sorted=False)
            tensor, l_states[i] = getattr(self,"layer_{}".format(i))(tensor,l_states[i])
            tensor = rnn_utils.pad_packed_sequence(tensor)[0]
            tensor = tensor + skip
        
        context = self.ast_attn(tensor,ast_enc,ast_enc,ast_mask)
        context = context + self.api_attn(tensor,api_enc,api_enc,api_mask)
        tensor = torch.cat([tensor,context],-1)
        tensor = rnn_utils.pack_padded_sequence(tensor,lengths,enforce_sorted=False)
        tensor, l_states[-1] = self.rnn(tensor,l_states[-1])
        tensor = rnn_utils.pad_packed_sequence(tensor)[0]
        tensor = self.fc(tensor)
        return tensor, l_states

    def get_loss(self,ast_enc,ast_h,api_enc,api_h,targets):
        device = self.device
        config = self.config
        start_token = config.START_TOKEN
        end_token = config.END_TOKEN
        tgt_lengths = [len(x)+1 for x in targets]
        inputs = [[start_token]+x for x in targets]
        targets = [x+[end_token] for x in targets]
        ast_lengths = [x.shape[0] for x in ast_enc]
        ast_mask = [torch.ones(x).to(device) for x in ast_lengths]
        ast_mask = rnn_utils.pad_sequence(ast_mask)
        ast_mask = ast_mask.unsqueeze(0)
        ast_mask = ast_mask.repeat(max(tgt_lengths),1,1)
        ast_mask = ast_mask.eq(1)
        ast_enc = rnn_utils.pad_sequence(ast_enc)
        api_lengths = [x.shape[0] for x in api_enc]
        api_mask = [torch.ones(x).to(device) for x in api_lengths]
        api_mask = rnn_utils.pad_sequence(api_mask)
        api_mask = api_mask.unsqueeze(0)
        api_mask = api_mask.repeat(max(tgt_lengths),1,1)
        api_mask = api_mask.eq(1)
        api_enc = rnn_utils.pad_sequence(api_enc)
        ast_h = self.ast_h_dropout(ast_h)
        ast_enc = self.ast_enc_dropout(ast_enc)
        api_h = self.api_h_dropout(api_h)
        api_enc = self.api_enc_dropout(api_enc)
        l_states = [ast_h+api_h for _ in range(config.NUM_LAYER)]
        tensor, l_states = self.forward(inputs,l_states,ast_enc,ast_mask,api_enc,api_mask)
        loss = 0.
        for i in range(len(tgt_lengths)):
            loss = loss + self.loss_function(tensor[:tgt_lengths[i],i],torch.tensor(targets[i][:tgt_lengths[i]]).to(device))
        loss = loss / sum(tgt_lengths)
        return loss

    def translate(self,ast_enc,ast_h,api_enc,api_h):
        device = self.device
        config = self.config
        start_token = config.START_TOKEN
        end_token = config.END_TOKEN
        h = ast_h+api_h
        l_states = [h for _ in range(config.NUM_LAYER)]
        ast_lengths = [x.shape[0] for x in ast_enc]
        ast_mask = [torch.ones(x).to(device) for x in ast_lengths]
        ast_mask = rnn_utils.pad_sequence(ast_mask)
        ast_mask = ast_mask.unsqueeze(0)
        ast_mask = ast_mask.eq(1)
        api_lengths = [x.shape[0] for x in api_enc]
        api_mask = [torch.ones(x).to(device) for x in api_lengths]
        api_mask = rnn_utils.pad_sequence(api_mask)
        api_mask = api_mask.unsqueeze(0)
        api_mask = api_mask.eq(1)
        ast_enc = rnn_utils.pad_sequence(ast_enc)
        api_enc = rnn_utils.pad_sequence(api_enc)
        preds = [[start_token] for _ in range(len(ast_lengths))]
        for _ in range(config.MAX_COMMENT_LEN):
            inputs = [[x[-1]] for x in preds]
            tensor,l_states = self.forward(inputs,l_states,ast_enc,ast_mask,api_enc,api_mask)
            outputs = torch.argmax(tensor,-1)[-1].detach()
            for i in range(len(ast_lengths)):
                if preds[i][-1]!=end_token:
                    preds[i].append(int(outputs[i]))
        preds = [x[1:-1] if x[-1]==end_token else x[1:] for x in preds]
        return preds

class TLCodeSum(nn.Module):
    def __init__(self,config):
        super(TLCodeSum,self).__init__()
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") 
        self.config = config
        self.ast_encoder = Encoder(config)
        self.api_encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.set_trainer()
        self.to(self.device)
    
    def forward(self,inputs,mode,targets=None):
        #torch.cuda.empty_cache()
        if mode:
            return self.train_on_batch(inputs,targets)
        else:
            return self.translate(inputs)

    def train_on_batch(self,inputs,targets):
        self.optimizer.zero_grad()
        self.train()
        ast_enc, ast_h = self.ast_encoder(inputs[0])
        api_enc, api_h = self.api_encoder(inputs[1])
        loss =self.decoder.get_loss(ast_enc,ast_h,api_enc,api_h,targets)
        loss.backward()
        self.optimizer.step()
        return float(loss)
    
    def translate(self, inputs):
        self.eval()
        ast_enc, ast_h = self.ast_encoder(inputs[0])
        api_enc, api_h = self.api_encoder(inputs[1])
        return self.decoder.translate(ast_enc,ast_h,api_enc,api_h)

    def save(self,path):
        checkpoint = {
            'config':self.config,
            'ast_encoder':self.ast_encoder,
            'api_encoder':self.api_encoder,
            'decoder':self.decoder,
            'optimizer':self.optimizer
        }
        torch.save(checkpoint,path)

    def load(self,path):
        checkpoint = torch.load(path)
        #self.config = checkpoint['config']
        self.ast_encoder = checkpoint['ast_encoder']
        self.api_encoder = checkpoint['api_encoder']
        self.decoder = checkpoint['decoder']
        self.optimizer = checkpoint['optimizer']

    def set_trainer(self):
        config = self.config
        self.optimizer = optim.Adam(params=[
            {"params":self.parameters()}
        ],lr=config.LR)