import torch
import torch.nn.functional as F


class STEmbModel(torch.nn.Module):
    def __init__(self, SEDims, TEDims, OutDims, device):
        super(STEmbModel, self).__init__()
        self.TEDims = TEDims
        self.fc3 = torch.nn.Linear(SEDims, OutDims)
        self.fc4 = torch.nn.Linear(OutDims, OutDims)
        self.fc5 = torch.nn.Linear(TEDims, OutDims)
        self.fc6 = torch.nn.Linear(OutDims, OutDims)
        self.device = device


    def forward(self, SE, TE):
        #print("****Inside STEmbedding block***")
        #print("SEShape:", SE.shape)
        SE = SE.unsqueeze(0).unsqueeze(0)
        #print("SEShape:", SE.shape)
        SE = self.fc4(F.relu(self.fc3(SE)))
        #print("SEShape:", SE.shape)
        #print("TEShape:", TE.shape)
        dayofweek = F.one_hot(TE[..., 0], num_classes = 7)
        timeofday = F.one_hot(TE[..., 1], num_classes = self.TEDims-7)
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        #print("TEShape:", TE.shape)
        TE = TE.unsqueeze(2).type(torch.FloatTensor).to(self.device)
        #print("TEShape:", TE.shape)
        TE = self.fc6(F.relu(self.fc5(TE)))
        #print("TEShape:", TE.shape)
        sum_tensor = torch.add(SE, TE)
        #print("SumShape:", sum_tensor.shape)
        return sum_tensor


class SpatialAttentionModel(torch.nn.Module):
    def __init__(self, K, d):
        super(SpatialAttentionModel, self).__init__()
        D = K*d
        self.fc7 = torch.nn.Linear(2*D, D)
        self.fc8 = torch.nn.Linear(2*D, D)
        self.fc9 = torch.nn.Linear(2*D, D)
        self.fc10 = torch.nn.Linear(D, D)
        self.fc11 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X, STE):
        #print("****Inside spatialAttention block***")
        #print("XShape:", X.shape)
        X = torch.cat((X, STE), dim=-1)
        #print("XShape:", X.shape)
        query = F.relu(self.fc7(X))
        #print("queryShape:", query.shape)
        key = F.relu(self.fc8(X))
        #print("keyShape:", key.shape)
        value = F.relu(self.fc9(X))
        #print("valueShape:", value.shape)
        #print("SplitLength", len(torch.split(query, self.d, dim=-1)))
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        #print("queryShape:", query.shape)
        #print("keyShape:", key.shape)
        #print("valueShape:", value.shape)
        attention = torch.matmul(query, torch.transpose(key, 2, 3))
        #print("attentionShape:", attention.shape)
        attention /= (self.d ** 0.5)
        #TODO: What is mask????
        attention = self.softmax(attention)
        #print("attentionShape:", attention.shape)
        X = torch.matmul(attention, value)
        #print("XShape:", X.shape)
        
        # ~ liss = torch.split(X, X.shape[0]//self.K, dim=0)
        # ~ for xxx in liss:
            # ~ print(xxx.shape)
        X = torch.cat(torch.split(X, X.shape[0]//self.K, dim=0), dim=-1)
        #print("XShape:", X.shape)
        X = self.fc11(F.relu(self.fc10(X)))
        #print("XShape:", X.shape)
        #print("****Exiting spatialAttention block***\n\n")
        return X
        

class TemporalAttentionModel(torch.nn.Module):
    def __init__(self, K, d):
        super(TemporalAttentionModel, self).__init__()
        D = K*d
        self.fc12 = torch.nn.Linear(2*D, D)
        self.fc13 = torch.nn.Linear(2*D, D)
        self.fc14 = torch.nn.Linear(2*D, D)
        self.fc15 = torch.nn.Linear(D, D)
        self.fc16 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X, STE):
        #print("****Inside temporalAttention block***")
        #print("XShape:", X.shape)
        X = torch.cat((X, STE), dim=-1)
        #print("XShape:", X.shape)
        query = F.relu(self.fc12(X))
        #print("queryShape:", query.shape)
        key = F.relu(self.fc13(X))
        #print("keyShape:", key.shape)
        value = F.relu(self.fc14(X))
        #print("valueShape:", value.shape)
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        #print("queryShape:", query.shape)
        #print("keyShape:", key.shape)
        #print("valueShape:", value.shape)
        query = torch.transpose(query, 2, 1)
        #print("queryShape:", query.shape)
        key = torch.transpose(torch.transpose(key, 1, 2), 2, 3)
        #print("keyShape:", key.shape)
        value = torch.transpose(value, 2, 1)
        #print("valueShape:", value.shape)
        attention = torch.matmul(query, key)
        #print("attentionShape:", attention.shape)
        attention /= (self.d ** 0.5)
        attention = self.softmax(attention)
        #print("attentionShape:", attention.shape)
        #print("XShape:", X.shape)
        X = torch.matmul(attention, value)
        #print("XShape:", X.shape)
        X = torch.transpose(X, 2, 1)
        #print("XShape:", X.shape)
        X = torch.cat(torch.split(X, X.shape[0]//self.K, dim=0), dim=-1)
        #print("XShape:", X.shape)
        X = self.fc16(F.relu(self.fc15(X)))
        #print("XShape:", X.shape)
        #print("****Exiting temporalAttention block***\n\n")
        return X



class GatedFusionModel(torch.nn.Module):
    def __init__(self, K, d):
        super(GatedFusionModel, self).__init__()
        D = K*d
        self.fc17 = torch.nn.Linear(D, D)
        self.fc18 = torch.nn.Linear(D, D)
        self.fc19 = torch.nn.Linear(D, D)
        self.fc20 = torch.nn.Linear(D, D)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, HS, HT):
        #print("****Inside gatedFusion block***")
        #print("HSShape:", HS.shape)
        #print("HTShape:", HT.shape)
        XS = self.fc17(HS)
        #print("XSShape:", XS.shape)
        XT = self.fc18(HT)
        #print("XTShape:", XT.shape)
        z = self.sigmoid(torch.add(XS, XT))
        #print("zShape:", z.shape)
        H = torch.add((z* HS), ((1-z)* HT))
        #print("HShape:", H.shape)
        H = self.fc20(F.relu(self.fc19(H)))
        #print("HShape:", H.shape)
        #print("****Exiting gatedFusion block***")
        return H


class STAttModel(torch.nn.Module):
    def __init__(self, K, d):
        super(STAttModel, self).__init__()
        self.spatialAttention = SpatialAttentionModel(K, d)
        self.temporalAttention = TemporalAttentionModel(K, d)
        self.gatedFusion = GatedFusionModel(K, d)

    def forward(self, X, STE):
        HS = self.spatialAttention(X, STE)
        HT = self.temporalAttention(X, STE)
        H = self.gatedFusion(HS, HT)
        #print("****Inside STAttblock***")
        #print("XShape:", X.shape)
        #print("HShape:", H.shape)
        #print("addShape:", torch.add(X, H).shape)
        #print("****Inside STAttblock***")
        return torch.add(X, H)


class TransformAttentionModel(torch.nn.Module):
    def __init__(self, K, d):
        super(TransformAttentionModel, self).__init__()
        D = K * d
        self.fc21 = torch.nn.Linear(D, D)
        self.fc22 = torch.nn.Linear(D, D)
        self.fc23 = torch.nn.Linear(D, D)
        self.fc24 = torch.nn.Linear(D, D)
        self.fc25 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X, STE_P, STE_Q):
        #print("****Inside transformAttention block***")
        #print("XShape:", X.shape)
        #print("STE_PShape:", STE_P.shape)
        #print("STE_QShape:", STE_Q.shape)
        query = F.relu(self.fc21(STE_Q))
        key = F.relu(self.fc22(STE_P))
        value = F.relu(self.fc23(X))
        #print("queryShape:", query.shape)
        #print("keyShape:", key.shape)
        #print("valueShape:", value.shape)
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        #print("queryShape:", query.shape)
        #print("keyShape:", key.shape)
        #print("valueShape:", value.shape)
        query = torch.transpose(query, 2, 1)
        key = torch.transpose(torch.transpose(key, 1, 2), 2, 3)
        value = torch.transpose(value, 2, 1)
        #print("queryShape:", query.shape)
        #print("keyShape:", key.shape)
        #print("valueShape:", value.shape)
        attention = torch.matmul(query, key)
        #print("attentionShape:", attention.shape)
        attention /= (self.d ** 0.5)
        attention = self.softmax(attention)
        #print("attentionShape:", attention.shape)
        X = torch.matmul(attention, value)
        #print("XShape:", X.shape)
        X = torch.transpose(X, 2, 1)
        #print("XShape:", X.shape)
        X = torch.cat(torch.split(X, X.shape[0]//self.K, dim=0), dim=-1)
        #print("XShape:", X.shape)
        X = self.fc25(F.relu(self.fc24(X)))
        #print("XShape:", X.shape)
        #print("****Exiting transformAttention block***\n\n")
        return X


class GMAN(torch.nn.Module):
    def __init__(self, K, d, SEDims, TEDims, P, L, device):
        super(GMAN, self).__init__()
        D = K*d
        self.fc1 = torch.nn.Linear(1, D)
        self.fc2 = torch.nn.Linear(D, D)
        self.STEmb = STEmbModel(SEDims, TEDims, K*d, device)
        self.STAttBlockEnc = STAttModel(K, d)
        self.STAttBlockDec = STAttModel(K, d)
        self.transformAttention = TransformAttentionModel(K, d)
        self.P = P
        self.L = L
        self.fc26 = torch.nn.Linear(D, D)
        self.fc27 = torch.nn.Linear(D, 1)

    def forward(self, X, SE, TE):
        X = X.unsqueeze(3)
        X = self.fc2(F.relu(self.fc1(X)))
        STE = self.STEmb(SE, TE)
        STE_P = STE[:, : self.P]
        STE_Q = STE[:, self.P :]
        #for _ in range(self.L):
        X = self.STAttBlockEnc(X, STE_P)
        X = self.transformAttention(X, STE_P, STE_Q)
        #for _ in range(L):
        X = self.STAttBlockDec(X, STE_Q)
        #print("X shape ater STATTDec is")
        #print(X.shape)
        X = self.fc27(F.relu(self.fc26(X)))
        #print("X shape ater FinalFC is")
        #print(X.shape)
        return X.squeeze(3)


def mae_loss(pred, label, device):
    #print("****Inside mae_loss block***")
    #print("predShape:", pred.shape)
    #print("labelShape:", label.shape)
    mask = (label != 0)
    #print("maskShape:", mask.shape)
    mask = mask.type(torch.FloatTensor).to(device)
    #print("maskShape:", mask.shape)
    mask /= torch.mean(mask) #TODO:- Why is this needed??
    #print("maskShape:", mask.shape)
    #mask = tf.compat.v2.where(condition = tf.math.is_nan(mask), x = 0., y = mask)
    mask[mask!=mask] = 0 #TODO:- is this correct and efficient??
    #print("maskShape:", mask.shape)
    loss = torch.abs(pred - label)
    #print("lossShape:", loss.shape)
    loss *= mask
    #print("lossShape:", loss.shape)
    #loss = tf.compat.v2.where(condition = tf.math.is_nan(loss), x = 0., y = loss)
    loss[loss!=loss] = 0 #TODO:- is this correct and efficient??
    #print("lossShape:", loss.shape)
    loss = torch.mean(loss)
    #print("lossitem:", loss)
    #print("****Exiting mae_loss block***")
    return loss
