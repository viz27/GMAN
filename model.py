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
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.fc4(F.relu(self.fc3(SE)))
        dayofweek = F.one_hot(TE[..., 0], num_classes = 7)
        timeofday = F.one_hot(TE[..., 1], num_classes = self.TEDims-7)
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(2).type(torch.FloatTensor).to(self.device)
        TE = self.fc6(F.relu(self.fc5(TE)))
        sum_tensor = torch.add(SE, TE)
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
        X = torch.cat((X, STE), dim=-1)
        query = F.relu(self.fc7(X))
        key = F.relu(self.fc8(X))
        value = F.relu(self.fc9(X))
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        attention = torch.matmul(query, torch.transpose(key, 2, 3))
        attention /= (self.d ** 0.5)
        attention = self.softmax(attention)
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, X.shape[0]//self.K, dim=0), dim=-1)
        X = self.fc11(F.relu(self.fc10(X)))
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
        X = torch.cat((X, STE), dim=-1)
        query = F.relu(self.fc12(X))
        key = F.relu(self.fc13(X))
        value = F.relu(self.fc14(X))
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        query = torch.transpose(query, 2, 1)
        key = torch.transpose(torch.transpose(key, 1, 2), 2, 3)
        value = torch.transpose(value, 2, 1)
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = self.softmax(attention)
        X = torch.matmul(attention, value)
        X = torch.transpose(X, 2, 1)
        X = torch.cat(torch.split(X, X.shape[0]//self.K, dim=0), dim=-1)
        X = self.fc16(F.relu(self.fc15(X)))
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
        XS = self.fc17(HS)
        XT = self.fc18(HT)
        z = self.sigmoid(torch.add(XS, XT))
        H = torch.add((z* HS), ((1-z)* HT))
        H = self.fc20(F.relu(self.fc19(H)))
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
        query = F.relu(self.fc21(STE_Q))
        key = F.relu(self.fc22(STE_P))
        value = F.relu(self.fc23(X))
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        query = torch.transpose(query, 2, 1)
        key = torch.transpose(torch.transpose(key, 1, 2), 2, 3)
        value = torch.transpose(value, 2, 1)
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = self.softmax(attention)
        X = torch.matmul(attention, value)
        X = torch.transpose(X, 2, 1)
        X = torch.cat(torch.split(X, X.shape[0]//self.K, dim=0), dim=-1)
        X = self.fc25(F.relu(self.fc24(X)))
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
        X = self.STAttBlockEnc(X, STE_P)
        X = self.transformAttention(X, STE_P, STE_Q)
        X = self.STAttBlockDec(X, STE_Q)
        X = self.fc27(F.relu(self.fc26(X)))
        return X.squeeze(3)


def mae_loss(pred, label, device):
    mask = (label != 0)
    mask = mask.type(torch.FloatTensor).to(device)
    mask /= torch.mean(mask)
    mask[mask!=mask] = 0
    loss = torch.abs(pred - label)
    loss *= mask
    loss[loss!=loss] = 0
    loss = torch.mean(loss)
    return loss
