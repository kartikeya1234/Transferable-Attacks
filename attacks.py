import math
import torch
from tqdm import tqdm
from utils import adv_loss


class Attack:
    def __init__(self, model, X, Y, targeted=False, device='cuda:0') -> None:
        self.model = model
        self.X = X
        self.Y = Y
        self.targeted = targeted
        self.device = device


class L1_MAD_attack(Attack):
    def __init__(self, model, X, Y, Lambda, targeted=False, numIters=500, device='cuda:0') -> None:
        super().__init__(model, X, Y, targeted, device)
        self.lamb = Lambda
        self.numIters = numIters

    
    def attack(self):
        print('L1 weighted by MAD attack on NN')

        entire_X = self.X

        X_pert = torch.tensor(entire_X.clone() + 0.01 * torch.rand_like(entire_X), requires_grad=True)
        X_pert = X_pert.to(self.device)
            
        y_target = 1 - self.Y
        y_target = y_target.to(self.device)
        

        adv_optimizer = torch.optim.Adam([X_pert],lr = 1e-2)
        
        for param in self.model.parameters():
            param.requires_grad = False

        print(f'Performing attack on the dataset for {self.numIters} epochs with the initial value of lambda as {self.lamb}.')
        pert_bar = tqdm(range(self.numIters))

        for i in pert_bar:

            X_pert.requires_grad = True

            adv_logits = self.model(X_pert).squeeze()
            loss = adv_loss(lamb=self.lamb,
                                     adv_logits=adv_logits,
                                     y_target=y_target,
                                     x_orig=entire_X,
                                     x_pert=X_pert,
                                    )

            adv_optimizer.zero_grad()
            loss.backward()
            adv_optimizer.step()
                
            pert_bar.set_postfix(loss = float(loss))

            if i % 10 == 0:
                self.lamb *= 1.9
            
        X_pert = X_pert.detach()
        X_pert = torch.where(X_pert > 1, torch.ones_like(X_pert), X_pert)
        X_pert = torch.where(X_pert < 0, torch.zeros_like(X_pert), X_pert)

        with torch.no_grad():
            X_pert_pred = torch.round(self.model(X_pert))
            print(f"Number of successful counterfactuals : {torch.sum(X_pert_pred.squeeze() == y_target)} / {entire_X.shape[0]}")
        
        return X_pert
        

class SAIF(Attack):
    def __init__(self, 
                 model, 
                 X, 
                 Y, 
                 lossFunction, 
                 eps=1.0, 
                 k=1, 
                 numIters=500, 
                 targeted=False, 
                 device='cuda:0') -> None:

        super().__init__(model, X, Y, targeted, device)
        self.numIters = numIters
        self.lossFunction = lossFunction
        self.eps = eps
        self.k = k


    def attack(self):
        print('SAIF method on NN')

        entire_X = self.X
        
        mask = torch.ones_like(entire_X)

        for param in self.model.parameters():
            param.requires_grad = False

        input_clone = entire_X.clone()
        input_clone.requires_grad = True

        if not self.targeted:
            y_target = self.Y.clone()
        else:
            y_target = 1 - self.Y.clone()

        y_target = y_target.to(self.device)

        out = self.model(input_clone)
        loss = -self.lossFunction(out,y_target.reshape(-1,1))
        loss.backward()

        p = -self.eps * input_clone.grad.sign()
        p = p.detach()
    
        kSmallest = torch.topk(-input_clone.grad,k = self.k,dim = 1)[1]
        kSmask = torch.zeros_like(input_clone.grad,device = self.device)
        kSmask.scatter_(1,kSmallest,1)
        s = kSmask.detach().float()

        r = 1

        print(f'Performing attack on the dataset for {self.numIters} epochs with eps {self.eps} and k {self.k}')
        epochs_bar = tqdm(range(self.numIters))

        for epoch in epochs_bar:

            s.requires_grad = True
            p.requires_grad = True
            out = self.model(entire_X + mask*s*p).squeeze()

            if self.targeted:
                loss = self.lossFunction(out,y_target)
            else: 
                loss = -self.lossFunction(out,y_target)

            loss.backward()

            mp = p.grad
            ms = s.grad

            with torch.no_grad():

                v = -self.eps * mp.sign()
                
                kSmallest = torch.topk(-ms,k = self.k,dim = 1)[1]
                kSmask = torch.zeros_like(ms,device = self.device)
                kSmask.scatter_(1,kSmallest,1)
                
                z = torch.logical_and(kSmask, ms < 0).float()

                mu = 1 / (2 ** r * math.sqrt(epoch + 1))

                if self.targeted:
                    while self.lossFunction(self.model(entire_X + (s + mu * (z - s)) * (p + mu * (v - p))),y_target.reshape(-1, 1)) > loss:
                        r += 1
                        mu = 1. / (2 ** r * math.sqrt(epoch + 1))
                else:
                    while -self.lossFunction(self.model(entire_X + (s + mu * (z - s)) * (p + mu * (v - p))),y_target.reshape(-1, 1)) > loss:
                        r += 1.
                        mu = 1. / (2 ** r * math.sqrt(epoch + 1))

                p = p + mask * mu * (v - p)
                s = s + mask * mu * (z - s)
                
                X_adv = torch.clamp(entire_X + p, 0,1)
                p = X_adv - entire_X
                
                succCfIndexes = self.model(entire_X + s*p).argmax(dim=1) != y_target
                mask[succCfIndexes] = 0.0

            epochs_bar.set_postfix(loss = float(loss))

            
            if (epoch + 1) % 150 == 0:
                self.k += 1
            
        X_adv = entire_X + s*p
        X_adv = torch.where(X_adv > 1, torch.ones_like(X_adv), X_adv)
        X_adv = torch.where(X_adv < 0, torch.zeros_like(X_adv), X_adv)

        with torch.no_grad():
            X_adv_pred = torch.round(self.model(X_adv))
            print(f"Number of successful counterfactuals : {torch.sum(X_adv_pred.squeeze() != y_target)} / {entire_X.shape[0]}")
        X_adv = X_adv.detach()
        return X_adv 
