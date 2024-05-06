import torch
from attacks import SAIF, L1_MAD_attack

class DNN(torch.nn.Module):

    def __init__(self,input_shape,output_shape, attackMethod='SAIF'):

        super().__init__()
        self.inputShape = input_shape
        self.outputShape = output_shape
        self.attackMethod = attackMethod
        self.model = torch.nn.Sequential(

                              torch.nn.Linear(
                                              in_features=self.inputShape,
                                              out_features=20,
                                              ),
                              torch.nn.ReLU(),
                              torch.nn.Linear(
                                              in_features=20,
                                              out_features=20,
                                              ),
                              torch.nn.ReLU(),
                              torch.nn.Linear(
                                              in_features=20,
                                              out_features=20,
                                              ),
                              torch.nn.ReLU(),
                              torch.nn.Linear(
                                              in_features=20,
                                              out_features=self.outputShape,
                                              ),
                              torch.nn.Sigmoid() 
                     )

    def forward(self,input):
        return self.model(input)

    def selfTrain(self, dataloader):
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3)
        lossFunction = torch.nn.BCELoss()
        numEpochs = 200

        self.model.train()

        for epoch in  range(numEpochs):

            runningLoss =  0
            for _, (x, y) in enumerate(dataloader):
                pred = self.model(x)
                loss = lossFunction(pred, y.unsqueeze(1))
                    
                for param in self.model.parameters():
                    param.grad = None
                    
                loss.backward()
                optim.step()

                runningLoss += loss.item() * x.size(0)
                
            print(f"Epoch {epoch+1} | Training Loss = {runningLoss/len(dataloader.dataset):.4f}")


    def selfAttack(self, X, Y):
        if self.attackMethod == 'SAIF':
            
            lossFunction = torch.nn.BCELoss()
            attackMethod=SAIF(model=self.model,
                              X=X,
                              Y=Y,
                              lossFunction=lossFunction)
            advX = attackMethod.attack()
        
        elif self.attackMethod == 'L1_MAD':
            
            attackMethod=L1_MAD_attack(model=self.model,
                                       X=X,
                                       Y=Y,
                                       Lambda=1e-10)
            advX = attackMethod.attack()

        else:
            raise NotImplementedError(f"Attack method {self.attackMethod} not implemented.")
        
        return advX






