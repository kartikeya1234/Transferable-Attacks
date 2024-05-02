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
                     )

    def forward(self,input):
        return self.model(input)
    
    def selfAttack(self, X, Y):
        if self.attackMethod == 'SAIF':
            
            lossFunction = torch.nn.CrossEntropyLoss()
            attackMethod=SAIF(model=self.model,
                              X=X,
                              Y=Y,
                              lossFunction=lossFunction)
            advX = attackMethod.attack()
        
        elif self.attackMethod == 'L1_MAD':
            
            attackMethod=L1_MAD_attack(model=self.model,
                                       X=X,
                                       Y=Y,
                                       Lambda=1e-3)
            advX = attackMethod.attack()

        else:
            raise NotImplementedError(f"Attack method {self.attackMethod} not implemented.")






