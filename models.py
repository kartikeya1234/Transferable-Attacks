import torch

class DNN(torch.nn.Module):

    def __init__(self,input_shape,output_shape):

        super().__init__()
        self.inputShape = input_shape
        self.outputShape = output_shape
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

class LR(torch.nn.Module):

    def __init__(self, inputDim,  outputDim):
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.model = torch.nn.Sequential(
                            torch.nn.Linear(self.inputDim, self.outputDim, bias=True)
                            )

    def forward(self, input):
        return self.model(input)

class SVM(torch.nn.Module):

    def __init__(self, inputDim, outputDim) -> None:
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(self.inputDim, self.outputDim, bias=True)
        )

    def forward(self, input):
        return torch.sign(self.model(input))