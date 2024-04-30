from art.estimators.classification import LightGBMClassifier
from art.estimators.classification import XGBoostClassifier
from art.estimators.classification.scikitlearn import SklearnClassifier
from art.estimators.classification.pytorch import PyTorchClassifier
from art.attacks.evasion import HopSkipJump

from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader

from models import DNN
from CustomDataset import CustomDataset

def GetNSplits(features, 
               labels, 
               nSplits=5, 
               isNN=False) -> dict:
    """GetNSplits
    Accepts features and labels and provides n disjoint sets

    Args:
        features (_type_): Features
        labels (_type_): Labels
        nSplits (int, optional): Number of splits of data. Defaults to 5.
        isNN (bool, optional): Set to True if data splits for a neural network in pytorch. Defaults to False.

    Returns:
        dict: _description_
    """

    dataSplitsDict = {}
    kf = KFold(n_splits=nSplits, shuffle=False)

    for i, (_, dataIndices) in enumerate(kf.split(features)):
        
        splitFeatures = features[dataIndices]
        splitLabels = labels[dataIndices]

        if isNN:
            splitFeatures = torch.tensor(splitFeatures)
            splitLabels = torch.tensor(splitLabels)

        dataSplitsDict[i] = [splitFeatures, splitLabels]

    return dataSplitsDict


def IntraModelTransfer(trainingFeatures, 
                       trainingLabels,
                       testFeatures,
                       testLabels, 
                       modelType,
                       numModelsReplicas, 
                       isNN=False):
    
    dataSplitsDict = GetNSplits(features=trainingFeatures,
                                labels=trainingLabels,
                                nSplits=numModelsReplicas,
                                isNN=isNN)
    trainedModelsDict = {}
    
    # Training the models
    for i in range(numModelsReplicas):

        X = dataSplitsDict[i][0]
        Y = dataSplitsDict[i][1]

        if not isNN:
            if modelType == 'RandomForest':
                from sklearn.ensemble import RandomForestClassifier
                
                model = RandomForestClassifier(n_estimators=100)
                model = SklearnClassifier(model=model, clip_values=(0,1))
                model.fit(X, Y)
            
            elif modelType == 'SVM':
                from sklearn.svm import SVC

                model = SVC(kernel='linear')
                model = SklearnClassifier(model=model, clip_values=(0,1))
                model.fit(X, Y)

            elif modelType == 'XGBoost':
                from xgboost import XGBClassifier

                model = XGBClassifier()
                model = XGBoostClassifier(model=model, clip_values=(0,1))
                model.fit(X, Y)

            elif modelType == 'Lightgbm':
                from lightgbm import LGBMClassifier

                model = LGBMClassifier()
                model = LightGBMClassifier(model=model,clip_values=(0,1))
                model.fit(X, Y)

            elif modelType == 'LR':
                from sklearn.linear_model import LogisticRegression

                model = LogisticRegression()
                model = SklearnClassifier(model=model, clip_values=(0,1))
                model.fit(X, Y)

        else:
            data = CustomDataset(X=X, Y=Y)
            trainDataLoader = DataLoader(dataset=data, batch_size=30, shuffle=True)
            
            model = DNN(input_shape=X.shape[1], output_shape=2)
            
            optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
            lossFunction = torch.nn.CrossEntropyLoss
            numEpochs = 100

            model.train()

            for epoch in  range(numEpochs):

                runningLoss =  0
                for _, (x, y) in enumerate(trainDataLoader):

                    pred = model(x)
                    loss = lossFunction(pred, y)
                    
                    for param in model.parameters():
                        param.grad = None
                    
                    loss.backward()
                    optim.step()

                    runningLoss += loss.item() * x.size(0)

                
                print(f"Epoch {epoch+1} | Training Loss = {runningLoss/len(trainDataLoader.dataset):.4f}")

