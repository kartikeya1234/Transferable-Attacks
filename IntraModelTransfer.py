from art.estimators.classification.scikitlearn import ScikitlearnClassifier, ScikitlearnDecisionTreeClassifier
from art.estimators.classification import XGBoostClassifier
from art.attacks.evasion import HopSkipJump
from art.attacks.evasion import DecisionTreeAttack

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import DNN
from CustomDataset import CustomDataset

import warnings
warnings.filterwarnings('ignore')


def GetNSplits(features, 
               labels, 
               nSplits=5, 
               isNN=False) -> dict:
    """GetNSplits
    Accepts features and labels and provides n disjoint sets.

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
            splitFeatures = torch.tensor(splitFeatures, dtype=torch.float32)
            splitLabels = torch.tensor(splitLabels, dtype=torch.float32)

        dataSplitsDict[i] = [splitFeatures, splitLabels]

    return dataSplitsDict


def IntraModelTransfer(trainingFeatures, 
                       trainingLabels,
                       testFeatures,
                       testLabels, 
                       modelType,
                       numModelInstances,
                       NNAttackMethod='SAIF'):
    
    print("================================================================================================================")
    print(f"Conducting Intra Model Transferability for model {modelType} with {numModelInstances} instances.")
    print("================================================================================================================")

    hyperparameters = {
        'LR' : {
            'logisticregression__solver' : ['newton-cg', 'lbfgs', 'saga'],
            'logisticregression__C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        },
        'GNB' : {
            'var_smoothing': np.logspace(0,-9, num=100)
        },
        'SVM' : {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
        },
        'XGB' : {
            'n_estimators': [50, 100, 300, 500, 1000],
            'learning_rate' : [0.001, 0.01, 0.1, 0.2, 0.3],
            'max_depth': [1, 3, 5, 7, 10, 15]
        },
        'DT' : {
            "max_depth": [3, None],
            "max_features": randint(1, 8),
            "min_samples_leaf": randint(1, 9),
            "criterion": ["gini", "entropy"]
        },
        'KNN' : {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]  # For Minkowski distance
        }
    }

    pipelines = {
        'LR' : LogisticRegression(), 
        'GNB' : GaussianNB(),
        'SVM' : SVC(kernel='rbf'), 
        'XGB' : XGBClassifier(n_jobs=4),
        'DT' : DecisionTreeClassifier(),
        'KNN' : KNeighborsClassifier()
    }

    if modelType == 'NN':
        dataSplitsDict = GetNSplits(features=trainingFeatures,
                                    labels=trainingLabels,
                                    nSplits=numModelInstances,
                                    isNN=True)
    else: 
        dataSplitsDict = GetNSplits(features=trainingFeatures,
                                    labels=trainingLabels,
                                    nSplits=numModelInstances,
                                    isNN=False)
    
    trainedModelsDict = {}
    
    print(f"Training the models now.")
    # Training the models
    for i in range(numModelInstances):
        
        print(f"Training Model {i+1}")

        X = dataSplitsDict[i][0]
        Y = dataSplitsDict[i][1]

        if modelType != 'NN':   
            model = RandomizedSearchCV(pipelines[modelType], 
                                           hyperparameters[modelType],
                                           cv=5,
                                           n_jobs=2,
                                           random_state=42)
            model.fit(X, Y)

        else:
            data = CustomDataset(X=X, Y=Y)
            trainDataLoader = DataLoader(dataset=data, batch_size=10, shuffle=True)
            
            model = DNN(input_shape=X.shape[1], output_shape=1, attackMethod=NNAttackMethod)
            model.selfTrain(dataloader=trainDataLoader)

        trainedModelsDict[f'Model_{i}'] = model

    print("================================================================================================================")
    print(f"Testing the models now on the test set.")
    # Measuring the accuracies of the models on test set

    if modelType == 'NN':
        testFeatures = torch.tensor(testFeatures, dtype=torch.float32)
        testLabels = torch.tensor(testLabels, dtype=torch.float32)


    for modelIndex in trainedModelsDict.keys():
        model = trainedModelsDict[modelIndex]

        if modelType != 'NN':
            pred = model.predict(testFeatures)
            accuracy = accuracy_score(testLabels, pred)
            
            print(f"Accuracy for {modelIndex} on test set is {accuracy * 100:.2f}%")

        else:
            with torch.no_grad():
                pred = model(testFeatures)
                
                numCorrect = (pred.squeeze(1).round() == testLabels).sum()
                numSamples = testFeatures.shape[0]

            print(f"Accuracy for {modelIndex} on test set is {numCorrect/numSamples * 100:.2f}%")

    print("================================================================================================================")
    print('Attacking the models now with each other.')

    for modelIndex in trainedModelsDict.keys():
        if modelType != 'NN':
            model = trainedModelsDict[modelIndex]
            print(type(model))

            if modelType not in ['XGB','DT']:
                model = ScikitlearnClassifier(model=model)
            
            elif modelType == 'XGB':
                model = XGBoostClassifier(model=model)

            elif modelType == 'DT':
                model = ScikitlearnDecisionTreeClassifier(model=model)

            if modelType != 'DT':
                attackMethod = HopSkipJump(classifier=model, targeted=False)
            
            elif modelType == 'DT':
                attackMethod = DecisionTreeAttack(classifier=model)

            advTestFeatures = attackMethod.generate(x=testFeatures)

            for testModelIndex in trainedModelsDict.keys():
                evalModel = trainedModelsDict[testModelIndex]
                pred = evalModel.predict(advTestFeatures)

                transferPercent = 1 - accuracy_score(testLabels, pred)

                print(f"Percentage of transferability to {testModelIndex} for adversarial inputs created for {modelIndex} is {transferPercent*100:.2f}%")
        
        else:
            model = trainedModelsDict[modelIndex]
            model.eval()

            advTestFeatures = model.selfAttack(X=testFeatures,
                                               Y=testLabels)
            
            for testModelIndex in trainedModelsDict.keys():
                evalModel = trainedModelsDict[testModelIndex]
                pred = evalModel(advTestFeatures)
                
                numCorrect = (pred.round().squeeze(1) == testLabels).sum()
                numSamples = testFeatures.shape[0]

                print(f"Percentage of transferability to {testModelIndex} for adversarial inputs created for {modelIndex} is {(1 - numCorrect/numSamples)*100:.2f}%")
        print("================================================================================================================")





if __name__ == '__main__':
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    data = pd.read_csv('Data/diabetes.csv')
    
    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values

    scaler = StandardScaler()

    XScaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(XScaled, Y, test_size=0.2, random_state=42)

    IntraModelTransfer(X_train, y_train, X_test, y_test, 'XGB',3,NNAttackMethod='L1_MAD')

