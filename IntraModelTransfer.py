from art.estimators.classification.scikitlearn import ScikitlearnClassifier, ScikitlearnDecisionTreeClassifier
from art.estimators.classification import XGBoostClassifier
from art.attacks.evasion import HopSkipJump, FastGradientMethod
from art.attacks.evasion import DecisionTreeAttack

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

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
            splitLabels = torch.tensor(splitLabels, dtype=torch.long)

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
            
            if modelType == 'SVM':
                from sklearn.svm import SVC

                model = SVC(kernel='rbf')
                model.fit(X, Y)

            elif modelType == 'XGB':
                from xgboost import XGBClassifier

                model = XGBClassifier()
                model.fit(X, Y)

            elif modelType == 'GNB':
                from sklearn.naive_bayes import GaussianNB

                model = GaussianNB()
                model.fit(X, Y)

            elif modelType == 'LR':
                from sklearn.linear_model import LogisticRegression

                model = LogisticRegression(max_iter=10)
                model.fit(X, Y)

            elif modelType == 'DT':
                from sklearn.tree import DecisionTreeClassifier

                model = DecisionTreeClassifier()
                model.fit(X, Y)

            elif modelType == 'KNN':
                from sklearn.neighbors import KNeighborsClassifier

                model = KNeighborsClassifier()
                model.fit(X, Y)

            else:
                raise NotImplementedError(f"{modelType} model not implemented.")
        
        else:
            data = CustomDataset(X=X, Y=Y)
            trainDataLoader = DataLoader(dataset=data, batch_size=30, shuffle=True)
            
            if NNAttackMethod == 'SAIF':
                model = DNN(input_shape=X.shape[1], output_shape=2)
            else:
                model = DNN(input_shape=X.shape[1], output_shape=1, attackMethod='L1_MAD')

            optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
            lossFunction = torch.nn.CrossEntropyLoss()
            numEpochs = 200

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

        trainedModelsDict[f'Model_{i}'] = model

    print("================================================================================================================")
    print(f"Testing the models now on the test set.")
    # Measuring the accuracies of the models on test set

    if modelType == 'NN':
        testFeatures = torch.tensor(testFeatures, dtype=torch.float32)
        testLabels = torch.tensor(testLabels, dtype=torch.long)


    for modelIndex in trainedModelsDict.keys():
        model = trainedModelsDict[modelIndex]

        if modelType != 'NN':
            pred = model.predict(testFeatures)
            accuracy = accuracy_score(testLabels, pred)
            
            print(f"Accuracy for {modelIndex} on test set is {accuracy * 100:.2f}%")

        else:
            with torch.no_grad():

                pred = model(testFeatures)
                
                numCorrect = (torch.argmax(pred, dim=1) == testLabels).sum()
                numSamples = testFeatures.shape[0]

            print(f"Accuracy for {modelIndex} on test set is {numCorrect/numSamples * 100:.2f}%")

    print("================================================================================================================")
    print('Attacking the models now with each other.')

    for modelIndex in trainedModelsDict.keys():
        if modelType != 'NN':
            model = trainedModelsDict[modelIndex]
        
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
                pred = model(advTestFeatures)

                numCorrect = (torch.argmax(pred, dim=1) == testLabels).sum()
                numSamples = testFeatures.shape[0]

                print(f"Accuracy of {testModelIndex} on adversarial inputs created for {modelIndex} is {numCorrect/numSamples*100:.2f}%")
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
    X_train, X_test, y_train, y_test = train_test_split(XScaled, Y, test_size=0.3, random_state=42)

    IntraModelTransfer(X_train, y_train, X_test, y_test, 'GNB',5,NNAttackMethod='SAIF')

