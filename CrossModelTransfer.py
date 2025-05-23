from art.estimators.classification.scikitlearn import SklearnClassifier, ScikitlearnDecisionTreeClassifier
from art.attacks.evasion import HopSkipJump
from art.attacks.evasion import DecisionTreeAttack

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import DNN
from CustomDataset import CustomDataset

import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)


def CrossModelTransfer(trainingFeatures, 
                       trainingLabels,
                       testFeatures,
                       testLabels,
                       scaler, 
                       NNAttackMethod='SAIF'):

    print("================================================================================================================")
    print(f"Conducting Cross Model Transferability.")
    print("================================================================================================================")    

    modelTypeList = ['NN','KNN','SVM','LR','DT','GNB']
    modelDict = {}

    hyperparameters = {
        'LR' : {
            'logisticregression__solver' : ['newton-cg', 'lbfgs', 'saga'],
            'logisticregression__C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        },
        'GNB' : {
            'var_smoothing': np.logspace(0,-9, num=100)
        },
        'SVM' : {
            'svc__C': [0.1, 1, 10, 100, 1000],
            'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001]
        },
        'DT' : {
            'max_depth': [2, 3, 5, 10, 20],
            'min_samples_leaf': [5, 10, 20, 50, 100],
            'criterion': ["gini", "entropy"]
        },
        'KNN' : {
            'kneighborsclassifier__n_neighbors': np.arange(1,11, 1),
            'kneighborsclassifier__leaf_size': np.arange(20,41,1),
            'kneighborsclassifier__p': (1,2),
            'kneighborsclassifier__weights': ('uniform', 'distance'),
            'kneighborsclassifier__metric': ('minkowski', 'chebyshev')
        }
    }

    pipelines = {
        'LR' : make_pipeline(MinMaxScaler(), LogisticRegression()), 
        'GNB' : GaussianNB(),
        'SVM' : make_pipeline(MinMaxScaler(), SVC(kernel='rbf')), 
        'DT' : DecisionTreeClassifier(),
        'KNN' : make_pipeline(MinMaxScaler(), KNeighborsClassifier())
    }

    print(f"Training the models now.")
    print("================================================================================================================")

    for modelName in modelTypeList:
        if modelName == 'NN':
            trainingFeaturesTensor = torch.tensor(scaler.transform(trainingFeatures), dtype=torch.float32, device='cuda:0')
            trainingLabelsTensor= torch.tensor(trainingLabels, dtype=torch.float32, device='cuda:0')

            data = CustomDataset(X=trainingFeaturesTensor, Y=trainingLabelsTensor)
            trainDataLoader = DataLoader(dataset=data, batch_size=24, shuffle=True)
            
            model = DNN(input_shape=trainingFeaturesTensor.shape[1], output_shape=1, attackMethod=NNAttackMethod)
            model.train()
            model.selfTrain(dataloader=trainDataLoader)

        else:
            model = GridSearchCV(pipelines[modelName], 
                                           hyperparameters[modelName],
                                           cv=5,
                                           n_jobs=-1) 
            model.fit(trainingFeatures, trainingLabels)

        modelDict[modelName] = model
        print(f"{modelName} trained")
    
    print(f"Testing the models now on the test set.")
    print("================================================================================================================")
    # Measuring the accuracies of the models on test set

    for modelName in modelDict.keys():
        model = modelDict[modelName]

        if modelName != 'NN':
            pred = model.predict(testFeatures)
            accuracy = accuracy_score(y_true=testLabels, y_pred=pred)
            
            print(f"Accuracy for {modelName} on test set is {accuracy * 100:.2f}%")

        else:
            testFeaturesTensor = torch.tensor(scaler.transform(testFeatures), dtype=torch.float32, device='cuda:0')
            testLabelsTensor = torch.tensor(testLabels, dtype=torch.float32, device='cuda:0')
            
            model.eval()

            with torch.no_grad():
                pred = model(testFeaturesTensor)
                
                numCorrect = (pred.squeeze(1).round() == testLabelsTensor).sum()
                numSamples = testFeaturesTensor.shape[0]

            print(f"Accuracy for {modelName} on test set is {numCorrect/numSamples * 100:.2f}%")
    
    print("================================================================================================================")
    print('Attacking the models now with each other.')

    
    for modelName in modelDict.keys():
        if modelName != 'NN':
            model = modelDict[modelName]

            if modelName not in ['DT']:
                model = SklearnClassifier(model=model)
            
            elif modelName == 'DT':
                model = ScikitlearnDecisionTreeClassifier(model=model.best_estimator_)

            if modelName != 'DT':
                attackMethod = HopSkipJump(classifier=model, 
                                           targeted=False,
                                           max_iter=10, 
                                           max_eval=5000, 
                                           verbose=True)
            
            elif modelName == 'DT':
                attackMethod = DecisionTreeAttack(classifier=model, 
                                                  offset=1)

            advTestFeatures = attackMethod.generate(x=testFeatures)
        
        else:
            model = modelDict[modelName]
            model.eval()

            advTestFeatures = model.selfAttack(X=testFeaturesTensor,
                                               Y=testLabelsTensor)

        for evalModelName in modelDict.keys():
            if evalModelName != 'NN':
                evalModel = modelDict[evalModelName]

                if modelName == 'NN': # If previous model is NN, we want inverse transformed adversarial samples
                    
                    # Selecting the adversarial counterpart of only those samples that are being correctly classified by the model
                    corrTestSamplesIndices = evalModel.predict(testFeatures) == testLabels
                    corrLabeledAdvTestFeatures = advTestFeatures[corrTestSamplesIndices]
                    corrTestLabels = testLabels[corrTestSamplesIndices]
                    pred = evalModel.predict(scaler.inverse_transform(corrLabeledAdvTestFeatures.cpu().numpy()))

                else: # Else

                    # Selecting the adversarial counterpart of only those samples that are being correctly classified by the model  
                    corrTestSamplesIndices = evalModel.predict(testFeatures) == testLabels
                    corrLabeledAdvTestFeatures = advTestFeatures[corrTestSamplesIndices]
                    corrTestLabels = testLabels[corrTestSamplesIndices]
                    pred = evalModel.predict(corrLabeledAdvTestFeatures)   
                
                transferPercent = 1 - accuracy_score(y_true=corrTestLabels, y_pred=pred)
                print(f"Percentage of transferability to {evalModelName} for adversarial inputs created for {modelName} is {transferPercent*100:.2f}%")

            else:
                evalModel = modelDict[evalModelName]
                evalModel.eval()

                if modelName == 'NN':

                    # Selecting the adversarial counterpart of only those samples that are being correctly classified by the model
                    corrTestSamplesIndices = evalModel(testFeaturesTensor).round().squeeze(1) == testLabelsTensor
                    corrLabeledAdvTestFeatures = advTestFeatures[corrTestSamplesIndices]
                    corrTestLabels = testLabelsTensor[corrTestSamplesIndices]
                    pred = evalModel(corrLabeledAdvTestFeatures)

                else:

                    # Selecting the adversarial counterpart of only those samples that are being correctly classified by the model
                    corrTestSamplesIndices = evalModel(testFeaturesTensor).round().squeeze(1) == testLabelsTensor
                    advTestFeaturesTensor = torch.tensor(advTestFeatures, device='cuda:0')
                    corrLabeledAdvTestFeatures = advTestFeaturesTensor[corrTestSamplesIndices]
                    corrTestLabels = testLabelsTensor[corrTestSamplesIndices]
                    pred = evalModel(torch.tensor(scaler.transform(corrLabeledAdvTestFeatures.cpu().numpy()), dtype=torch.float32,device='cuda:0'))

                numCorrect = (pred.round().squeeze(1) != corrTestLabels).sum()
                numSamples = corrLabeledAdvTestFeatures.shape[0]

                print(f"Percentage of transferability to {evalModelName} for adversarial inputs created for {modelName} is {(numCorrect/numSamples)*100:.2f}%")
        print("================================================================================================================")


if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv('Data/mushroom_cleaned.csv')
    
    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values

    scaler = MinMaxScaler()
    
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        Y, 
                                                        test_size=0.2, 
                                                        shuffle=True,
                                                        stratify=Y, 
                                                        random_state=42)
    scaler.fit(X_train)
    CrossModelTransfer(trainingFeatures=X_train, 
                       trainingLabels=y_train, 
                       testFeatures=X_test, 
                       testLabels=y_test, 
                       scaler=scaler, 
                       NNAttackMethod='SAIF')
