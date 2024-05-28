from art.estimators.classification.scikitlearn import SklearnClassifier, ScikitlearnDecisionTreeClassifier
from art.attacks.evasion import HopSkipJump
from art.attacks.evasion import DecisionTreeAttack

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
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


def BlackBoxTransfer(trainingFeatures, 
                     trainingLabels,
                     testFeatures,
                     testLabels,
                     scaler,
                     NNAttackMethod):
    

    print("================================================================================================================")
    print(f"Conducting Black Box Transferability.")
    print("================================================================================================================")    

    modelTypeList = ['NN','KNN','SVM','LR','DT','GNB']
    targetModelDict = {}

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

    print(f"Training the target models now.")
    print("================================================================================================================")

    for targetModelName in modelTypeList:
        if targetModelName == 'NN':
            trainingFeaturesTensor = torch.tensor(scaler.transform(trainingFeatures), dtype=torch.float32, device='cuda:0')
            trainingLabelsTensor= torch.tensor(trainingLabels, dtype=torch.float32, device='cuda:0')

            data = CustomDataset(X=trainingFeaturesTensor, Y=trainingLabelsTensor)
            trainDataLoader = DataLoader(dataset=data, batch_size=24, shuffle=True)
            
            model = DNN(input_shape=trainingFeaturesTensor.shape[1], output_shape=1, attackMethod=NNAttackMethod)
            model.train()
            model.selfTrain(dataloader=trainDataLoader)

        else:
            model = GridSearchCV(pipelines[targetModelName], 
                                           hyperparameters[targetModelName],
                                           cv=5,
                                           n_jobs=-1) 
            model.fit(trainingFeatures, trainingLabels)

        targetModelDict[targetModelName] = model
        print(f"{targetModelName} trained")

    print(f"Testing the models now on the test set.")
    print("================================================================================================================")
    # Measuring the accuracies of the models on test set

    for targetModelName in targetModelDict.keys():
        model = targetModelDict[targetModelName]

        if targetModelName != 'NN':
            pred = model.predict(testFeatures)
            accuracy = accuracy_score(y_true=testLabels, y_pred=pred)
            
            print(f"Accuracy for target model {targetModelName} on test set is {accuracy * 100:.2f}%")

        else:
            testFeaturesTensor = torch.tensor(scaler.transform(testFeatures), dtype=torch.float32, device=model.device)
            testLabelsTensor = torch.tensor(testLabels, dtype=torch.float32, device=model.device)
            
            model.eval()

            with torch.no_grad():
                pred = model(testFeaturesTensor)
                
                numCorrect = (pred.squeeze(1).round() == testLabelsTensor).sum()
                numSamples = testFeaturesTensor.shape[0]

            print(f"Accuracy for target model {targetModelName} on test set is {numCorrect/numSamples * 100:.2f}%")

    print("================================================================================================================")

    # Selecting target models one by one
    for targetModelName in targetModelDict.keys():
        print(f"Target Model: {targetModelName}")
        print(f"Training local models with the labels provided by target model {targetModelName}")

        localModelDict = {}

        targetModel = targetModelDict[targetModelName]
        
        if targetModelName == 'NN':
            newTrainingLabels = targetModel(trainingFeaturesTensor).round().squeeze(1).detach()
        else:
            newTrainingLabels = targetModel.predict(trainingFeatures)

        # Training local models with labels provided by targetModel
        for localModelName in modelTypeList:
            if localModelName == 'NN':
                
                localModel = DNN(input_shape=trainingFeaturesTensor.shape[1],
                                 output_shape=1, 
                                 attackMethod=NNAttackMethod)

                if targetModelName == 'NN':
                    data = CustomDataset(X=trainingFeaturesTensor, Y=torch.tensor(newTrainingLabels,dtype=torch.float32))
                else:
                    newTrainingLabelsTensor = torch.tensor(newTrainingLabels, 
                                                           dtype=torch.float32,
                                                           device=localModel.device)
                    data = CustomDataset(X=trainingFeaturesTensor, Y=newTrainingLabelsTensor)

                trainDataLoader = DataLoader(dataset=data, batch_size=24, shuffle=True)
                localModel.train()
                localModel.selfTrain(dataloader=trainDataLoader)

            else:
                localModel = GridSearchCV(pipelines[localModelName], 
                                            hyperparameters[localModelName],
                                            cv=5,
                                            n_jobs=-1) 
                if targetModelName == 'NN':
                    localModel.fit(X=trainingFeatures, y=newTrainingLabels.cpu().numpy())
                else:
                    localModel.fit(X=trainingFeatures, y=newTrainingLabels)

            localModelDict[localModelName] = localModel

            print(f"Trained local model {localModelName} using labels provided by {targetModelName}")

        print("----------------------------------------------------------------------------------------------------------------")
        print(f"Testing the local models on the test set.")

        # Testing the local models on the test set
        for localModelName in localModelDict.keys():
            localModel = localModelDict[localModelName]

            if localModelName != 'NN':
                pred = localModel.predict(testFeatures)
                accuracy = accuracy_score(y_true=testLabels, y_pred=pred)
                
                print(f"Accuracy for {localModelName} on test set is {accuracy * 100:.2f}%")

            else:
                localModel.eval()

                with torch.no_grad():
                    pred = localModel(testFeaturesTensor)
                    
                    numCorrect = (pred.squeeze(1).round() == testLabelsTensor).sum()
                    numSamples = testFeaturesTensor.shape[0]

                print(f"Accuracy of {localModelName} on test set is {numCorrect/numSamples * 100:.2f}%")
        
        print("----------------------------------------------------------------------------------------------------------------")
        print("Conducting adversarial attacks on local models")
        print("----------------------------------------------------------------------------------------------------------------")

        for localModelName in localModelDict.keys():
            localModel = localModelDict[localModelName]

            if localModelName != 'NN':
                if localModelName not in ['DT']:
                    model = SklearnClassifier(model=localModel)
                
                elif localModelName == 'DT':
                    model = ScikitlearnDecisionTreeClassifier(model=localModel.best_estimator_)

                if localModelName != 'DT':
                    attackMethod = HopSkipJump(classifier=model, 
                                            targeted=False,
                                            max_iter=10, 
                                            max_eval=5000, 
                                            verbose=True)
                
                elif localModelName == 'DT':
                    attackMethod = DecisionTreeAttack(classifier=model, 
                                                    offset=1)

                advTestFeatures = attackMethod.generate(x=testFeatures)
                advAccuracy = accuracy_score(y_true=testLabels, y_pred=localModel.predict(advTestFeatures))
                advTestFeaturesIndices = localModel.predict(advTestFeatures) != testLabels

            else:
                localModel.eval()

                advTestFeatures = localModel.selfAttack(X=testFeaturesTensor,
                                                Y=testLabelsTensor)
                
                advAccuracy = (localModel(advTestFeatures).round().squeeze(1) == testLabelsTensor).sum().item()
                advAccuracy /= testLabelsTensor.shape[0]
                advTestFeaturesIndices = localModel(advTestFeatures).round().squeeze(1) != testLabelsTensor

            print(f"Accuracy of {localModelName} on adversarial test set is {advAccuracy * 100:.2f}%")

            if targetModelName == 'NN':
                if localModelName == 'NN':
                    advTargetModelAccuracy = (targetModel(advTestFeatures).round().squeeze(1) == testLabelsTensor).sum().item()
                    advTargetModelAccuracy = advTargetModelAccuracy / testLabelsTensor.shape[0]

                else:
                    advTestFeatures = torch.tensor(scaler.transform(advTestFeatures),dtype=torch.float32,device=targetModel.device)
                    advTargetModelAccuracy = (targetModel(advTestFeatures).round().squeeze(1) == testLabelsTensor).sum()
                    advTargetModelAccuracy = advTargetModelAccuracy / testLabelsTensor.shape[0]

                advSuccTestFeatures = advTestFeatures[advTestFeaturesIndices]
                advSuccTestFeaturesCorrectLabels = testLabelsTensor[advTestFeaturesIndices]
                numAttacksTransferred = (targetModel(advSuccTestFeatures).round().squeeze(1) != advSuccTestFeaturesCorrectLabels).sum()

                print(f"Number of attacks transferred from local model {localModelName} to target model {targetModelName} is {numAttacksTransferred}")
                print(f"Percentage of attacks transferred from local model {localModelName} to target model {targetModelName} is {numAttacksTransferred / advSuccTestFeatures.shape[0] * 100:.2f}%")

            else:
                if localModelName == 'NN':
                    pred = targetModel.predict(scaler.inverse_transform(advTestFeatures.cpu().numpy()))
                    advTargetModelAccuracy = accuracy_score(y_true=testLabels, y_pred=pred)  
                    advSuccTestFeaturesCorrectLabels = testLabels[advTestFeaturesIndices.cpu().numpy()]
                    advSuccTestFeatures = advTestFeatures[advTestFeaturesIndices].cpu().numpy()
                else:
                    advTargetModelAccuracy = accuracy_score(y_true=testLabels, y_pred=targetModel.predict(advTestFeatures))
                    advSuccTestFeaturesCorrectLabels = testLabels[advTestFeaturesIndices]
                    advSuccTestFeatures = advTestFeatures[advTestFeaturesIndices]
                
                numAttacksTransferred = (targetModel.predict(advSuccTestFeatures) != advSuccTestFeaturesCorrectLabels).sum()

                print(f"Number of attacks transferred from local model {localModelName} to target model {targetModelName} is {numAttacksTransferred}")
                print(f"Percentage of attacks transferred from local model {localModelName} to target model {targetModelName} is {numAttacksTransferred / advSuccTestFeatures.shape[0] * 100:.2f}%")


if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv('Data/diabetes.csv')
    
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
    BlackBoxTransfer(trainingFeatures=X_train, 
                       trainingLabels=y_train, 
                       testFeatures=X_test, 
                       testLabels=y_test, 
                       scaler=scaler, 
                       NNAttackMethod='L1_MAD') 


