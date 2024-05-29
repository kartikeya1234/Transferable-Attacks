from art.estimators.classification.scikitlearn import SklearnClassifier, ScikitlearnDecisionTreeClassifier
from art.attacks.evasion import HopSkipJump
from art.attacks.evasion import DecisionTreeAttack

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
import pickle
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
    """
    Black Box Transfer

    Args:
        trainingFeatures (_type_): _description_
        trainingLabels (_type_): _description_
        testFeatures (_type_): _description_
        testLabels (_type_): _description_
        scaler (_type_): _description_
        NNAttackMethod (_type_): _description_
    """

    print("================================================================================================================")
    print(f"Conducting Black Box Transferability.")
    print("================================================================================================================")    

    modelTypeList = ['NN','KNN','SVM','LR','DT','GNB']
    targetModelList = ['KNN','SVM','LR','DT','GNB']
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

    print(f"Loading pretrained target models")

    # Loading pre-trained target NN model
    targetNNModel = torch.load('trained_models/trained_NN_BlackBox.pt')
    targetNNModel.eval()
    targetModelDict['NN'] = targetNNModel
    print(f"Pre-Trained target NN model loaded")

    # Loading pre-trained target SVM model
    targetSVMModel = pickle.load(open('trained_models/trained_SVM_BlackBox.sav','rb'))
    targetModelDict['SVM'] = targetSVMModel
    print(f"Pre-Trained target SVM model loaded")

    # Loading pre-trained target KNN model
    targetKNNModel = pickle.load(open('trained_models/trained_KNN_BlackBox.sav','rb'))
    targetModelDict['KNN'] = targetKNNModel
    print(f"Pre-Trained target KNN model loaded")

    # Loading pre-trained target LR model
    targetLRModel = pickle.load(open('trained_models/trained_LR_BlackBox.sav','rb'))
    targetModelDict['LR'] = targetLRModel
    print(f"Pre-Trained target LR model loaded")

    # Loading pre-trained target GNB model
    targetGNBModel = pickle.load(open('trained_models/trained_GNB_BlackBox.sav','rb'))
    targetModelDict['GNB'] = targetGNBModel
    print(f"Pre-Trained target GNB model loaded")

    # Loading pre-trained target DT model
    targetDTModel = pickle.load(open('trained_models/trained_DT_BlackBox.sav','rb'))
    targetModelDict['DT'] = targetDTModel
    print(f"Pre-Trained target DT model loaded")

    print("================================================================================================================")
    print(f"Testing the target models now on the test set.")
    print("================================================================================================================")
    # Measuring the accuracies of the models on test set

    for targetModelName in targetModelDict.keys():
        targetModel = targetModelDict[targetModelName]

        if targetModelName != 'NN':
            pred = targetModel.predict(testFeatures)
            accuracy = accuracy_score(y_true=testLabels, y_pred=pred)
            
            print(f"Accuracy for target model {targetModelName} on test set is {accuracy * 100:.2f}%")

        else:
            testFeaturesTensor = torch.tensor(scaler.transform(testFeatures),
                                              dtype=torch.float32,
                                              device=targetModel.device)
            testLabelsTensor = torch.tensor(testLabels, 
                                            dtype=torch.float32,
                                            device=targetModel.device)
            
            targetModel.eval()

            with torch.no_grad():
                pred = targetModel(testFeaturesTensor)
                
                numCorrect = (pred.squeeze(1).round() == testLabelsTensor).sum()
                numSamples = testFeaturesTensor.shape[0]

            print(f"Accuracy for target model {targetModelName} on test set is {numCorrect/numSamples * 100:.2f}%")

    print("================================================================================================================")

    # Selecting target models one by one
    for targetModelName in targetModelDict.keys():
        print("================================================================================================================")
        print(f"Target Model: {targetModelName}")
        print(f"Training local models with the labels provided by target model {targetModelName}")

        localModelDict = {}
        targetModel = targetModelDict[targetModelName]
        
        # Extracting labels for the training set from the target models
        if targetModelName == 'NN':
            targetModel.eval()
            trainingFeaturesTensor = torch.tensor(scaler.transform(trainingFeatures),
                                                  dtype=torch.float32, 
                                                  device=targetModel.device)
            newTrainingLabels = targetModel(trainingFeaturesTensor).round().squeeze(1).detach()
        else:
            newTrainingLabels = targetModel.predict(trainingFeatures)

        # Training local models with labels provided by the target model
        for localModelName in modelTypeList:
            if localModelName == 'NN':
                localModel = DNN(input_shape=trainingFeaturesTensor.shape[1],
                                 output_shape=1, 
                                 attackMethod=NNAttackMethod)

                if targetModelName == 'NN':
                    data = CustomDataset(X=trainingFeaturesTensor, 
                                         Y=torch.tensor(newTrainingLabels,dtype=torch.float32))
                else:
                    newTrainingLabelsTensor = torch.tensor(newTrainingLabels, 
                                                           dtype=torch.float32,
                                                           device=localModel.device)
                    data = CustomDataset(X=trainingFeaturesTensor, 
                                         Y=newTrainingLabelsTensor)

                trainDataLoader = DataLoader(dataset=data, 
                                             batch_size=24, 
                                             shuffle=True)
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

            print(f"Trained local model {localModelName} using labels provided by target model {targetModelName}")

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

        # Conducting adversarial attacks on the local models now
        for localModelName in localModelDict.keys():
            localModel = localModelDict[localModelName]

            if localModelName != 'NN':
                if localModelName not in ['DT']:
                    model = SklearnClassifier(model=localModel)
                
                elif localModelName == 'DT':
                    model = ScikitlearnDecisionTreeClassifier(model=localModel.best_estimator_)

                # HopSkipJunp attack for all machine learning models except for Decision tree
                if localModelName != 'DT':
                    attackMethod = HopSkipJump(classifier=model, 
                                               targeted=False,
                                               max_iter=10, 
                                               max_eval=5000, 
                                               verbose=True)
                
                # Decision Tree attack from Papernot et al.
                elif localModelName == 'DT':
                    attackMethod = DecisionTreeAttack(classifier=model, 
                                                      offset=1)
                
                # advAccuracy denotes the accuracy of a model on the adversarial samples
                advTestFeatures = attackMethod.generate(x=testFeatures)
                advAccuracy = accuracy_score(y_true=testLabels, y_pred=localModel.predict(advTestFeatures))
                advTestFeaturesIndices = localModel.predict(advTestFeatures) != testLabels

            else:
                localModel.eval()
                advTestFeatures = localModel.selfAttack(X=testFeaturesTensor,
                                                        Y=testLabelsTensor)
                
                # advAccuracy denotes the accuracy of a model on the adversarial samples
                advAccuracy = (localModel(advTestFeatures).round().squeeze(1) == testLabelsTensor).sum().item()
                advAccuracy = advAccuracy / testLabelsTensor.shape[0]
                advTestFeaturesIndices = localModel(advTestFeatures).round().squeeze(1) != testLabelsTensor

            print(f"Accuracy of local model {localModelName} on adversarial test set is {advAccuracy * 100:.2f}%")

            # How many attacks, that are successful on local model, are being transferred to the target model 
            if targetModelName == 'NN':
                if localModelName == 'NN':
                    advTargetModelAccuracy = (targetModel(advTestFeatures).round().squeeze(1) == testLabelsTensor).sum().item()
                    advTargetModelAccuracy = advTargetModelAccuracy / testLabelsTensor.shape[0]

                else:
                    advTestFeatures = torch.tensor(scaler.transform(advTestFeatures),
                                                   dtype=torch.float32,
                                                   device=targetModel.device)
                    advTargetModelAccuracy = (targetModel(advTestFeatures).round().squeeze(1) == testLabelsTensor).sum().item()
                    advTargetModelAccuracy = advTargetModelAccuracy / testLabelsTensor.shape[0]

                # Selecting adversarial samples that fooled the local model to see if it also fools the target model.
                # Looking at how many and what percentage of those samples also fooled the target model
                advSuccTestFeatures = advTestFeatures[advTestFeaturesIndices]
                advSuccTestFeaturesCorrectLabels = testLabelsTensor[advTestFeaturesIndices]
                numAttacksTransferred = (targetModel(advSuccTestFeatures).round().squeeze(1) != advSuccTestFeaturesCorrectLabels).sum().item()

                print(f"Accuracy of target model {targetModelName} on adversarial test set is {advTargetModelAccuracy * 100:.2f}%")
                print(f"Number of attacks transferred from local model {localModelName} to target model {targetModelName} is {numAttacksTransferred}")
                print(f"Percentage of attacks transferred from local model {localModelName} to target model {targetModelName} is {numAttacksTransferred / advSuccTestFeatures.shape[0] * 100:.2f}%")
                print("----------------------------------------------------------------------------------------------------------------")

            else:
                if localModelName == 'NN':
                    advTestFeatures = scaler.inverse_transform(advTestFeatures.cpu().numpy())
                    pred = targetModel.predict(advTestFeatures)
                    advTargetModelAccuracy = accuracy_score(y_true=testLabels, y_pred=pred)  
                    advSuccTestFeaturesCorrectLabels = testLabels[advTestFeaturesIndices.cpu().numpy()]
                    advSuccTestFeatures = advTestFeatures[advTestFeaturesIndices.cpu().numpy()]
                else:
                    advTargetModelAccuracy = accuracy_score(y_true=testLabels, 
                                                            y_pred=targetModel.predict(advTestFeatures))
                    advSuccTestFeaturesCorrectLabels = testLabels[advTestFeaturesIndices]
                    advSuccTestFeatures = advTestFeatures[advTestFeaturesIndices]
                
                # Selecting adversarial samples that fooled the local model to see if it fools the target model.
                # Looking at how many and what percentage of those samples also fooled the target model
                numAttacksTransferred = (targetModel.predict(advSuccTestFeatures) != advSuccTestFeaturesCorrectLabels).sum().item()

                print(f"Accuracy of target model {targetModelName} on adversarial test set is {advTargetModelAccuracy * 100:.2f}%")
                print(f"Number of attacks transferred from local model {localModelName} to target model {targetModelName} is {numAttacksTransferred}")
                print(f"Percentage of attacks transferred from local model {localModelName} to target model {targetModelName} is {numAttacksTransferred / advSuccTestFeatures.shape[0] * 100:.2f}%")
                print("----------------------------------------------------------------------------------------------------------------")

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
    BlackBoxTransfer(trainingFeatures=X_train, 
                       trainingLabels=y_train, 
                       testFeatures=X_test, 
                       testLabels=y_test, 
                       scaler=scaler, 
                       NNAttackMethod='L1_MAD') 


