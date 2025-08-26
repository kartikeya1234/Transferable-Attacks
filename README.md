# Transferable Attacks
- We explore transferability across different machine learning models.
- We consider the **Gaussian Naive Bayes**, **Support Vector Machine**, **Neural Network**, **K Nearest Neighbors**, **Logistic Regression** and **Decision Tree** models.
- We consider three methods to compare transferability, namely Black Box , Intra Model  and Cross Model Transferabilities.
- Install all the requirements using the `requirements.txt` file.

### Intra Model Transferability
- In this, we consider multiple instances of a model which differ on the basis of subsection of training data and hyperparameters chosen. We then create adversarial attacks for a single instance and then examine whether those attacks are able to fool the other instances.
- For running it, execute the following command.
```bash
python IntraModelTransfer.py
```

### Cross Model Transferability
- In this, we consider multiple type of models which are trained on the same training dataset. We then create adversarial attacks for a model type and then examine whether those attacks are able to fool the other model types.
- For running it, execute the following command.
```bash
python CrossModelTransfer.py
```