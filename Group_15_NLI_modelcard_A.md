language: en

license: cc-by-4.0

tags:
- text-classification

repo: https://github.com/Sasha-XIONG/NLI

---

# Model Card for w15122jl-NLI

<!-- Provide a quick summary of what the model is/does. -->

This model is intended to determine if a hypothesis is true based on a given premise.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

The model uses a linearSVC traditional machine learning model of SVM where feature embedding uses roberta to process the text. And GridSearch is given to select the best parameter values.

- **Developed by:** Jinyang Li
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** SVM models trained using Roberta embeddings as features
- **Base model:** LinearSVC
- **Feature embedding model:** xlm-roberta-base

### Model Resources

<!-- Provide links where applicable. -->

- **Base model repository:** https://github.com/scikit-learn/scikit-learn
- **Base model documentation:** https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#
- **Feature embedding model repository:** https://huggingface.co/FacebookAI/xlm-roberta-base

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

26K premise-hypothesis pairs including different languages, publishing platforms.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - linear_svc__C: 0.01
      - linear_svc__class_weight: None
      - scaler_standard__with_mean: True 
      - scaler_standard__with_std: True
      - dual: False
      - Max iteration: 20000

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: about 3 hours
      - model size: 143KB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

About 6K premise-hypothesis pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision: 0.6537482635574783
      - Recall: 0.6320320617485528
      - F1-score: 0.6230948973111805
      - Accuracy: 0.6320320617485528

### Results

The model obtained an F1-score of 62% and an accuracy of 63%.

## Technical Specifications

### Hardware


      - RAM: 51 GB
      - Storage: about 225GB （used about 30GB）
      - GPU: N/A

### Software


      - Scikit-learn 1.2.2
      - Scipy 1.11.4
      - Sklearn-pandas 2.2.0
      - Transformers 4.40.0
      - Torch 2.2.1+cu121

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Despite the presence of multiple languages in the training data and the use of a multilingual model for feature embedding, only the English model was used for text processing.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

Hyperparameters are selected by defining the range, the best performing set using GridSearchCV, cross-validation and the number of iterations are changed manually experimentally

---
