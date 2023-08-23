# 1.INTRODUCTION
 we will be discussing the technology used for speech-emotion recognition,
different Machine Learning (ML) and Deep Learning (DL) models and various research and
literature papers and reviews available and the machine specifications for running the project
properly. Basically we will be training and testing various models which can be used for
speech-emotion recognition and we will select a model which classifies the emotion more
accurately than other models. Also we will be adding visual diagrams which will help in
comparing different used models. After model selection we will try to enhance its accuracy.
Various factors affect the accuracy of models like: pre-processing, feature engineering, feature
selection(quantity, quality,etc) from audio file, model compatibility with given data, etc. so
understanding these factors well and improving accuracy is our aim.
By leveraging the power of feature engineering and the capabilities of advanced machine
learning techniques, SER can contribute to applications such as affective computing,
human-computer interaction, and sentiment analysis in various domains.

# 1.1 Problem Statement
Employing ML techniques for analysis of speech emotion recognition. We will be given speech/ audio and we have to classify the emotions of audio.
Audio is a time series data, to identify
emotion from audio we have to extract its intrinsic properties which can be done via features of
audio. Our aim is to understand feature engineering for our project.
Once the features are extracted, the next step is to select an appropriate machine
learning model for classification,selection of the best suitable model,which depends on factors
such as the complexity of the data, availability of labeled training data, and computational
resources, and finally improve the accuracy of selected models.

# 1.2 Aim and Objectives
Effective recognition of emotions by speech via machine. Identification is done with the help of
various machine learning and deep learning models using python. Approach is done by
extracting vocal features from speech from a dataset or database.

Objectives:
1. To make machines learn to detect emotions from speech.
2. To provide automation in emotion recognition without human intervention.
3. To improve recognition accuracy.
4. To provide a valuable output to users.

# 1.3 Problem Specification
Using an existing pre-built model instead of creating a whole package from scratch. This
gives us a head start towards achieving our goals and saves time. And we have also
increased the accuracy of pre-built models . For speech-emotion recognition we have various
machine learning as well as deep learning models available for us. Out of them we have
selected those which are compatible with time series data. Considered models are Random
Forest (RF), K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Multi Layer
Perceptron (MLP) and ensemble model.
We trained them and tested our data on them, and we have selected an ensemble model (RF
+ SVM + KNN + MLP) for emotion recognition

# IMPLEMENTATION
##  2.1 Recurrent Neural Network (RNN)
The model defines an RNN model using the Sequential API from Keras. It extracts MFCC
(Mel-frequency cepstral coefficients) features using the librosa library. It consists of three
SimpleRNN layers with dropout layers in between. The output layer is a Dense layer with
softmax activation. The training set consists of 80% of the data, and the test set contains the
remaining 20%. The training is performed for 50 epochs with a batch size of 32. The testing
accuracy came not more than 48%

Further implementation and steps given in report

Finally Ensemble learning model was chosen as it gave maximum accuracy on given RAVDESS dataset compared to other models as shown.

# FUTURE SCOPE
This project has the potential to enrich our understanding of human emotions, enhance
human-computer interaction, improve mental health diagnostics, and provide valuable insights in
various domains where emotions play a significant role.
Run time recognition of emotions can be added, this is particularly useful in applications where
immediate feedback or response is required, such as emotion-aware virtual assistants,
emotion-based human-computer interaction, or affective computing systems.
Multilingual and cross-cultural emotion recognition: Emotions can be expressed differently
across languages and cultures. Future research may explore methods for cross-cultural and
multilingual emotion recognition, enabling models to recognize and adapt to diverse emotional
expressions.
