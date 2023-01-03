# facial-expression-detection
This repo is based on Jaffe.You can get the Jaffe database from https://zenodo.org/record/3451524#.Y7RgN9VBy3A.

## "rethinking"
To overcome overfitting in this database,we use datahance.py to enhance the data and the final test accuracy is over 80%.

## Train and test

#### Dataset

SEE details in train.json test.json and train_1.json.

#### Run

Run python train.py and the model is based on VGG16.

We also try Resnet50,and freeze different layers in Resnet50 to see which part can be transferred from Imagenet to Jaffe.SEE DETAILS IN src.

self_cnn.py is a CNN base model in which we use fewer Convolutional Neural Networks to overcome overfitting.

## Some Tricks

In this task,batchsize is a sensitive hyperparameter which influences a lot rather than lr and so on.

## TO DO
Build a model which is transformer-based.




