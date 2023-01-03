# facial-expression-detection
this is based on Jaffe.You can get the Jaffe database from https://zenodo.org/record/3451524#.Y7RgN9VBy3A.

# "rethinking"
To overcome overfitting in this database,we use datahance.py to enhance the data and the final test accuracy is over 80%.

# Train and test

Run python train.py and the model is based on VGG16.

We also try Resnet50,and freeze different layers in Resnet50 to see which part can be transferred from Imagenet to Jaffe.SEE DETAILS IN src.

# Some Find

In this task,batchsize is a sensitive hyperparameter which influences a lot.

# TO DO
Build a model which is transformer-based.




