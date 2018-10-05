# API Reference

## Feedforward Neural Network

### FNNClassifier

```python
class FNNClassifier (hidden_layers=(50,), dropout=0.5, epochs=100, batch_size=128,   learning_rate='auto', early_stopping=5, class_weight=None, validation_split=0.1, validation_data=None,  timeit=True, verbosity=2, metrics=('accuracy',), activation='softmax', optimizer='adam', loss='categorical_crossentropy', callbacks=(), l1_penalty=0., l2_penalty=0., gradient_clipping_norm=None, gradient_clipping_value=None)
```

This class provide implementation of feedforward neural network for classification task. Both binary and multiway classification is supported. This class creates the computation graph, based on the hyperparameters provided by user.

#### parameters
+ hidden_layers (tuple, optional): A tuple containing `n` integers. Each integer represents the number of units in the corresponding hidden layer. For example, the tuple (100, 50, 10) will create computation graph with three hidden layers with 100 units in first hidden layer, 50 in second and 10 in the last one. Note that the output layer is not included in the tuple and is created based on the shape of label `y`.

+ dropout (float or tuple, optional): If you provide a real number between 0 to 1, that dropout will be applied to every layer. If you wish to use different dropout value for each layer, you can provide a tuple with exact length as hidden_layers. This will apply individual dropout to corresponding layers.

+ epochs (int, optional): The number of epochs, you want the network to be trained for. An epoch is one pass of over complete training data.

+ batch_size(int, optional): The number of data points on in one mini_batch. Each forward pass and backward pass is trained on one mini_batch. Having a very large or small value can negatively affect performance.

+ learning_rate ('auto' or float, optional): The learning rate for optimizer. Only supported with 'adam', 'sgd' and 'rmsprop' optimizer. With other optimizers, must use the default which is 'auto'. 'auto' will use the default rate from keras.

+ early_stopping (int, optional): If set to a positive integer `n` it will stop training after `n` epochs without improvement in objective. If set to 0, early stopping is disabled.

+ class_weight (dict or 'balanced', optional): Relative weights for each class. You can either provide a dictionary with custom weights or use 'balanced' to assign weights inversely proportional to class occurrence. This is useful when dealing with imbalanced data.

+ validation_split (float, optional): The percentage of data used as validation set. Eg, 0.1 assigns 10% as validation data. This paramerters is ignored if *validation_data* is set.

+ validation_data (tuple, optional): A tuple with validation data (X_val, y_val). If set, this will supersede *validation_split*.

+ timeit (bool, optional): Displays training time, if set to true.

+ verbosity (int, optional): Valid values are 0, 1 or 2. Defines the level of verbosity. With 0 displaying nothing and 2 displaying everything.

+ metrics (tuple, optional): The metrics to be displayed while training. A tuple of valid names for Keras. Keep in mind that metric does not affect the outcome.

+ activation ('softmax' or 'sigmoid', optional): By default 'softmax' activation is used for multiway classification and 'sigmoid' for binary classification. Only change this parameters if you exactly know what your doing.

+ optimizer (str or keras.Optimizer, optional): Any valid optimizer name or optimizer object from Keras will work. However, the learning rate parameter is only supported for 'adam', 'sgd' and 'rmsprop'. You can also supply a keras.Optimizer object with your own parameters. If you are not sure what the hack is this, go with the default, it's state of art.

+ loss (str, optional): By default, 'categorical_crossentropy' is used for multiway classification and 'binary_crossentropy' for binary classification. Only change if you know what you are doing. This one is closely coupled with activation.

+ callbacks (tuple, optional): A tuple of keras callback objects. You need to know how they work, in order to use this parameter.

+ l1_penalty (float, optional): The l1 penalty on the weights for each layer. Only change if you know what it means.

+ l2_penalty (float, optional): The l2 penalty on the weights for each layer. Only change if you know what it means.

+ gradient_clipping_norm (float, optional): If set, performs gradient clipping by norm.

+ gradient_clipping_value (float, optional): If set, performs gradient clipping by value.
