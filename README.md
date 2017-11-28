Tensorflow R
================
Park Beom-Jin
2017년 11월 28일

### Easy-to-use neural network function using tensorflow for R users

``` r
Tensorflow_DNN(X, y, hidden_layer, 
              learning = list(optimizer_type = "GradientDescentOptimizer", 
                              init_learning_rate = 0.001, 
                              decay_rate = 0.96, 
                              decay_steps = 1e+5),
              act_fun = "relu", init_fun = "truncated_normal",
              batch_size = 50, dropout_rate = 1.0,  
              epoch = 20000, validation_set = NULL, type = "Classification",
              regularization = list(type = "l2_loss", cost = 0.0), 
              verb = TRUE, save = FALSE, output = 1)
```

<table style="width:86%;">
<colgroup>
<col width="5%" />
<col width="80%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">Arguments</th>
<th align="left"></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">X</td>
<td align="left">a matrix(or data frame) containing the explanatory variables(independent variable)</td>
</tr>
<tr class="even">
<td align="center">y</td>
<td align="left">a vector of class or response variable(dependent variable)</td>
</tr>
<tr class="odd">
<td align="center">hidden_layer</td>
<td align="left">a vector of integers specifying the number of hidden neurons(nodes) in each layer</td>
</tr>
<tr class="even">
<td align="center">learning</td>
<td align="left">a list containing optimization parameter. see &quot;Detail&quot;</td>
</tr>
<tr class="odd">
<td align="center">act_fun</td>
<td align="left">a character specifying type of activation function</td>
</tr>
<tr class="even">
<td align="center">init_fun</td>
<td align="left">a character specifying type of function to initialize weight</td>
</tr>
<tr class="odd">
<td align="center">batch_size</td>
<td align="left">number of samples that using to optimization. If it is 'NULL', batch_size = nrow(X)</td>
</tr>
<tr class="even">
<td align="center">dropout_rate</td>
<td align="left">percentage of remaining connections of nodes</td>
</tr>
<tr class="odd">
<td align="center">epoch</td>
<td align="left">the number of repetitions for the neural network's training</td>
</tr>
<tr class="even">
<td align="center">validation_set</td>
<td align="left">a list containing validation set.</td>
</tr>
<tr class="odd">
<td align="center">type</td>
<td align="left">a character specifying type of Neural Network</td>
</tr>
<tr class="even">
<td align="center">regularization</td>
<td align="left">a list containing regularization type and cost</td>
</tr>
<tr class="odd">
<td align="center">verb</td>
<td align="left">Display training accuracy</td>
</tr>
<tr class="even">
<td align="center">save</td>
<td align="left">if save = TRUE, save model in working directory</td>
</tr>
<tr class="odd">
<td align="center">output</td>
<td align="left">if output = 1, result value contains weight value by node</td>
</tr>
</tbody>
</table>

------------------------------------------------------------------------

### Example Code

``` r
source("./Tensorflow_DNN.R")
data("iris")
ind = sample(nrow(iris), size = 0.7 * nrow(iris))
train_iris = iris[ind, ]
test_iris = iris[-ind, ]

ind_v = sample(nrow(train_iris), size = 0.7 * nrow(train_iris))
vali_iris = train_iris[-ind_v, ]
train_iris = train_iris[ind_v, ]

validation_set = list(X = vali_iris[, 1:4], y = vali_iris$Species)
hidden_layer = c(10, 10)

iris_dnn = tensorflow_DNN(X = train_iris[, 1:4], y = train_iris$Species, hidden_layer,
               validation_set = validation_set,
               regularization = list(type = "l2_loss", cost = 0.0),
               dropout_rate = 1.0, epoch = 5000, output = 1)
```

    ## Loading required package: dplyr

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

    ## Loading required package: tensorflow

    ## --------------------------------------------------- 
    ## Setting Parameter for Classification Neural Network 
    ## 
    ##  <Layer Parameter> 
    ##  Activation Function: relu 
    ##  Initial Function: truncated_normal 
    ##  Dropout Rate: 1 
    ##  Layer1:10
    ##  Layer2:10
    ##  
    ##  <Optimization Parameter> 
    ##  Optimizer: GradientDescentOptimizer 
    ##  Learning Rate: 0.001 
    ##  Decay Rate: 0.96 
    ##  Decay steps: 1e+05 
    ##  Epoch: 5000 
    ##  
    ##  <Regularization Parameter> 
    ##  Regularizer: l2_loss 
    ##  Cost: 0 
    ## --------------------------------------------------- 
    ## Epoch 500. Loss 0.208178, Training accuracy 0.913043 
    ## Epoch 1000. Loss 0.159512, Training accuracy 0.956522 
    ## Epoch 1500. Loss 0.135208, Training accuracy 0.956522 
    ## Epoch 2000. Loss 0.120115, Training accuracy 0.956522 
    ## Epoch 2500. Loss 0.108294, Training accuracy 0.956522 
    ## Epoch 3000. Loss 0.098384, Training accuracy 0.956522 
    ## Epoch 3500. Loss 0.090175, Training accuracy 0.956522 
    ## Epoch 4000. Loss 0.083230, Training accuracy 0.956522 
    ## Epoch 4500. Loss 0.077113, Training accuracy 0.956522 
    ## Epoch 5000. Loss 0.071429, Validation accuracy 0.875000

``` r
pred_y = predict_DNN(model = iris_dnn, newdata = test_iris[, 1:4], type = "class")
table(test_iris[, 5], pred_y)
```

    ##             pred_y
    ##              setosa versicolor virginica
    ##   setosa         12          0         0
    ##   versicolor      0         20         1
    ##   virginica       0          0        12
