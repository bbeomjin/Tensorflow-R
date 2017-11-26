rm(list = ls())
gc()

setwd("J:\\R\\tensorflowR")
source("./tensorflow DNN.R")

data("iris")
ind = sample(nrow(iris), size = 0.7 * nrow(iris))
train_iris = iris[ind, ]
test_iris = iris[-ind, ]

ind_v = sample(nrow(train_iris), size = 0.7 * nrow(train_iris))
vali_iris = train_iris[-ind_v, ]
train_iris = train_iris[ind_v, ]

validation_set = list(X = vali_iris[, 1:4], y = vali_iris$Species)
hidden_layer = c(10L, 10L)

iris_dnn = tf_DNN(X = train_iris[, 1:4], y = train_iris$Species, hidden_layer, 
               validation_set = validation_set, 
               regularization = list(type = "l2_loss", cost = 0), 
               drop_rate = 0, epoch = 10000, output = 1)

pred_y = predict_DNN(model = iris_dnn, newdata = test_iris[, 1:4], type = "class")
table(test_iris[, 5], pred_y)

