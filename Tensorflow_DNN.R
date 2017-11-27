tensorflow_DNN = function(X, y, hidden_layer, 
                  learning = list(optimizer_type = "GradientDescentOptimizer", init_learning_rate = 0.001, decay_rate = 0.96, decay_steps = 1e+5),
                  act_fun = "relu", init_fun = "truncated_normal",
                  batch_size = 50, dropout_rate = 1.0,  
                  epoch = 20000, validation_set = NULL, type = "Classification",
                  regularization = list(type = "l2_loss", cost = 0.0), verb = TRUE, save = FALSE,
                  output = 1) {
  
  if (require(dplyr) == FALSE) {install.packages("dplyr")}
  require(dplyr)
  
  if (require(tensorflow) == FALSE) {stop("You must install tensorflow under R")}
  
  source("./R/TensorflowDNN_Corecode.R", encoding = "UTF-8")
  
  if (is.null(names(learning)) == TRUE) {names(learning) = c("optimizer_type",
                                                             "init_learning_rate",
                                                             "decay_rate",
                                                             "decay_steps")}
  
  if (is.null(names(regularization)) == TRUE) {names(regularization) = c("type", "cost")}
  
  if (!(type %in% c("Classification", "Regression"))) {
    warning("Not Support type, Default : Classification")
    type = "Classification"
    }
  
  if (verb == TRUE) {
    cat("--------------------------------------------------- \n")
    cat("Setting Parameter for", type, "Neural Network", "\n\n",
        "<Layer Parameter> \n", "Activation Function:", act_fun, "\n",
        "Initial Function:", init_fun, "\n", "Dropout Rate:", dropout_rate, "\n",
        paste("Layer", 1:length(hidden_layer), ":", hidden_layer, "\n", sep = ""),
        "\n", "<Optimization Parameter> \n", 
        "Optimizer:",learning[["optimizer_type"]], "\n", 
        "Learning Rate:", learning[["init_learning_rate"]], "\n",
        "Decay Rate:", learning[["decay_rate"]], "\n", "Decay steps:", learning[["decay_steps"]], "\n",
        "Epoch:", epoch, "\n \n",
        "<Regularization Parameter> \n", "Regularizer:", regularization[["type"]], "\n",
        "Cost:", regularization[["cost"]], "\n")
    cat("--------------------------------------------------- \n")
  }
  
  
  target_X = as.matrix(X)
  
  p = ncol(X) %>% as.integer(.)
  n = nrow(X) %>% as.integer(.)
  
  if (is.null(hidden_layer) == TRUE) {
    hidden_layer = NULL
  } else{
    hidden_layer = as.integer(hidden_layer)  
  }
  
  if (is.null(batch_size) == TRUE) {
    batch_size = nrow(X)
  }
  batch_size = as.integer(batch_size)
  
  if (type == "Classification") {
    y = as.factor(y)
    f_lev = levels(y)
    nclass = f_lev %>% length(.) %>% as.integer(.)
    
    target_y = model.matrix(~ y - 1)
    
  }
  
  if (type == "Regression") {
    nclass = 1L
    f_lev = NULL
    y = as.numeric(y)
    target_y = y %>% as.matrix(.)
  }
  
  if (is.null(validation_set) == FALSE) {
    if (is.null(names(validation_set)) == TRUE) {names(validation_set) = c("X", "y")}
    
    target_vali_X = validation_set[["X"]] %>% as.matrix(.)
    vali_y = validation_set[["y"]]
    
    if (type == "Classification") {
      vali_y = factor(vali_y, levels = f_lev)
      
      if (any(is.na(vali_y)) == TRUE) {stop("Validation labels not equal train labels")}
      
      target_vali_y = model.matrix(~ vali_y - 1)
    }
    
    if (type == "Regression") {
      vali_y = as.numeric(vali_y)
      target_vali_y = vali_y %>% as.matrix(.)
    }
  } else {
    vali_X = vali_y = NULL
  }
  
  result = DNN_fun(X = target_X, y = target_y, n = n, p = p, nclass = nclass, labels = f_lev, 
                   hidden_layer = hidden_layer, learning = learning, act_fun = act_fun, init_fun,
                   batch_size = batch_size, dropout_rate = dropout_rate, epoch = epoch, 
                   vali_X = target_vali_X, vali_y = target_vali_y, type = type, 
                   regularization = regularization,
                   verb = verb, save = save, output = output)
   
  return(result)
}

predict_DNN = function(model, newdata, type = "class") {
  
  model_tensor = model[["model"]][["model_tensor"]]
  sess = model[["model"]][["sess"]]
  input_X = model[["model"]][["output_tensor"]][["X"]]
  keep_prob = model[["model"]][["output_tensor"]][["keep_prob"]]
  y_labels = model[["model"]][["labels"]]
  
  newdata = as.matrix(newdata)
  
  if (model[["model"]][["type"]] == "Classification") {
    pred_y = sess$run(model_tensor, feed_dict = dict(input_X = newdata,
                                                     keep_prob = 1.0))
    
    if (type == "class") {
      result = sess$run(tf$argmax(pred_y, 1L)) %>% factor(., levels = 0:(length(y_labels) - 1),
                                                          labels = y_labels)
    }
    
    if (type == "raw") {
      result = pred_y
      colnames(result) = y_labels
    }
  }
  
  if (model[["model"]][["type"]] == "Regression") {
    result = sess$run(model_tensor, feed_dict = dict(input_X = newdata,
                                                     keep_prob = 1.0))
  }
  
  return(result)
}
