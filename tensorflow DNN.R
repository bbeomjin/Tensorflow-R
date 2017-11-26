require(tensorflow)

tf_DNN = function(X, y, hidden_layer, 
                   learning = list(optimizer_type = "GradientDescentOptimizer", init_learning_rate = 0.001, decay_rate = 0.96, decay_steps = 1e+5),
                   batch_size = 50L, drop_rate = 0.1,  
                   epoch = 20000, validation_set = NULL, type = "Classification",
                   regularization = list(type = "l2_loss", cost = 0.1), verb = TRUE, save = FALSE,
                   output = 1) {
  
  if (require(dplyr) == FALSE) {install.packages("dplyr")}
  require(dplyr)
  
  if (is.null(names(learning)) == TRUE) {names(learning) = c("optimizer_type",
                                                             "init_learning_rate",
                                                             "decay_rate",
                                                             "decay_steps")}
  
  if (is.null(names(regularization)) == TRUE) {names(regularization) = c("type", "cost")}
  
  sess = tf$Session()
  
  X = as.matrix(X)
  p = ncol(X) %>% as.integer(.)
  n = nrow(X) %>% as.integer(.)
  
  if (type == "Classification") {
    y = as.factor(y)
    f_lev = levels(y)
    nclass = f_lev %>% length(.) %>% as.integer(.)
    y = as.integer(y) - 1L
    onehot_labels = tf$one_hot(y, nclass, 
                               on_value = 1., off_value = 0., axis = -1)
    target_y = sess$run(onehot_labels)
    
  }
  
  if (type == "Regression") {
    nclass = 1L
    f_lev = NULL
    y = as.numeric(y)
    target_y = y %>% as.matrix(.)
  }
  
  if (is.null(validation_set) == FALSE) {
    if (is.null(names(validation_set)) == TRUE) {names(validation_set) = c("X", "y")}
    
    vali_X = validation_set[["X"]] %>% as.matrix(.)
    vali_y = validation_set[["y"]]
    
    if (type == "Classification") {
      vali_y = factor(vali_y, levels = f_lev) %>% as.integer(.) - 1L
      
      if (any(is.na(vali_y)) == TRUE) {warnings("validation labels not equal train labels")}
      
      v_onehot_labels = tf$one_hot(vali_y, nclass, 
                                   on_value = 1., off_value = 0., axis = -1)
      target_vali_y = sess$run(v_onehot_labels)
    }
    
    if (type == "Regression") {
      vali_y = as.numeric(vali_y)
      target_vali_y = vali_y %>% as.matrix(.)
    }
  }
  
  input_X = tf$placeholder(tf$float32, list(NULL, p))
  input_y = tf$placeholder(tf$float32, list(NULL, nclass))
  
  keep_prob = tf$placeholder(tf$float32)
  
  .hidden = function(n_feature, n_output, n_node) {
    
    w_list = layer_list = list()
    w = c(n_feature, n_node, n_output)
    
    for (i in 1:(length(w) - 1)) {
      w_list[[i]] = list(w_p = tf$Variable(tf$random_normal(list(w[i], w[i + 1]))),
                         b = tf$Variable(tf$random_normal(list(w[i + 1]))))
    }
    
    layer_list[[1]] = input_X
    
    if (is.null(n_node) == FALSE) {
      for (j in 1:length(n_node)) {
        layer_list[[j + 1]] = (tf$matmul(layer_list[[j]], w_list[[j]][["w_p"]]) + w_list[[j]][["b"]]) %>%
          tf$nn$relu(.) %>%
          tf$nn$dropout(., keep_prob)
      }
    }
    
    out_layer = tf$matmul(layer_list[[length(layer_list)]], 
                          w_list[[length(w_list)]][["w_p"]]) + w_list[[length(w_list)]][["b"]]
    
    return(list(weight = w_list, model = out_layer))
  }
  
  nn_str = .hidden(n_feature = p, n_output = nclass, n_node = hidden_layer)
  
  model = nn_str[["model"]]
  
  if (type == "Classification") {
    loss = tf$reduce_mean(tf$nn$softmax_cross_entropy_with_logits(logits = model, labels = input_y))
    correct_pred = tf$equal(tf$argmax(model, 1L), tf$argmax(input_y, 1L))
    accuracy = tf$reduce_mean(tf$cast(correct_pred, tf$float32))
  }
  
  if (type == "Regression") {
    loss = tf$reduce_sum(tf$square(model - input_y))
    accuracy = tf$reduce_mean(tf$square(model - input_y))
  }
  
  if (regularization[["type"]] == "l2_loss") {
    regularizers = Reduce("+", lapply(nn_str[["weight"]], 
                                      function(x) tf$nn$l2_loss(x[["w_p"]]))) 
  }
  
  loss_r = loss + regularization[["cost"]] * regularizers
  
  global_step = tf$Variable(0L)
  learning_rate = tf$train$exponential_decay(learning_rate = learning[["init_learning_rate"]],
                                             global_step = global_step, 
                                             decay_steps = learning[["decay_steps"]], 
                                             decay_rate = learning[["decay_rate"]],
                                             staircase = T)

  optimizer = tf$train[[learning[["optimizer_type"]]]](learning_rate)$minimize(loss_r, global_step = global_step)
  
  init = tf$global_variables_initializer()
  
  sess$run(init)
  
  for (j in 1:epoch) {
    for (i in seq(1, nrow(X), batch_size)) {
      
      if (i + batch_size > nrow(X)) {
        max_idx = nrow(X)
      } else {
        max_idx = i + batch_size 
      }

      batch_data = X[i:max_idx, ]
      batch_target_y = target_y[i:max_idx, , drop = F]
      accuracy_val_ = sess$run(list(optimizer, loss_r, accuracy), 
                               feed_dict = dict(input_X = batch_data, 
                                                input_y = batch_target_y, 
                                                keep_prob = (1 - drop_rate)))
    }
    
    if (j %% 500 == 0) {
      if ((j %% 5000 == 0) & (is.null(validation_set) == FALSE)) {
        vali_acc = accuracy$eval(session = sess, 
                                 feed_dict = dict(input_X = vali_X, 
                                                  input_y = target_vali_y, 
                                                  keep_prob = 1.0))
        txt = sprintf('Epoch %d. Loss %f, Validation accuracy %f', j, accuracy_val_[[2]], vali_acc)
      } else {
        txt = sprintf('Epoch %d. Loss %f, Training accuracy %f', j, accuracy_val_[[2]], accuracy_val_[[3]])
      }
      
      if (verb == TRUE) {
        cat(paste(txt, "\n", sep = " "))
      }
    }
  }
  
  if (save == TRUE) {
    saver = tf$train$Saver()
    saver$save(sess = sess, save_path = paste(getwd(), "/", "my_model", sep = ""), global_step = j)
  }
  
  if (output == 0) {
    return(list(model = list(model_tensor = model, sess = sess, 
                output_tensor = list(X = input_X, y = input_y, keep_prob = keep_prob), 
                type = type, labels = f_lev)))
  }
  
  if (output == 1) {
    return(list(model = list(model_tensor = model, sess = sess, 
                output_tensor = list(X = input_X, y = input_y, keep_prob = keep_prob), 
                type = type, labels = f_lev), weight = sess$run(nn_str$weight)))
  }
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
                                                     keep_prob = 1.))
    
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
                                                     keep_prob = 1.))
  }
  
  return(result)
}


