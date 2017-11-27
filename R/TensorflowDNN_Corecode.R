require(tensorflow)

DNN_fun = function(X, y, n, p, nclass, labels, hidden_layer, learning, act_fun, init_fun,
                   batch_size, dropout_rate, epoch, vali_X, vali_y, type, regularization, 
                   verb, save, output) {
  
  sess = tf$Session()
  
  input_X = tf$placeholder(tf$float32, list(NULL, p))
  input_y = tf$placeholder(tf$float32, list(NULL, nclass))
  
  keep_prob = tf$placeholder(tf$float32)
  
  # Setting hidden layer
  .hidden = function(n_feature, n_output, n_node, act_fun, init_fun) {
    
    w_list = layer_list = list()
    w = c(n_feature, n_node, n_output)
    
    for (i in 1:(length(w) - 1)) {
      w_list[[i]] = list(w_p = tf$Variable(tf[[init_fun]](list(w[i], w[i + 1]))),
                         b = tf$Variable(tf[[init_fun]](list(w[i + 1]))))
    }
    
    layer_list[[1]] = input_X
    
    if (is.null(n_node) == FALSE) {
      for (j in 1:length(n_node)) {
        layer_list[[j + 1]] = (tf$matmul(layer_list[[j]], w_list[[j]][["w_p"]]) + w_list[[j]][["b"]]) %>%
          tf$nn[[act_fun]](.) %>%
          tf$nn$dropout(., keep_prob)
      }
    }
    
    out_layer = tf$matmul(layer_list[[length(layer_list)]], 
                          w_list[[length(w_list)]][["w_p"]]) + w_list[[length(w_list)]][["b"]]
    
    return(list(weight = w_list, model = out_layer))
  }
  
  nn_str = .hidden(n_feature = p, n_output = nclass, n_node = hidden_layer, act_fun = act_fun, init_fun)
  
  model = nn_str[["model"]]
  
  # Define loss and accuracy(or MSE)
  if (type == "Classification") {
    loss = tf$reduce_mean(tf$nn$softmax_cross_entropy_with_logits(logits = model, labels = input_y))
    correct_pred = tf$equal(tf$argmax(model, 1L), tf$argmax(input_y, 1L))
    accuracy = tf$reduce_mean(tf$cast(correct_pred, tf$float32))
  }
  
  if (type == "Regression") {
    loss = tf$reduce_sum(tf$square(model - input_y))
    accuracy = tf$reduce_mean(tf$square(model - input_y))
  }
  
  # Define regularization
  if (is.null(regularization) == FALSE) {
    if (regularization[["type"]] == "l2_loss") {
      regularizers = Reduce("+", lapply(nn_str[["weight"]], 
                                        function(x) tf$nn$l2_loss(x[["w_p"]]))) 
    }
    loss_r = loss + regularization[["cost"]] * regularizers
  } else {
    loss_r = loss
  }
  
  
  global_step = tf$Variable(0L)
  
  # Setting learning rate
  learning_rate = tf$train$exponential_decay(learning_rate = learning[["init_learning_rate"]],
                                             global_step = global_step, 
                                             decay_steps = learning[["decay_steps"]], 
                                             decay_rate = learning[["decay_rate"]],
                                             staircase = T)
  
  # Define optimizer
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
      batch_target_y = y[i:max_idx, , drop = FALSE]
      accuracy_val_ = sess$run(list(optimizer, loss_r, accuracy), 
                               feed_dict = dict(input_X = batch_data, 
                                                input_y = batch_target_y, 
                                                keep_prob = dropout_rate))
    }
    
    if (j %% 500 == 0) {
      if ((j %% 5000 == 0) & ((!is.null(vali_X) & !is.null(vali_y)) == TRUE)) {
        vali_acc = accuracy$eval(session = sess, 
                                 feed_dict = dict(input_X = vali_X, 
                                                  input_y = vali_y, 
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
                type = type, labels = labels)))
  }
  
  if (output == 1) {
    return(list(model = list(model_tensor = model, sess = sess, 
                output_tensor = list(X = input_X, y = input_y, keep_prob = keep_prob), 
                type = type, labels = labels), weight = sess$run(nn_str$weight)))
  }
}

