catb_tuning <- function(data, train_ctr, outer_folds, param_grid){
  
  Tune_df <- data.frame(matrix(ncol=7, nrow=0))
  colnames(Tune_df) = c('fold', 'iterations', 'learning_rate', 'depth', 
                        'l2_leaf_reg', 'rsm', 'border_count')
  
  for (i in 1:5){
    print(i)
    train <- data[-outer_folds[[i]],] 
    train_idx <- createDataPartition(train$is_canceled, p=0.7, list = FALSE)
    train_inner <- train[train_idx , ]
    
    catBModel <- train(train_inner %>% select(-is_canceled), 
                       as.factor(make.names(train_inner$is_canceled)),
                       method = catboost.caret,
                       tuneGrid = param_grid, 
                       trControl = train_ctr,
                       verbose = 0)
    
    Tune_df[nrow(Tune_df)+1, ] = c(i, 
                                   catBModel$bestTune$iterations, 
                                   catBModel$bestTune$learning_rate,
                                   catBModel$bestTune$depth,
                                   catBModel$bestTune$l2_leaf_reg,
                                   catBModel$bestTune$rsm,
                                   catBModel$bestTune$border_count)
    

  }
  return(Tune_df)
}