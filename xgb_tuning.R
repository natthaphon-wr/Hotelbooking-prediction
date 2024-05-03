xgb_tuning <- function(data, train_ctr, outer_folds, param_grid){
  
  Tune_df <- data.frame(matrix(ncol=8, nrow=0))
  colnames(Tune_df) = c('fold', 'nrounds', 'eta', 'max_depth', 'min_child_weight', 
                        'gamma', 'subsample', 'colsample_bytree')
  
  for (i in 1:5){
    print(i)
    train <- data[-outer_folds[[i]],] 
    train_idx <- createDataPartition(train$is_canceled, p=0.7, list = FALSE)
    train_inner <- train[train_idx , ]
    train_inner <- onehot_encode(train_inner)
    train_inner$is_canceled <- unclass(train_inner$is_canceled)%%2
    
    xgbModel <- train(is_canceled~.,
                      data = train_inner,
                      method = "xgbTree", 
                      trControl = train_ctr, 
                      tuneGrid = param_grid,
                      verbosity = 0)
    
    Tune_df[nrow(Tune_df)+1, ] = c(i, 
                                   xgbModel$bestTune$nrounds,
                                   xgbModel$bestTune$eta,
                                   xgbModel$bestTune$max_depth,
                                   xgbModel$bestTune$min_child_weight,
                                   xgbModel$bestTune$gamma,
                                   xgbModel$bestTune$subsample, 
                                   xgbModel$bestTune$colsample_bytree
                                   )
  }
}