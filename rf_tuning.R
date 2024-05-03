rf_tuning <- function(data, outer_folds, Ntree, Mtry, Nodesize, param){
  Tune_df <- data.frame(matrix(ncol=6, nrow=0))
  colnames(Tune_df) = c(param, 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')
  
  if(param == 'ntree'){
    for (n in Ntree){
      for (i in 1:5){
        print(paste(n, i))
        train <- data[-outer_folds[[i]],]
        train_idx <- createDataPartition(train$is_canceled, p=0.7, list = FALSE)
        train_inner <- train[train_idx , ]
        val_inner  <- train[-train_idx, ]
        train_inner <- onehot_encode(train_inner)
        val_inner <- onehot_encode(val_inner)
        
        RF <- randomForest(is_canceled~., data=train_inner, ntree=n, mtry=Mtry, nodesize=Nodesize)
        RF.pred <- predict(RF, val_inner, type='response')
        RF_CM <- confusionMatrix(RF.pred, val_inner$is_canceled)
        
        # train_inner_balance <- smotenc(train_inner, var = 'is_canceled', k = 5, over_ratio = 1)
        # train_inner_balance <- onehot_encode(train_inner_balance)
        # val_inner_balance <- smotenc(val_inner, var = 'is_canceled', k = 5, over_ratio = 1)
        # val_inner_balance <- onehot_encode(val_inner_balance)
        # RF <- randomForest(is_canceled~., data=train_inner_balance, ntree=n, mtry=Mtry, nodesize=Nodesize)
        # RF.pred <- predict(RF, val_inner_balance, type='response')
        # RF_CM <- confusionMatrix(RF.pred, val_inner_balance$is_canceled)
        
        Tune_df[nrow(Tune_df)+1, ] = c(n, i,
                                       RF_CM$overall['Accuracy'], RF_CM$byClass['Precision'],
                                       RF_CM$byClass['Recall'], RF_CM$byClass['F1'])
      }
    }
  } else if (param == 'mtry') {
    for (n in Mtry){
      for (i in 1:5){
        print(paste(n, i))
        train <- data[-outer_folds[[i]],]
        train_idx <- createDataPartition(train$is_canceled, p=0.7, list = FALSE)
        train_inner <- train[train_idx , ]
        val_inner  <- train[-train_idx, ]
        train_inner <- onehot_encode(train_inner)
        val_inner <- onehot_encode(val_inner)
        
        RF <- randomForest(is_canceled~., data=train_inner, ntree=Ntree, mtry=n, nodesize=Nodesize)
        RF.pred <- predict(RF, val_inner, type='response')
        RF_CM <- confusionMatrix(RF.pred, val_inner$is_canceled)
        Tune_df[nrow(Tune_df)+1, ] = c(n, i,
                                       RF_CM$overall['Accuracy'], RF_CM$byClass['Precision'],
                                       RF_CM$byClass['Recall'], RF_CM$byClass['F1'])
      }
    }
  } else if (param == 'nodesize') {
    for (n in Nodesize){
      for (i in 1:5){
        print(paste(n, i))
        train <- data[-outer_folds[[i]],]
        train_idx <- createDataPartition(train$is_canceled, p=0.7, list = FALSE)
        train_inner <- train[train_idx , ]
        val_inner  <- train[-train_idx, ]
        train_inner <- onehot_encode(train_inner)
        val_inner <- onehot_encode(val_inner)
        
        RF <- randomForest(is_canceled~., data=train_inner, ntree=Ntree, mtry=Mtry, nodesize=n)
        RF.pred <- predict(RF, val_inner, type='response')
        RF_CM <- confusionMatrix(RF.pred, val_inner$is_canceled)
        Tune_df[nrow(Tune_df)+1, ] = c(n, i,
                                       RF_CM$overall['Accuracy'], RF_CM$byClass['Precision'],
                                       RF_CM$byClass['Recall'], RF_CM$byClass['F1'])
      }
    }
  } else {
    print('Error from input parameters')
  }
  
  return(Tune_df)
}