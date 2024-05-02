gbm_tuning <- function(data, outer_folds, Ntrees, Shrinkage, Nminobsinnode, Bagfraction, Threshold, param){
  
  if(param == 'initial'){
    print('Initial Tuning')
    Tune_df <- data.frame(matrix(ncol=7, nrow=0))
    colnames(Tune_df) = c('n.trees', 'shrinkage', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')
    
    for (n in Ntrees){
      for (lr in Shrinkage){
        for (i in 1:5){
          print(paste(n, lr, i))
          train <- data[-outer_folds[[i]],] 
          train_idx <- createDataPartition(train$is_canceled, p=0.7, list=FALSE)
          train_inner <- train[train_idx , ]
          val_inner  <- train[-train_idx, ]
          train_inner <- onehot_encode(train_inner)
          val_inner <- onehot_encode(val_inner)
          train_inner$is_canceled <- unclass(train_inner$is_canceled)%%2 
          val_inner$is_canceled <- unclass(val_inner$is_canceled)%%2 
          
          # train_inner_balance <- smotenc(train_inner, var = 'is_canceled', k = 5, over_ratio = 1)
          # train_inner_balance <- onehot_encode(train_inner_balance)
          # train_inner_balance$is_canceled <- unclass(train_inner_balance$is_canceled)%%2 
          # val_inner_balance <- smotenc(val_inner, var = 'is_canceled', k = 5, over_ratio = 1)
          # val_inner_balance <- onehot_encode(val_inner_balance)
          # val_inner_balance$is_canceled <- unclass(val_inner_balance$is_canceled)%%2 
          # GBM <- gbm(is_canceled~., data=train_inner_balance, distribution="bernoulli", n.trees=n, 
          #            shrinkage=lr, n.minobsinnode=Nminobsinnode,  bag.fraction=Bagfraction, verbose=FALSE)
          # GBM.pred <- predict.gbm(GBM, val_inner_balance, type='response', verbose=FALSE)
          # GBM.pred[GBM.pred < Threshold] <- 0
          # GBM.pred[GBM.pred >= Threshold] <- 1
          # GBM.pred <- factor(GBM.pred, levels = c(1,0))
          # val_inner_balance$is_canceled <- factor(val_inner_balance$is_canceled, levels = c(1,0))
          # GBM_CM <- confusionMatrix(GBM.pred, val_inner_balance$is_canceled)

          GBM <- gbm(is_canceled~., data=train_inner, distribution="bernoulli", n.trees=n, 
                     shrinkage=lr, n.minobsinnode=Nminobsinnode,  bag.fraction=Bagfraction, verbose=FALSE)
          # summary(GBM) #compute relative inference of each variable
          
          GBM.pred <- predict.gbm(GBM, val_inner, type='response', verbose=FALSE)
          GBM.pred[GBM.pred < Threshold] <- 0
          GBM.pred[GBM.pred >= Threshold] <- 1
          GBM.pred <- factor(GBM.pred, levels = c(1,0))
          val_inner$is_canceled <- factor(val_inner$is_canceled, levels = c(1,0))
          GBM_CM <- confusionMatrix(GBM.pred, val_inner$is_canceled)
          Tune_df[nrow(Tune_df)+1, ] = c(n, lr, i,
                                         GBM_CM$overall['Accuracy'], GBM_CM$byClass['Precision'],
                                         GBM_CM$byClass['Recall'], GBM_CM$byClass['F1'])
        }
      }
    }

  } else if (param == 'n.minobsinnode') {
    Tune_df <- data.frame(matrix(ncol=6, nrow=0))
    colnames(Tune_df) = c('n.minobsinnode', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')
    for (nmin in Nminobsinnode){
      for (i in 1:5){
        print(paste(nmin, i))
        train <- data[-outer_folds[[i]],] 
        train_idx <- createDataPartition(train$is_canceled, p=0.7, list=FALSE)
        train_inner <- train[train_idx , ]
        val_inner  <- train[-train_idx, ]
        train_inner <- onehot_encode(train_inner)
        val_inner <- onehot_encode(val_inner)
        train_inner$is_canceled <- unclass(train_inner$is_canceled)%%2 
        val_inner$is_canceled <- unclass(val_inner$is_canceled)%%2 
        
        GBM <- gbm(is_canceled~., data=train_inner, distribution="bernoulli", n.trees=Ntrees, 
                   shrinkage=Shrinkage, n.minobsinnode=nmin, bag.fraction=Bagfraction, verbose=FALSE)
        # summary(GBM) #compute relative inference of each variable
        
        GBM.pred <- predict.gbm(GBM, val_inner, type='response', verbose=FALSE)
        GBM.pred[GBM.pred < Threshold] <- 0
        GBM.pred[GBM.pred >= Threshold] <- 1
        GBM.pred <- factor(GBM.pred, levels = c(1,0))
        
        val_inner$is_canceled <- factor(val_inner$is_canceled, levels = c(1,0))
        GBM_CM <- confusionMatrix(GBM.pred, val_inner$is_canceled)
        Tune_df[nrow(Tune_df)+1, ] = c(nmin, i,
                                       GBM_CM$overall['Accuracy'], GBM_CM$byClass['Precision'],
                                       GBM_CM$byClass['Recall'], GBM_CM$byClass['F1'])
      }
    }
    
  } else if (param == 'bag.fraction') {
    Tune_df <- data.frame(matrix(ncol=6, nrow=0))
    colnames(Tune_df) = c('bag.fraction', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')
    for (bf in Bagfraction){
      for (i in 1:5){
        print(paste(bf, i))
        train <- data[-outer_folds[[i]],] 
        train_idx <- createDataPartition(train$is_canceled, p=0.7, list=FALSE)
        train_inner <- train[train_idx , ]
        val_inner  <- train[-train_idx, ]
        train_inner <- onehot_encode(train_inner)
        val_inner <- onehot_encode(val_inner)
        train_inner$is_canceled <- unclass(train_inner$is_canceled)%%2 
        val_inner$is_canceled <- unclass(val_inner$is_canceled)%%2 
        
        GBM <- gbm(is_canceled~., data=train_inner, distribution="bernoulli", n.trees=Ntrees, 
                   shrinkage=Shrinkage, n.minobsinnode=Nminobsinnode, bag.fraction=bf, verbose=FALSE)
        # summary(GBM) #compute relative inference of each variable
        
        GBM.pred <- predict.gbm(GBM, val_inner, type='response', verbose=FALSE)
        GBM.pred[GBM.pred < Threshold] <- 0
        GBM.pred[GBM.pred >= Threshold] <- 1
        GBM.pred <- factor(GBM.pred, levels = c(1,0))
        
        val_inner$is_canceled <- factor(val_inner$is_canceled, levels = c(1,0))
        GBM_CM <- confusionMatrix(GBM.pred, val_inner$is_canceled)
        Tune_df[nrow(Tune_df)+1, ] = c(bf, i, 
                                       GBM_CM$overall['Accuracy'], GBM_CM$byClass['Precision'],
                                       GBM_CM$byClass['Recall'], GBM_CM$byClass['F1'])
      }
    }
    
  } else if (param == 'threshold'){
    Tune_df <- data.frame(matrix(ncol=6, nrow=0))
    colnames(Tune_df) = c('threshold', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')
    for (thresh in Threshold){
      for (i in 1:5){
        print(paste(thresh, i))
        train <- data[-outer_folds[[i]],] 
        train_idx <- createDataPartition(train$is_canceled, p=0.7, list=FALSE)
        train_inner <- train[train_idx , ]
        val_inner  <- train[-train_idx, ]
        train_inner <- onehot_encode(train_inner)
        val_inner <- onehot_encode(val_inner)
        train_inner$is_canceled <- unclass(train_inner$is_canceled)%%2 
        val_inner$is_canceled <- unclass(val_inner$is_canceled)%%2 
        
        GBM <- gbm(is_canceled~., data=train_inner, distribution="bernoulli", n.trees=Ntrees, 
                   shrinkage=Shrinkage, n.minobsinnode=Nminobsinnode, bag.fraction=Bagfraction, verbose=FALSE)
        # summary(GBM) #compute relative inference of each variable
        
        GBM.pred <- predict.gbm(GBM, val_inner, type='response', verbose=FALSE)
        GBM.pred[GBM.pred < thresh] <- 0
        GBM.pred[GBM.pred >= thresh] <- 1
        GBM.pred <- factor(GBM.pred, levels = c(1,0))
        
        val_inner$is_canceled <- factor(val_inner$is_canceled, levels = c(1,0))
        GBM_CM <- confusionMatrix(GBM.pred, val_inner$is_canceled)
        Tune_df[nrow(Tune_df)+1, ] = c(thresh, i, 
                                       GBM_CM$overall['Accuracy'], GBM_CM$byClass['Precision'],
                                       GBM_CM$byClass['Recall'], GBM_CM$byClass['F1'])
      }
    }

  } else {
    print('Error from input parameters')
  }
  
  return(Tune_df)
}