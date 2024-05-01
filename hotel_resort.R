source('preprocess.R')
source('onehot_encode.R')
library(tree)
library(tidyverse) 
library(caret)
library(randomForest)
library(gbm)
library(xgboost)
library(lightgbm)
library(data.table)
library(catboost)
library(themis)


# 1. Import data ----
## Not 1 hot encoding ----
hotel_data <- preprocess(one_hot = FALSE, feature_select = TRUE)
resort <- hotel_data[[2]]
sapply(resort, class)
summary(resort$is_canceled)

## 1 hot encoding ----
hotel_data_1hot <- preprocess(one_hot = TRUE, feature_select = TRUE)
resort_1hot <- hotel_data_1hot[[2]]
sapply(resort_1hot, class)
summary(resort_1hot$is_canceled)


# 2. Outer loop splitting ----
outer_folds <- createFolds(resort$is_canceled, k=5)


# 3 Decision Tree ----
DT_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(DT_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
for (i in 1:5){
  print(i)
  train <- resort[-outer_folds[[i]],] 
  train_balance <- smotenc(train, var = 'is_canceled', k = 5, over_ratio = 1)
  train_balance <- onehot_encode(train_balance)
  test <- resort[outer_folds[[i]],] 
  test <- onehot_encode(test)
  
  trees <- tree(is_canceled~., data=train_balance, split=c("deviance", "gini"))
  # summary(trees)
  
  cv.trees <- cv.tree(trees, FUN=prune.misclass, K=5)
  prune.trees <- prune.misclass(trees, best=cv.trees$size[which.min(cv.trees$dev)])
  # plot(cv.trees)
  # plot(prune.trees)
  # text(prune.trees, pretty=1)
  
  tree.pred <- predict(prune.trees, test, type='class')
  CM <- confusionMatrix(tree.pred, test$is_canceled)
  DT_result[nrow(DT_result)+1, ] = c(CM$overall['Accuracy'], CM$byClass['Precision'],
                                     CM$byClass['Recall'], CM$byClass['F1'])
}
colMeans(DT_result)
write.csv(DT_result, ".\\result\\DT_resort_Evaluation.csv", row.names=FALSE)
summary(prune.trees)$used


# 4 Random forest ----
# Parameters Tuning
#   - ntree = no.of trees (row samples size = nrow(train))
#   - mtry = no. features that samples
#   - nodesize  = min size of terminal nodes

## 4.1 Tune ntree ----
RF_ntree <- data.frame(matrix(ncol=6, nrow=0))
colnames(RF_ntree) = c('ntree', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')
for (n in seq(100, 500, 200)){
  for (i in 1:5){
    print(paste(n, i))
    train <- resort[-outer_folds[[i]],]
    train_idx <- createDataPartition(train$is_canceled, p=0.7, list = FALSE)
    train_inner <- train[train_idx , ]
    train_inner_balance <- smotenc(train_inner, var = 'is_canceled', k = 5, over_ratio = 1)
    train_inner_balance <- onehot_encode(train_inner_balance)
    val_inner  <- train[-train_idx, ]
    val_inner <- onehot_encode(val_inner)
    
    RF <- randomForest(is_canceled~., data=train_inner_balance, ntree=n, mtry=10, nodesize=1)
    RF.pred <- predict(RF, val_inner, type='response')
    RF_CM <- confusionMatrix(RF.pred, val_inner$is_canceled)
    RF_ntree[nrow(RF_ntree)+1, ] = c(n, i,
                                       RF_CM$overall['Accuracy'], RF_CM$byClass['Precision'],
                                       RF_CM$byClass['Recall'], RF_CM$byClass['F1'])
  }
}
write.csv(RF_ntree, ".\\result\\RF_resort_ntree.csv", row.names=FALSE)
RF_ntree %>% 
  group_by(ntree) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))
  
## 4.2 Tune mtry  ----
RF_mtry <- data.frame(matrix(ncol=6, nrow=0))
colnames(RF_mtry) = c('mtry', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')
for (n in c(5, 10, 20, 30)){
  for (i in 1:5){
    print(paste(n, i))
    train <- resort[-outer_folds[[i]],]
    train_idx <- createDataPartition(train$is_canceled, p=0.7, list = FALSE)
    train_inner <- train[train_idx , ]
    train_inner_balance <- smotenc(train_inner, var = 'is_canceled', k = 5, over_ratio = 1)
    train_inner_balance <- onehot_encode(train_inner_balance)
    val_inner  <- train[-train_idx, ]
    val_inner <- onehot_encode(val_inner)
    
    RF <- randomForest(is_canceled~., data=train_inner_balance, ntree=500, mtry=n, nodesize=1)
    RF.pred <- predict(RF, val_inner, type='response')
    RF_CM <- confusionMatrix(RF.pred, val_inner$is_canceled)
    RF_mtry[nrow(RF_mtry)+1, ] = c(n, i,
                                   RF_CM$overall['Accuracy'], RF_CM$byClass['Precision'],
                                   RF_CM$byClass['Recall'], RF_CM$byClass['F1'])
  }
}
write.csv(RF_mtry, ".\\result\\RF_resort_mtry.csv", row.names=FALSE)
RF_mtry %>% 
  group_by(mtry) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))

## 4.3 Tune nodesize ----
RF_nodesize <- data.frame(matrix(ncol=6, nrow=0))
colnames(RF_nodesize) = c('nodesize', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')
for (n in c(1, 5, 10, 20, 50)){
  for (i in 1:5){
    print(paste(n, i))
    train <- resort[-outer_folds[[i]],]
    train_idx <- createDataPartition(train$is_canceled, p=0.7, list = FALSE)
    train_inner <- train[train_idx , ]
    train_inner_balance <- smotenc(train_inner, var = 'is_canceled', k = 5, over_ratio = 1)
    train_inner_balance <- onehot_encode(train_inner_balance)
    val_inner  <- train[-train_idx, ]
    val_inner <- onehot_encode(val_inner)
    
    RF <- randomForest(is_canceled~., data=train_inner_balance, ntree=500, mtry=20, nodesize=n)
    RF.pred <- predict(RF, val_inner, type='response')
    RF_CM <- confusionMatrix(RF.pred, val_inner$is_canceled)
    RF_nodesize[nrow(RF_nodesize)+1, ] = c(n, i,
                                           RF_CM$overall['Accuracy'], RF_CM$byClass['Precision'],
                                           RF_CM$byClass['Recall'], RF_CM$byClass['F1'])
  }
}
write.csv(RF_nodesize, ".\\result\\RF_resort_nodesize.csv", row.names=FALSE)
RF_nodesize %>% 
  group_by(nodesize) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))

## 4.4 Final RF model ----
RF_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(RF_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
# RF_imp <- matrix(0, nrow = ncol(resort)-1, ncol=1)
# rownames(RF_imp) <- colnames(resort %>% select(-is_canceled))
# colnames(RF_imp) <- c('MeanDecreaseGini')

### Test SMOTE-NC ----
for (i in 1:5){
  print(i)
  train <- resort[-outer_folds[[i]],] 
  train_balance <- smotenc(train, var = 'is_canceled', k = 5, over_ratio = 1)
  train_balance <- onehot_encode(train_balance)
  test <- resort[outer_folds[[i]],] 
  test <- onehot_encode(test)
  
  RF <- randomForest(is_canceled~., data=train_balance, ntree=500, mtry=20, nodesize=1)
  # RF_imp <- RF_imp + importance(RF)
  
  RF.pred <- predict(RF, test, type='response')
  RF_CM <- confusionMatrix(RF.pred, test$is_canceled)
  RF_result[nrow(RF_result)+1, ] = c(RF_CM$overall['Accuracy'], RF_CM$byClass['Precision'],
                                     RF_CM$byClass['Recall'], RF_CM$byClass['F1'])
}
colMeans(RF_result)
write.csv(RF_result, ".\\result\\RF_resort_Evaluation.csv", row.names=FALSE)


### Normal ----

for (i in 1:5){
  print(i)
  train <- resort[-outer_folds[[i]],]
  test <- resort[outer_folds[[i]],]

  RF <- randomForest(is_canceled~., data=train, ntree=300, mtry=20, nodesize=1)
  RF_imp <- RF_imp + importance(RF)

  RF.pred <- predict(RF, test, type='response')
  RF_CM <- confusionMatrix(RF.pred, test$is_canceled)
  RF_result[nrow(RF_result)+1, ] = c(RF_CM$overall['Accuracy'], RF_CM$byClass['Precision'],
                                     RF_CM$byClass['Recall'], RF_CM$byClass['F1'])
}
colMeans(RF_result)
write.csv(RF_result, ".\\result\\RF_resort_Evaluation.csv", row.names=FALSE)

# Feature importance
RF_imp <- RF_imp/5
RF_imp <- data.frame('MeanDecreaseGini' = RF_imp)
RF_imp20 <- top_n(RF_imp, 20, MeanDecreaseGini)
ggplot(RF_imp20, aes(y=reorder(rownames(RF_imp20), MeanDecreaseGini), x=MeanDecreaseGini)) +
  geom_bar(stat = "identity") +
  ylab('Features') +
  ggtitle('Resort Hotel using Random Forest')
write.csv(RF_imp, ".\\result\\RF_resort_FI.csv", row.names=FALSE)

# 5. Gradient Boosting ----
# Parameters Tuning
#   - classification threshold 
#   - n.trees = total no. of tree
#   - shrinkage = learning rate
#   - n.minobsinnode = min no. of observations in terminal node
#   - bag.fraction = fraction of training set to be selected to build next tree

## GBM dataset ----
resort_GBM <- resort
resort_GBM$is_canceled <- unclass(resort_GBM$is_canceled)%%2 

## 5.1 Tuning n.trees and shrinkage -----
GBM_tune1 <- data.frame(matrix(ncol=7, nrow=0))
colnames(GBM_tune1) = c('n.trees', 'shrinkage', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')
for (n in seq(300, 500, 200)){
  for (lr in c(0.3, 0.4, 0.5)){
    for (i in 1:5){
      print(paste(n, lr, i))
      train <- resort_GBM[-outer_folds[[i]],] 
      train_idx <- createDataPartition(train$is_canceled, p=0.7, list=FALSE)
      train_inner <- train[train_idx,]
      val_inner <- train[-train_idx,]
      
      GBM <- gbm(is_canceled~., data=train_inner, distribution="bernoulli", n.trees=n, 
                 n.minobsinnode=10, shrinkage=lr, bag.fraction=0.5, verbose=FALSE)
      # summary(GBM) #compute relative inference of each variable
      
      GBM.pred <- predict.gbm(GBM, val_inner, type='response', verbose=FALSE)
      GBM.pred[GBM.pred < 0.5] <- 0
      GBM.pred[GBM.pred >= 0.5] <- 1
      GBM.pred <- factor(GBM.pred, levels = c(1,0))
      
      val_inner$is_canceled <- factor(val_inner$is_canceled, levels = c(1,0))
      GBM_CM <- confusionMatrix(GBM.pred, val_inner$is_canceled)
      GBM_tune1[nrow(GBM_tune1)+1, ] = c(n, lr, i,
                                         GBM_CM$overall['Accuracy'], GBM_CM$byClass['Precision'],
                                         GBM_CM$byClass['Recall'], GBM_CM$byClass['F1'])
    }
  }
}
write.csv(GBM_tune1, ".\\result\\GBM_resort_ntreelr.csv", row.names=FALSE)
GBM_tune1 %>% 
  group_by(n.trees, shrinkage) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))

## 5.2 Tuning n.minobsinnode -----
GBM_tune2 <- data.frame(matrix(ncol=6, nrow=0))
colnames(GBM_tune2) = c('n.minobsinnode', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')
for (nmin in c(50, 100, 150, 200)){
  for (i in 1:5){
    print(paste(nmin, i))
    train <- resort_GBM[-outer_folds[[i]],] 
    train_idx <- createDataPartition(train$is_canceled, p=0.7, list=FALSE)
    train_inner <- train[train_idx,]
    val_inner <- train[-train_idx,]
    
    GBM <- gbm(is_canceled~., data=train_inner, distribution="bernoulli", n.trees=500, 
               n.minobsinnode=nmin, shrinkage=0.2, bag.fraction=0.5, verbose=FALSE)
    # summary(GBM) #compute relative inference of each variable
    
    GBM.pred <- predict.gbm(GBM, val_inner, type='response', verbose=FALSE)
    GBM.pred[GBM.pred < 0.5] <- 0
    GBM.pred[GBM.pred >= 0.5] <- 1
    GBM.pred <- factor(GBM.pred, levels = c(1,0))
    
    val_inner$is_canceled <- factor(val_inner$is_canceled, levels = c(1,0))
    GBM_CM <- confusionMatrix(GBM.pred, val_inner$is_canceled)
    GBM_tune2[nrow(GBM_tune2)+1, ] = c(nmin, i,
                                       GBM_CM$overall['Accuracy'], GBM_CM$byClass['Precision'],
                                       GBM_CM$byClass['Recall'], GBM_CM$byClass['F1'])
  }
}
write.csv(GBM_tune2, ".\\result\\GBM_resort_nmins.csv", row.names=FALSE)
GBM_tune2 %>% 
  group_by(n.minobsinnode) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))


## 5.3 Tuning bag.fraction ----
GBM_tuning3 <- data.frame(matrix(ncol=6, nrow=0))
colnames(GBM_tuning3) = c('bag.fraction', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')
for (bf in seq(0.2, 1, 0.2)){
  for (i in 1:5){
    print(paste(bf, i))
    train <- resort_GBM[-outer_folds[[i]],] 
    train_idx <- createDataPartition(train$is_canceled, p=0.7, list=FALSE)
    train_inner <- train[train_idx,]
    val_inner <- train[-train_idx,]
    
    GBM <- gbm(is_canceled~., data=train_inner, distribution="bernoulli", n.trees=500, 
               shrinkage=0.2, n.minobsinnode=150, bag.fraction=bf, verbose=FALSE)
    # summary(GBM) #compute relative inference of each variable
    
    GBM.pred <- predict.gbm(GBM, val_inner, type='response', verbose=FALSE)
    GBM.pred[GBM.pred < 0.5] <- 0
    GBM.pred[GBM.pred >= 0.5] <- 1
    GBM.pred <- factor(GBM.pred, levels = c(1,0))
    
    val_inner$is_canceled <- factor(val_inner$is_canceled, levels = c(1,0))
    GBM_CM <- confusionMatrix(GBM.pred, val_inner$is_canceled)
    GBM_tuning3[nrow(GBM_tuning3)+1, ] = c(bf, i, 
                                         GBM_CM$overall['Accuracy'], GBM_CM$byClass['Precision'],
                                         GBM_CM$byClass['Recall'], GBM_CM$byClass['F1'])
  }
}
write.csv(GBM_tuning3, ".\\result\\GBM_resort_bagfrac.csv", row.names=FALSE)
GBM_tuning3 %>% 
  group_by(bag.fraction) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))


## 5.4 Tuning classify threshold -----
GBM_tuning4 <- data.frame(matrix(ncol=6, nrow=0))
colnames(GBM_tuning4) = c('threshold', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')
for (thresh in seq(0.3, 0.7, 0.1)){
  for (i in 1:5){
    print(paste(thresh, i))
    train <- resort_GBM[-outer_folds[[i]],] 
    train_idx <- createDataPartition(train$is_canceled, p=0.7, list=FALSE)
    train_inner <- train[train_idx,]
    val_inner <- train[-train_idx,]
    
    GBM <- gbm(is_canceled~., data=train_inner, distribution="bernoulli", n.trees=500, 
               shrinkage=0.2, n.minobsinnode=150, bag.fraction=1, verbose=FALSE)
    # summary(GBM) #compute relative inference of each variable
    
    GBM.pred <- predict.gbm(GBM, val_inner, type='response', verbose=FALSE)
    GBM.pred[GBM.pred < thresh] <- 0
    GBM.pred[GBM.pred >= thresh] <- 1
    GBM.pred <- factor(GBM.pred, levels = c(1,0))
    
    val_inner$is_canceled <- factor(val_inner$is_canceled, levels = c(1,0))
    GBM_CM <- confusionMatrix(GBM.pred, val_inner$is_canceled)
    GBM_tuning4[nrow(GBM_tuning4)+1, ] = c(thresh, i, 
                                           GBM_CM$overall['Accuracy'], GBM_CM$byClass['Precision'],
                                           GBM_CM$byClass['Recall'], GBM_CM$byClass['F1'])
  }
}
write.csv(GBM_tuning4, ".\\result\\GBM_resort_threshold.csv", row.names=FALSE)
GBM_tuning4 %>% 
  group_by(threshold) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))

## 5.5 Final GBM ----
GBM_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(GBM_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
for (i in 1:5){
  print(i)
  train <- resort_GBM[-outer_folds[[i]],] 
  test <- resort_GBM[outer_folds[[i]],] 
  
  GBM <- gbm(is_canceled~., data=train, distribution="bernoulli", n.trees=500, shrinkage=0.2, 
             n.minobsinnode=150, bag.fraction=1, verbose=FALSE)
  # summary(GBM) #compute relative inference of each variable
  
  GBM.pred <- predict.gbm(GBM, test, type='response', verbose=FALSE)
  GBM.pred[GBM.pred < 0.5] <- 0
  GBM.pred[GBM.pred >= 0.5] <- 1
  GBM.pred <- factor(GBM.pred, levels = c(1,0))
  
  test$is_canceled <- factor(test$is_canceled, levels = c(1,0))
  GBM_CM <- confusionMatrix(GBM.pred, test$is_canceled)
  GBM_result[nrow(GBM_result)+1, ] = c(GBM_CM$overall['Accuracy'], GBM_CM$byClass['Precision'],
                                       GBM_CM$byClass['Recall'], GBM_CM$byClass['F1'])
}
write.csv(GBM_result, ".\\result\\GBM_resort_Evaluation.csv", row.names=FALSE)
GBM_result
colMeans(GBM_result)


# 6. XGBoost ----
# Parameters Tuning
#   - eta = learning rate
#   - nrounds = max no. of boosting iterations
#   - gamma = for avoid overfitting 
#   - max_depth 
#   - min_child_weight
#   - subsample = subsample ratio for growing tree
#   - colsample_bytree = subsample ratio of columns

## XGB dataset ---- 
resort_xgb <- resort
resort_xgb$is_canceled <- unclass(resort_xgb$is_canceled)%%2
xgb_trainCtr = trainControl(method = "LGOCV", p = 0.7, number = 1, search = "grid")

## 6.1 Initial nrounds and learning rate ----
xgbGrid1 <-  expand.grid(eta = c(0.01, 0.1, 0.2, 0.3), 
                         nrounds = c(100, 300, 500),
                         # fixed values below
                         max_depth = 10, 
                         min_child_weight = 1,
                         gamma = 0.1,
                         subsample = 1,
                         colsample_bytree = 0.5
                         )

XGB_tune1 <- data.frame(matrix(ncol=3, nrow=0))
colnames(XGB_tune1) = c('fold', 'eta', 'nrounds')
for (i in 1:5){
  print(i)
  train_inner <- resort[-outer_folds[[i]],] 
  
  xgbModel <- train(is_canceled~.,
                   data = train_inner,
                   method = "xgbTree", 
                   trControl = xgb_trainCtr, 
                   tuneGrid = xgbGrid1,
                   verbosity = 0)
  
  XGB_tune1[nrow(XGB_tune1)+1, ] = c(i, xgbModel$bestTune$eta, xgbModel$bestTune$nrounds)
}
write.csv(XGB_tune1, ".\\result\\XGB_resort_etanrounds.csv", row.names=FALSE)
XGB_tune1


## 6.2 Tune max_depth, min_child_weight ----
xgbGrid2 <-  expand.grid(max_depth = c(5, 10, 15),
                         min_child_weight = c(1, 5, 10),
                         # fixed values below
                         eta = 0.1,
                         nrounds = 500,
                         gamma = 0.1,
                         subsample = 1,
                         colsample_bytree = 0.5
                         )

XGB_tune2<- data.frame(matrix(ncol=3, nrow=0))
colnames(XGB_tune2) = c('fold', 'max_depth', 'min_child_weight')
for (i in 1:5){
  print(i)
  train_inner <- resort[-outer_folds[[i]],] 
  
  xgbModel <- train(is_canceled~.,
                    data = train_inner,
                    method = "xgbTree", 
                    trControl = xgb_trainCtr, 
                    tuneGrid = xgbGrid2,
                    verbosity = 0)
  
  XGB_tune2[nrow(XGB_tune2)+1, ] = c(i, xgbModel$bestTune$max_depth, xgbModel$bestTune$min_child_weight)
}
write.csv(XGB_tune2, ".\\result\\XGB_resort_depth.csv", row.names=FALSE)
XGB_tune2

## 6.3 Tune gamma ----
xgbGrid3 <-  expand.grid(gamma = c(0, 0.01, 0.1, 0.2),
                         # fixed values below
                         eta = 0.1,
                         nrounds = 500,
                         max_depth = 15,
                         min_child_weight = 1,
                         subsample = 1,
                         colsample_bytree = 0.5
                         )
XGB_tune3 <- data.frame(matrix(ncol=2, nrow=0))
colnames(XGB_tune3) = c('fold', 'gamma')
for (i in 1:5){
  print(i)
  train_inner <- resort[-outer_folds[[i]],] 
  
  xgbModel <- train(is_canceled~.,
                    data = train_inner,
                    method = "xgbTree", 
                    trControl = xgb_trainCtr, 
                    tuneGrid = xgbGrid3,
                    verbosity = 0)
  
  XGB_tune3[nrow(XGB_tune3)+1, ] = c(i, xgbModel$bestTune$gamma)
}
write.csv(XGB_tune3, ".\\result\\XGB_resort_gamma.csv", row.names=FALSE)
XGB_tune3

## 6.4 Tune subsample and colsample_bytree ----
xgbGrid4 <-  expand.grid(colsample_bytree = c(0.7, 0.9, 1), 
                         subsample = c(0.7, 0.9, 1),
                         # fixed values below
                         gamma = 0.1,
                         eta = 0.1,
                         nrounds = 500,
                         max_depth = 15,
                         min_child_weight = 1
                         )
XGB_tune4 <- data.frame(matrix(ncol=3, nrow=0))
colnames(XGB_tune4) = c('fold', 'colsample_bytree', 'subsample')
for (i in 1:5){
  print(i)
  train_inner <- resort[-outer_folds[[i]],] 
  
  xgbModel <- train(is_canceled~.,
                    data = train_inner,
                    method = "xgbTree", 
                    trControl = xgb_trainCtr, 
                    tuneGrid = xgbGrid4,
                    verbosity = 0)
  
  XGB_tune4[nrow(XGB_tune4)+1, ] = c(i, xgbModel$bestTune$colsample_bytree, xgbModel$bestTune$subsample)
}
write.csv(XGB_tune4, ".\\result\\XGB_resort_sample.csv", row.names=FALSE)
XGB_tune4

## 6.5 Tune classification threshold ----
xgb_params <- list(eta = 0.1, 
                   gamma = 0.1, 
                   max_depth = 15, 
                   min_child_weight = 1, 
                   subsample = 0.9, 
                   colsample_bytree = 0.7,
                   booster = "gbtree", objective="binary:logistic", eval_metric="error")

xgb_threshold <- data.frame(matrix(ncol=6, nrow=0))
colnames(xgb_threshold) = c('Threshold', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')
for (threshold in seq(0.4, 0.6, 0.1)){
  for (i in 1:5){
    print(paste(threshold, i))
    train <- resort_xgb[-outer_folds[[i]],]
    train_idx <- createDataPartition(train$is_canceled, p=0.7, list=FALSE)
    train_inner <- train[train_idx,]
    val_inner <- train[-train_idx,]
  
    dtrain = xgb.DMatrix(as.matrix(sapply(train_inner %>% select(-is_canceled), as.numeric)),
                         label=train_inner$is_canceled)
    dval = xgb.DMatrix(as.matrix(sapply(val_inner %>% select(-is_canceled), as.numeric)),
                        label=val_inner$is_canceled)
    
    # nrounds on this below line
    XGB <- xgboost(params = xgb_params, data = dtrain, nrounds = 500, verbose=0)
    xgb.pred <- predict(XGB, newdata=dval)
    xgb.pred <- ifelse (xgb.pred >= threshold, 1, 0)
    xgb.pred <- factor(xgb.pred, levels = c(1,0))
    
    val_inner$is_canceled <- factor(val_inner$is_canceled, levels = c(1,0))
    xgb.cm <- confusionMatrix(xgb.pred, val_inner$is_canceled)
    xgb_threshold[nrow(xgb_threshold)+1, ] = c(threshold, i, 
                                               xgb.cm$overall['Accuracy'], xgb.cm$byClass['Precision'],
                                               xgb.cm$byClass['Recall'], xgb.cm$byClass['F1'])
  }
}
xgb_threshold
xgb_threshold %>% 
  group_by(Threshold) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))
write.csv(xgb_threshold, ".\\result\\XGB_resort_threshold.csv", row.names=FALSE)

## 6.6 Final result ----
xgb_params <- list(eta = 0.1, 
                   gamma = 0.1, 
                   max_depth = 15, 
                   min_child_weight = 1, 
                   subsample = 0.9, 
                   colsample_bytree = 0.7,
                   booster = "gbtree", objective="binary:logistic", eval_metric="error")
XGB_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(XGB_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
XGB_imp <- matrix(0, nrow = ncol(resort_xgb)-1, ncol=0)
XGB_imp <- cbind(XGB_imp, colnames(resort_xgb %>% select(-is_canceled)))
colnames(XGB_imp) <- c('Feature')

for (i in 1:5){
  print(i)
  train <- resort_xgb[-outer_folds[[i]],]
  test <- resort_xgb[outer_folds[[i]],]
  
  dtrain = xgb.DMatrix(as.matrix(sapply(train %>% select(-is_canceled), as.numeric)),
                       label=train$is_canceled)
  dtest = xgb.DMatrix(as.matrix(sapply(test %>% select(-is_canceled), as.numeric)), 
                      label=test$is_canceled)
  
  # nrounds on this below line
  XGB <- xgboost(params = xgb_params, data = dtrain, nrounds = 500, verbose=0)
  xgb_gain <- xgb.importance(model=XGB)[, 1:2]
  colnames(xgb_gain)[2] <- paste0('Gain.', as.character(i))
  XGB_imp <- merge(x = XGB_imp,
                   y = xgb_gain,
                   by.x = "Feature",
                   by.y = "Feature",
                   all.x = TRUE
                   )
  
  xgb.pred <- predict(XGB, newdata=dtest)
  xgb.pred <- ifelse (xgb.pred >= 0.5, 1, 0)
  xgb.pred <- factor(xgb.pred, levels = c(1,0))
  
  test$is_canceled <- factor(test$is_canceled, levels = c(1,0))
  xgb.cm <- confusionMatrix(xgb.pred, test$is_canceled)
  XGB_result[nrow(XGB_result)+1, ] = c(xgb.cm$overall['Accuracy'], xgb.cm$byClass['Precision'],
                                       xgb.cm$byClass['Recall'], xgb.cm$byClass['F1'])
}
XGB_result
colMeans(XGB_result)
write.csv(XGB_result, ".\\result\\XGB_resort_Evaluation.csv", row.names=FALSE)

# Feature importance
XGB_imp[is.na(XGB_imp)] <- 0
XGB_imp$MeansGain <- rowMeans(XGB_imp[2:6])
XGB_imp20 <- top_n(XGB_imp, 20, MeansGain)
ggplot(XGB_imp20, aes(y=reorder(Feature, MeansGain), x=MeansGain)) +
  geom_bar(stat = "identity") +
  xlab('Gain') +
  ylab('Features') +
  ggtitle('Resort Hotel using XGBoost')
write.csv(XGB_imp, ".\\result\\XGB_resort_FI.csv", row.names=FALSE)


# 7. LightGBM ----

# Fixed parameters
#   - objective = 'binary'
#   - data_sample_strategy = 'goss'

# 1. Boosting: gbdt
#   - boosting = 'gbdt'
#   - num_iterations
#   - learning_rate
#   - num_leaves
#   - min_data_in_leaf
#   - feature_fraction
#   - top_rate
#   - other_rate
#   - lambda_l1 
#   - lambda_l2

# 2. Boosting: dart
#   - boosting = 'dart'
#   - num_iterations
#   - learning_rate
#   - num_leaves
#   - min_data_in_leaf
#   - feature_fraction
#   - top_rate
#   - other_rate
#   - lambda_l1 
#   - lambda_l2
#   - xgboost_dart_mode = true
#   - drop_rate
#   - max_drop
#   - skip_drop
#   - uniform_drop (boolean)

# params_gbdt <- list(objective='binary', data_sample_strategy='goss', boosting='gbdt', 
#                 num_iterations=100, learning_rate=0.1, num_leaves=31, 
#                 min_data_in_leaf=20, 
#                 feature_fraction=1,
#                 top_rate=0.2, other_rate=0.1,
#                 lambda_l1=0, lambda_l2=0)

## LGB dataset ----
resort_lgbm <- resort
resort_lgbm$is_canceled <- unclass(resort_lgbm$is_canceled)%%2

## 7.1 Boosting: gbdt ----
### 7.1.1 Initial tree ----
# num_iterations, learning_rate, num_leaves
lgbmGrid1 <- expand.grid(num_iterations = c(500),
                         learning_rate = c(0.01, 0.1),
                         num_leaves = c(100, 250))
lgbm_tune1 <- data.frame(matrix(ncol=5, nrow=0))
colnames(lgbm_tune1) = c('fold', 'num_iterations', 'learning_rate', 'num_leaves', 'binary_logloss')

for (i in 1:5){
  print(i)
  train <- resort_lgbm[-outer_folds[[i]],] 
  train_idx <- createDataPartition(train$is_canceled, p=0.7, list=FALSE)
  train_inner <- train[train_idx,]
  val_inner <- train[-train_idx,]
  
  dtrain <- lgb.Dataset(as.matrix(sapply(train_inner %>% select(-is_canceled), as.numeric)),
                        label=train_inner$is_canceled)
  dval <- lgb.Dataset.create.valid(dataset = dtrain, 
                                   data = as.matrix(sapply(val_inner %>% select(-is_canceled), as.numeric)), 
                                   label = val_inner$is_canceled)
  valids <- list(test = dval)
  
  model <- list()
  loss <- numeric(nrow(lgbmGrid1))
  for (j in 1:nrow(lgbmGrid1)){
    model[[j]] <- lgb.train(params = list(num_iterations = lgbmGrid1[j, 'num_iterations'],
                                          learning_rate = lgbmGrid1[j, 'learning_rate'],
                                          num_leaves = lgbmGrid1[j, 'num_leaves'],
                                          # fixed value below
                                          objective = 'binary', 
                                          data_sample_strategy = 'goss',
                                          boosting = 'gbdt', 
                                          min_data_in_leaf = 20, 
                                          feature_fraction = 1,
                                          top_rate = 0.2, 
                                          other_rate = 0.1,
                                          lambda_l1 = 0, 
                                          lambda_l2 = 0
                                          ),
                            data = dtrain,
                            valids = valids,
                            verbose = 0
                            )
    loss[j] <- min(rbindlist(model[[j]]$record_evals$test$binary_logloss))
  }
  lgbm_tune1[nrow(lgbm_tune1)+1, ] = c(i,
                                       lgbmGrid1[which.min(loss), "num_iterations"],
                                       lgbmGrid1[which.min(loss), "learning_rate"],
                                       lgbmGrid1[which.min(loss), "num_leaves"],
                                       min(loss))
  print(lgbm_tune1[i,])  
}
lgbm_tune1
write.csv(lgbm_tune1, ".\\result\\LGBM_resort_etantree.csv", row.names=FALSE)

### 7.1.2 Tuning tree parameters ----
# min_data_in_leaf
lgbmGrid2 <- expand.grid(min_data_in_leaf = c(1, 10))
lgbm_tune2 <- data.frame(matrix(ncol=3, nrow=0))
colnames(lgbm_tune2) = c('fold', 'min_data_in_leaf', 'binary_logloss')

for (i in 1:5){
  print(i)
  train <- resort_lgbm[-outer_folds[[i]],] 
  train_idx <- createDataPartition(train$is_canceled, p=0.7, list=FALSE)
  train_inner <- train[train_idx,]
  val_inner <- train[-train_idx,]
  
  dtrain <- lgb.Dataset(as.matrix(sapply(train_inner %>% select(-is_canceled), as.numeric)),
                        label=train_inner$is_canceled)
  dval <- lgb.Dataset.create.valid(dataset = dtrain, 
                                   data = as.matrix(sapply(val_inner %>% select(-is_canceled), as.numeric)), 
                                   label = val_inner$is_canceled)
  valids <- list(test = dval)
  
  model <- list()
  loss <- numeric(nrow(lgbmGrid2))
  for (j in 1:nrow(lgbmGrid2)){
    model[[j]] <- lgb.train(params = list(min_data_in_leaf = lgbmGrid2[j, 'min_data_in_leaf'], 
                                          # fixed value below
                                          num_iterations = 500,
                                          learning_rate = 0.01,
                                          num_leaves = 250,
                                          objective = 'binary', 
                                          data_sample_strategy = 'goss',
                                          boosting = 'gbdt',
                                          feature_fraction = 1,
                                          top_rate = 0.2, 
                                          other_rate = 0.1,
                                          lambda_l1 = 0, 
                                          lambda_l2 = 0),
                            data = dtrain,
                            valids = valids,
                            verbose = 0)
    loss[j] <- min(rbindlist(model[[j]]$record_evals$test$binary_logloss))
  }
  lgbm_tune2[nrow(lgbm_tune2)+1, ] = c(i,
                                       lgbmGrid2[which.min(loss), "min_data_in_leaf"],
                                       min(loss))
  print(lgbm_tune2[i,])  
}
lgbm_tune2
write.csv(lgbm_tune2, ".\\result\\LGBM_resort_mindataleaf.csv", row.names=FALSE)

### 7.1.3 Tuning feature_fraction ----
# feature_fraction
lgbmGrid3 <- expand.grid(feature_fraction = c(1, 10))
lgbm_tune3 <- data.frame(matrix(ncol=3, nrow=0))
colnames(lgbm_tune3) = c('fold', 'feature_fraction', 'binary_logloss')

for (i in 1:5){
  print(i)
  train <- resort_lgbm[-outer_folds[[i]],] 
  train_idx <- createDataPartition(train$is_canceled, p=0.7, list=FALSE)
  train_inner <- train[train_idx,]
  val_inner <- train[-train_idx,]
  
  dtrain <- lgb.Dataset(as.matrix(sapply(train_inner %>% select(-is_canceled), as.numeric)),
                        label=train_inner$is_canceled)
  dval <- lgb.Dataset.create.valid(dataset = dtrain, 
                                   data = as.matrix(sapply(val_inner %>% select(-is_canceled), as.numeric)), 
                                   label = val_inner$is_canceled)
  valids <- list(test = dval)
  
  model <- list()
  loss <- numeric(nrow(lgbmGrid3))
  for (j in 1:nrow(lgbmGrid3)){
    model[[j]] <- lgb.train(params = list(feature_fraction = lgbmGrid3[j, 'feature_fraction'],
                                          # fixed value below
                                          num_iterations = 500,
                                          learning_rate = 0.01,
                                          num_leaves = 250,
                                          objective = 'binary', 
                                          data_sample_strategy = 'goss',
                                          boosting = 'gbdt',
                                          min_data_in_leaf = , 
                                          top_rate = 0.2, 
                                          other_rate = 0.1,
                                          lambda_l1 = 0, 
                                          lambda_l2 = 0),
                            data = dtrain,
                            valids = valids,
                            verbose = 0)
    loss[j] <- min(rbindlist(model[[j]]$record_evals$test$binary_logloss))
  }
  lgbm_tune3[nrow(lgbm_tune3)+1, ] = c(i,
                                       lgbmGrid3[which.min(loss), "feature_fraction"],
                                       min(loss))
  print(lgbm_tune3[i,])  
}
lgbm_tune3
write.csv(lgbm_tune3, ".\\result\\LGBM_resort_featfrac.csv", row.names=FALSE)

### 7.1.4 Tuning goss rates ----
# top_rate, other_rate 
lgbmGrid4 <- expand.grid(top_rate = c(),
                         other_rate = c())
lgbm_tune4 <- data.frame(matrix(ncol=4, nrow=0))
colnames(lgbm_tune4) = c('fold', 'top_rate', 'other_rate', 'binary_logloss')

for (i in 1:5){
  print(i)
  train <- resort_lgbm[-outer_folds[[i]],] 
  train_idx <- createDataPartition(train$is_canceled, p=0.7, list=FALSE)
  train_inner <- train[train_idx,]
  val_inner <- train[-train_idx,]
  
  dtrain <- lgb.Dataset(as.matrix(sapply(train_inner %>% select(-is_canceled), as.numeric)),
                        label=train_inner$is_canceled)
  dval <- lgb.Dataset.create.valid(dataset = dtrain, 
                                   data = as.matrix(sapply(val_inner %>% select(-is_canceled), as.numeric)), 
                                   label = val_inner$is_canceled)
  valids <- list(test = dval)
  
  model <- list()
  loss <- numeric(nrow(lgbmGrid4))
  for (j in 1:nrow(lgbmGrid4)){
    model[[j]] <- lgb.train(params = list(top_rate = lgbmGrid4[which.min(loss), "top_rate"], 
                                          other_rate = lgbmGrid4[which.min(loss), "other_rate"],
                                          # fixed value below
                                          num_iterations = 500,
                                          learning_rate = 0.01,
                                          num_leaves = 250,
                                          objective = 'binary', 
                                          data_sample_strategy = 'goss',
                                          boosting = 'gbdt',
                                          min_data_in_leaf = , 
                                          feature_fraction = ,
                                          lambda_l1 = 0, 
                                          lambda_l2 = 0),
                            data = dtrain,
                            valids = valids,
                            verbose = 0)
    loss[j] <- min(rbindlist(model[[j]]$record_evals$test$binary_logloss))
  }
  lgbm_tune4[nrow(lgbm_tune4)+1, ] = c(i,
                                       lgbmGrid4[which.min(loss), "top_rate"],
                                       lgbmGrid4[which.min(loss), "other_rate"],
                                       min(loss))
  print(lgbm_tune4[i,])  
}
lgbm_tune4
write.csv(lgbm_tune4, ".\\result\\LGBM_resort_gossrate.csv", row.names=FALSE)

## 7.2 Tuning classification threshold ----
lgbm_params <- list(objective = 'binary', data_sample_strategy = 'goss', boosting = 'gbdt',
                    num_iterations = 500, learning_rate = 0.1, num_leaves = 250,
                    min_data_in_leaf = 1, feature_fraction = 0.8,
                    top_rate = 0.2, other_rate = 0.5,
                    lambda_l1 = 0, lambda_l2 = 0)

lgbm_threshold <- data.frame(matrix(ncol=6, nrow=0))
colnames(lgbm_threshold) = c('Threshold', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')

for (threshold in seq(0.4, 0.6, 0.1)){
  for (i in 1:5){
    print(paste(threshold, i))
    train <- resort_lgbm[-outer_folds[[i]],]
    train_idx <- createDataPartition(train$is_canceled, p=0.7, list=FALSE)
    train_inner <- train[train_idx,]
    val_inner <- train[-train_idx,]
    
    dtrain <- lgb.Dataset(as.matrix(sapply(train_inner %>% select(-is_canceled), as.numeric)),
                          label=train_inner$is_canceled)
    dval <- as.matrix(sapply(val_inner %>% select(-is_canceled), as.numeric))
    
    LGBM <- lgb.train(params = lgbm_params,
                      data = dtrain,
                      verbose = 0)
    
    LGBM.pred <- predict(LGBM, newdata=dval)
    LGBM.pred <- ifelse (LGBM.pred >= threshold, 1, 0)
    LGBM.pred <- factor(LGBM.pred, levels = c(1,0))
    
    val_inner$is_canceled <- factor(val_inner$is_canceled, levels = c(1,0))
    LGBM.cm <- confusionMatrix(LGBM.pred, val_inner$is_canceled)
    lgbm_threshold[nrow(lgbm_threshold)+1, ] = c(threshold, i,
                                                 LGBM.cm$overall['Accuracy'], LGBM.cm$byClass['Precision'],
                                                 LGBM.cm$byClass['Recall'], LGBM.cm$byClass['F1'])
  }
}
lgbm_threshold %>% 
  group_by(Threshold) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))
write.csv(lgbm_threshold, ".\\result\\LGBM_resort_threshold.csv", row.names=FALSE)

## 7.3 Final LGBM with gbdt boosting ----
lgbm_params <- list(objective = 'binary', data_sample_strategy = 'goss', boosting = 'gbdt',
                    num_iterations = 500, learning_rate = 0.1, num_leaves = 250,
                    min_data_in_leaf = 1, feature_fraction = 0.8,
                    top_rate = 0.2, other_rate = 0.5,
                    lambda_l1 = 0, lambda_l2 = 0)

lgbm_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(lgbm_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
LGB_imp <- matrix(0, nrow = ncol(resort_lgbm)-1, ncol=0)
LGB_imp <- cbind(LGB_imp, colnames(resort_lgbm %>% select(-is_canceled)))
colnames(LGB_imp) <- c('Feature')

for (i in 1:5){
  print(i)
  train <- resort_lgbm[-outer_folds[[i]],]
  test <- resort_lgbm[outer_folds[[i]],]
  
  dtrain <- lgb.Dataset(as.matrix(sapply(train %>% select(-is_canceled), as.numeric)),
                        label=train$is_canceled)
  dtest <- as.matrix(sapply(test %>% select(-is_canceled), as.numeric))
  
  LGBM <- lgb.train(params = lgbm_params,
                    data = dtrain,
                    verbose = 0)
  lgb_gain <- lgb.importance(model=LGBM, percentage=TRUE)[, 1:2]
  colnames(lgb_gain)[2] <- paste0('Gain.', as.character(i))
  LGB_imp <- merge(x = LGB_imp,
                   y = lgb_gain,
                   by.x = "Feature",
                   by.y = "Feature",
                   all.x = TRUE
                   )
  LGBM.pred <- predict(LGBM, newdata=dtest)
  LGBM.pred <- ifelse (LGBM.pred >= 0.6, 1, 0)
  LGBM.pred <- factor(LGBM.pred, levels = c(1,0))
  
  test$is_canceled <- factor(test$is_canceled, levels = c(1,0))
  LGBM.cm <- confusionMatrix(LGBM.pred, test$is_canceled)
  lgbm_result[nrow(lgbm_result)+1, ] = c(LGBM.cm$overall['Accuracy'], LGBM.cm$byClass['Precision'],
                                         LGBM.cm$byClass['Recall'], LGBM.cm$byClass['F1'])
}
lgbm_result
colMeans(lgbm_result)
write.csv(lgbm_result, ".\\result\\LGBM_resort_Evaluation.csv", row.names=FALSE)

# Feature importance
LGB_imp[is.na(LGB_imp)] <- 0
LGB_imp$MeansGain <- rowMeans(LGB_imp[2:6])
LGB_imp20 <- top_n(LGB_imp, 20, MeansGain)
ggplot(LGB_imp20, aes(y=reorder(Feature, MeansGain), x=MeansGain)) +
  geom_bar(stat = "identity") +
  xlab('Gain') +
  ylab('Features') +
  ggtitle('Resort Hotel using LightGBM')
write.csv(LGB_imp, ".\\result\\LGBM_resort_FI.csv", row.names=FALSE)

# 8. CatBoost ----
# Tuning parameters
# - depth = Tree Depth 
# - learning_rate
# - iterations = Number of trees
# - l2_leaf_reg = L2 regularization coefficient
# - rsm = The percentage of features to use at each iteration
# - border_count = The number of splits for numerical features

## catBoost dataset ----
hotel_catb <- preprocess(one_hot = FALSE, feature_select = TRUE)
resort_catb <- hotel_catb[[2]]
sapply(resort_catb, class)
summary(resort_catb$is_canceled)
catb_trainCR <- trainControl(method = 'LGOCV',  p = 0.7, number = 1, search = 'grid')

## 8.1 Initial trees ----
# Tune iterations and learning_rate
catbGrid1 <- expand.grid(iterations = c(100, 300, 500),
                         learning_rate = c(0.01, 0.1, 0.3),
                         # fixed value
                         depth = 8,
                         l2_leaf_reg = 0.01,
                         rsm = 1,
                         border_count = 512)
catB_tune1 <- data.frame(matrix(ncol = 3, nrow = 0))
colnames(catB_tune1) = c('fold', 'iterations', 'learning_rate')
for (i in 1:5){
  print(i)
  train_inner <- resort_catb[-outer_folds[[i]],]
  
  catBModel <- train(train_inner %>% select(-is_canceled), 
                     as.factor(make.names(train_inner$is_canceled)),
                     method = catboost.caret,
                     tuneGrid = catbGrid1, 
                     trControl = catb_trainCR,
                     verbose = 0)
  catB_tune1[nrow(catB_tune1)+1, ] = c(i, catBModel$bestTune$iterations, catBModel$bestTune$learning_rate)
}
write.csv(catB_tune1, ".\\result\\catB_resort_iterlr.csv", row.names=FALSE)
catB_tune1

## 8.2 Tune tree depth ----
catbGrid2 <- expand.grid(depth = c(4, 6, 8, 10, 12),
                         # fixed value
                         iterations = 500,
                         learning_rate = 0.1,
                         l2_leaf_reg = 0.01,
                         rsm = 1,
                         border_count = 512)
catB_tune2 <- data.frame(matrix(ncol = 2, nrow = 0))
colnames(catB_tune2) = c('fold', 'depth')
for (i in 1:5){
  print(i)
  train_inner <- resort_catb[-outer_folds[[i]],]
  
  catBModel <- train(train_inner %>% select(-is_canceled), 
                     as.factor(make.names(train_inner$is_canceled)),
                     method = catboost.caret,
                     tuneGrid = catbGrid2, 
                     trControl = catb_trainCR,
                     verbose = 0)
  catB_tune2[nrow(catB_tune2)+1, ] = c(i, catBModel$bestTune$depth)
}
write.csv(catB_tune2, ".\\result\\catB_resort_depth.csv", row.names=FALSE)
catB_tune2

## 8.3 Tune rsm ----
catbGrid3 <- expand.grid(rsm = c(0.4, 0.6, 0.8, 1),
                         # fixed value
                         iterations = 500,
                         learning_rate = 0.1,
                         depth = 10,
                         l2_leaf_reg = 0.01,
                         border_count = 512)
catB_tune3 <- data.frame(matrix(ncol = 2, nrow = 0))
colnames(catB_tune3) = c('fold', 'rsm')
for (i in 1:5){
  print(i)
  train_inner <- resort_catb[-outer_folds[[i]],]
  
  catBModel <- train(train_inner %>% select(-is_canceled), 
                     as.factor(make.names(train_inner$is_canceled)),
                     method = catboost.caret,
                     tuneGrid = catbGrid3, 
                     trControl = catb_trainCR,
                     verbose = 0)
  catB_tune3[nrow(catB_tune3)+1, ] = c(i, catBModel$bestTune$rsm)
}
write.csv(catB_tune3, ".\\result\\catB_resort_rsm.csv", row.names=FALSE)
catB_tune3

## 8.4 Tune L2 reg ----
catbGrid4 <- expand.grid(l2_leaf_reg = c(0, 0.001, 0.01, 0.1),
                         # fixed value
                         iterations = 500,
                         learning_rate = 0.1,
                         depth = ,
                         rsm = ,
                         border_count = 512)
catB_tune4 <- data.frame(matrix(ncol = 2, nrow = 0))
colnames(catB_tune4) = c('fold', 'l2_leaf_reg')
for (i in 1:5){
  print(i)
  train_inner <- resort_catb[-outer_folds[[i]],]
  
  catBModel <- train(train_inner %>% select(-is_canceled), 
                     as.factor(make.names(train_inner$is_canceled)),
                     method = catboost.caret,
                     tuneGrid = catbGrid4, 
                     trControl = catb_trainCR,
                     verbose = 0)
  catB_tune4[nrow(catB_tune4)+1, ] = c(i, catBModel$bestTune$l2_leaf_reg)
}
write.csv(catB_tune4, ".\\result\\catB_resort_l2.csv", row.names=FALSE)
catB_tune4

## 8.5 Tune classification threshold ----
catb_params <- list(iterations = 500, 
                    learning_rate = 0.1,
                    rsm = ,
                    depth = ,
                    l2_leaf_reg = ,
                    border_count = 512,
                    logging_level = 'Silent')
catb_threshold <- data.frame(matrix(ncol=6, nrow=0))
colnames(catb_threshold) = c('Threshold', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')
for (threshold in seq(0.4, 0.6, 0.1)){
  for (i in 1:5){
    print(paste(threshold, i))
    train <- resort_catb[-outer_folds[[i]],]
    train_idx <- createDataPartition(train$is_canceled, p=0.7, list=FALSE)
    train_inner <- train[train_idx,]
    val_inner <- train[-train_idx,]
    
    train_pool <- catboost.load_pool(data = train_inner %>% select(-is_canceled), 
                                     label = unclass(train_inner$is_canceled)%%2)
    val_pool <- catboost.load_pool(data = val_inner %>% select(-is_canceled), 
                                    label = unclass(val_inner$is_canceled)%%2)
  
    catb_model <- catboost.train(train_pool, params = catb_params)
    
    catb.pred <- catboost.predict(catb_model, val_pool, prediction_type = 'Probability')
    catb.pred <- ifelse (catb.pred >= threshold, 1, 0)
    catb.pred <- factor(catb.pred, levels = c(1,0))
    
    val_inner$is_canceled <- factor(val_inner$is_canceled, levels = c(1,0))
    catb.cm <- confusionMatrix(catb.pred, val_inner$is_canceled)
    catb_threshold[nrow(catb_threshold)+1, ] = c(threshold, i, 
                                                 catb.cm$overall['Accuracy'], catb.cm$byClass['Precision'],
                                                 catb.cm$byClass['Recall'], catb.cm$byClass['F1'])
  }
}
catb_threshold %>% 
  group_by(Threshold) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))
write.csv(catb_threshold, ".\\result\\catB_resort_threshold.csv", row.names=FALSE)

## 8.6 Final CatBoost ----
catb_params <- list(iterations = 500, 
                    learning_rate = 0.1,
                    rsm = 1,
                    depth = 10,
                    l2_leaf_reg = 0.01,
                    border_count = 512,
                    logging_level = 'Silent')

catb_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(catb_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
catb_imp <- matrix(0, nrow = ncol(resort_catb)-1, ncol=0)
catb_imp <- cbind(catb_imp, colnames(resort_catb %>% select(-is_canceled)))
colnames(catb_imp) <- c('Feature')

for (i in 1:5){
  print(i)
  train <- resort_catb[-outer_folds[[i]],]
  test <- resort_catb[outer_folds[[i]],]
  
  train_pool <- catboost.load_pool(data = train %>% select(-is_canceled), 
                                   label = unclass(train$is_canceled)%%2)
  test_pool <- catboost.load_pool(data = test %>% select(-is_canceled), 
                                  label = unclass(test$is_canceled)%%2)
  
  catb_model <- catboost.train(train_pool, params = catb_params)
  catb_fi <- catboost.get_feature_importance(catb_model)
  catb_fi <- data.frame(Feature = row.names(catb_fi), catb_fi)
  colnames(catb_fi)[2] <- paste0('FI.', as.character(i))
  catb_imp <- merge(x = catb_imp,
                    y = catb_fi,
                    by.x = "Feature",
                    by.y = "Feature",
                    all.x = TRUE)
  
  catb.pred <- catboost.predict(catb_model, test_pool,prediction_type = 'Probability')
  catb.pred <- ifelse (catb.pred >= 0.6, 1, 0)
  catb.pred <- factor(catb.pred, levels = c(1,0))
  
  test$is_canceled <- factor(test$is_canceled, levels = c(1,0))
  catb.cm <- confusionMatrix(catb.pred, test$is_canceled)
  catb_result[nrow(catb_result)+1, ] = c(catb.cm$overall['Accuracy'], catb.cm$byClass['Precision'],
                                         catb.cm$byClass['Recall'], catb.cm$byClass['F1'])
}
colMeans(catb_result)
catb_result
write.csv(catb_result, ".\\result\\catB_resort_Evaluation.csv", row.names=FALSE)

# Feature importance
catb_imp$MeansFI <- rowMeans(catb_imp[2:5])
ggplot(catb_imp, aes(y=reorder(Feature, MeansFI), x=MeansFI)) +
  geom_bar(stat = "identity") +
  xlab('Importance') +
  ylab('Features') +
  ggtitle('Resort Hotel using CatBoost')
write.csv(catb_imp, ".\\result\\catB_resort_FI.csv", row.names=FALSE)

