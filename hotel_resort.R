source('preprocess.R')
library(tree)
library(tidyverse) 
library(randomForest)
library(gbm)


# 1. Import data ----
hotel_data <- preprocess()
resort <- hotel_data[[2]]
sapply(resort, class)
summary(resort$is_canceled)


# 2. Split cross-validation ----
set.seed(1)
folds <- createFolds(resort$is_canceled, k=10)


# 3 Decision Tree ----
DT_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(DT_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
for (i in 1:10){
  train <- resort[-folds[[i]],] 
  test <- resort[folds[[i]],] 
  
  trees <- tree(is_canceled~., data=train, split=c("deviance", "gini"))
  # summary(trees)
  # plot(trees)
  # text(trees, pretty=0)

  cv.trees <- cv.tree(trees, FUN=prune.misclass, K=10)
  prune.trees <- prune.misclass(trees, best=cv.trees$size[which.min(cv.trees$dev)])
  # plot(cv.trees)
  # plot(prune.trees)
  # text(prune.trees, pretty=0)
  # summary(prune.trees)$used

  tree.pred <- predict(prune.trees, test, type='class')
  DT_CM <- confusionMatrix(tree.pred, test$is_canceled)
  DT_result[nrow(DT_result)+1, ] = c(DT_CM$overall['Accuracy'], DT_CM$byClass['Precision'],
                                     DT_CM$byClass['Recall'], DT_CM$byClass['F1'])
}
clcolMeans(DT_Result)
summary(prune.trees)$used


# 4 Random forest ----
# Parameters Tuning
#   - ntree = no.of trees (row samples size = nrow(train))
#   - mtry = no. features that samples
#   - nodesize  = min size of terminal nodes

## 4.1 Tune ntree (manual) ----
for (n in seq(100, 500, 100)){
  RF_result <- data.frame(matrix(ncol=4, nrow=0))
  colnames(RF_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
  for (i in 1:10){
    train <- resort[-folds[[i]],] 
    test <- resort[folds[[i]],] 
  
    RF <- randomForest(is_canceled~., data=train, ntree=n, mtry=sqrt(33), nodesize=1)
    
    RF.pred <- predict(RF, test, type='response')
    RF_CM <- confusionMatrix(RF.pred, test$is_canceled)
    RF_result[nrow(RF_result)+1, ] = c(RF_CM$overall['Accuracy'], RF_CM$byClass['Precision'],
                                       RF_CM$byClass['Recall'], RF_CM$byClass['F1'])
  }
  print(n)
  print(colMeans(RF_result))
}

## 4.2 Tune mtry with tuneRF function ----
RF_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(RF_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
for (i in 1:10){
  train <- resort[-folds[[i]],] 
  test <- resort[folds[[i]],] 
  
  tune_mtry <- tuneRF(train[,-1], train[,1], stepFactor = 2, ntreeTry=50, plot=TRUE, trace=FALSE)
  print(tune_mtry)
}

## 4.3 Tune nodesize (manual) ----
for (n in seq(1, 3, 1)){
  RF_result <- data.frame(matrix(ncol=4, nrow=0))
  colnames(RF_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
  for (i in 1:10){
    train <- resort[-folds[[i]],] 
    test <- resort[folds[[i]],] 
    
    RF <- randomForest(is_canceled~., data=train, ntree=100, mtry=10, nodesize=n)
    
    RF.pred <- predict(RF, test, type='response')
    RF_CM <- confusionMatrix(RF.pred, test$is_canceled)
    RF_result[nrow(RF_result)+1, ] = c(RF_CM$overall['Accuracy'], RF_CM$byClass['Precision'],
                                       RF_CM$byClass['Recall'], RF_CM$byClass['F1'])
  }
  print(n)
  print(colMeans(RF_result))
}

## 4.4 Final result ----
RF_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(RF_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
for (i in 1:10){
  train <- resort[-folds[[i]],] 
  test <- resort[folds[[i]],] 
  
  RF <- randomForest(is_canceled~., data=train, ntree=100, mtry=10, nodesize=1)
  
  RF.pred <- predict(RF, test, type='response')
  RF_CM <- confusionMatrix(RF.pred, test$is_canceled)
  RF_result[nrow(RF_result)+1, ] = c(RF_CM$overall['Accuracy'], RF_CM$byClass['Precision'],
                                     RF_CM$byClass['Recall'], RF_CM$byClass['F1'])
}

colMeans(RF_result)
importance(RF)


# 5. Gradient Boosting ----
# Parameters Tuning
#   - classification threshold 
#   - n.trees = total no. of tree
#   - shrinkage = learning rate
#   - n.minobsinnode = min no. of observations in terminal node
#   - bag.fraction = fraction of training set to be selected to build next tree

resort_GBM <- resort
resort_GBM$is_canceled <- unclass(resort_GBM$is_canceled)%%2 #Change variable type

## 5.1 Tuning classify threshold -----
GBM_tuning0 <- data.frame(matrix(ncol=5, nrow=0))
colnames(GBM_tuning0) = c('threshold', 'Accuracy', 'Precision', 'Recall', 'F1')
for (j in seq(0.35, 0.45, 0.01)){
  GBM_result <- data.frame(matrix(ncol=4, nrow=0))
  colnames(GBM_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
  for (i in 1:10){
    train <- resort_GBM[-folds[[i]],] 
    test <- resort_GBM[folds[[i]],] 
    
    GBM <- gbm(is_canceled~., data=train, distribution="bernoulli", n.trees=500, n.minobsinnode=80, 
               shrinkage=0.02, bag.fraction=0.4, verbose=FALSE)
    # summary(GBM) #compute relative inference of each variable
  
    GBM.pred <- predict.gbm(GBM, test, type='response', verbose=FALSE)
    GBM.pred[GBM.pred < j] <- 0
    GBM.pred[GBM.pred >= j] <- 1
    GBM.pred <- factor(GBM.pred, levels = c(1,0))
    
    test$is_canceled <- factor(test$is_canceled, levels = c(1,0))
    GBM_CM <- confusionMatrix(GBM.pred, test$is_canceled)
    GBM_result[nrow(GBM_result)+1, ] = c(GBM_CM$overall['Accuracy'], GBM_CM$byClass['Precision'],
                                         GBM_CM$byClass['Recall'], GBM_CM$byClass['F1'])
  }
  # print(j)
  # print(colMeans(GBM_result))
  GBM_tuning0[nrow(GBM_tuning0)+1, ] = c(j, colMeans(GBM_result)[1], colMeans(GBM_result)[2],
                                       colMeans(GBM_result)[3], colMeans(GBM_result)[4])
}
GBM_tuning0

## 5.2 Tuning n.trees and shrinkage -----
GBM_tuning <- data.frame(matrix(ncol=6, nrow=0))
colnames(GBM_tuning) = c('n.trees', 'shrinkage', 'Accuracy', 'Precision', 'Recall', 'F1')
for (n in seq(500, 1000, 100)){
  for (rl in seq(0.01, 0.03, 0.01)){
    GBM_result <- data.frame(matrix(ncol=4, nrow=0))
    colnames(GBM_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
    for (i in 1:10){
      train <- resort_GBM[-folds[[i]],] 
      test <- resort_GBM[folds[[i]],] 
      
      GBM <- gbm(is_canceled~., data=train, distribution="bernoulli", n.trees=n, n.minobsinnode=10, 
                 shrinkage=rl, bag.fraction=0.5, verbose=FALSE)
      # summary(GBM) #compute relative inference of each variable
      
      GBM.pred <- predict.gbm(GBM, test, type='response', verbose=FALSE)
      GBM.pred[GBM.pred < 0.35] <- 0
      GBM.pred[GBM.pred >= 0.35] <- 1
      GBM.pred <- factor(GBM.pred, levels = c(1,0))
      
      test$is_canceled <- factor(test$is_canceled, levels = c(1,0))
      GBM_CM <- confusionMatrix(GBM.pred, test$is_canceled)
      GBM_result[nrow(GBM_result)+1, ] = c(GBM_CM$overall['Accuracy'], GBM_CM$byClass['Precision'],
                                           GBM_CM$byClass['Recall'], GBM_CM$byClass['F1'])
    }
    GBM_tuning[nrow(GBM_tuning)+1, ] = c(n, rl, colMeans(GBM_result)[1], colMeans(GBM_result)[2],
                                         colMeans(GBM_result)[3], colMeans(GBM_result)[4])
  }
}
GBM_tuning

## 5.3 Tuning n.minobsinnode -----
GBM_tuning2 <- data.frame(matrix(ncol=5, nrow=0))
colnames(GBM_tuning2) = c('n.minobsinnode', 'Accuracy', 'Precision', 'Recall', 'F1')
for (n in seq(10, 100, 10)){
  GBM_result <- data.frame(matrix(ncol=4, nrow=0))
  colnames(GBM_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
  for (i in 1:10){
    train <- resort_GBM[-folds[[i]],] 
    test <- resort_GBM[folds[[i]],] 
    
    GBM <- gbm(is_canceled~., data=train, distribution="bernoulli", n.trees=500, shrinkage=0.02, 
               n.minobsinnode=n, bag.fraction=0.5, verbose=FALSE)
    # summary(GBM) #compute relative inference of each variable
    
    GBM.pred <- predict.gbm(GBM, test, type='response', verbose=FALSE)
    GBM.pred[GBM.pred < 0.35] <- 0
    GBM.pred[GBM.pred >= 0.35] <- 1
    GBM.pred <- factor(GBM.pred, levels = c(1,0))
    
    test$is_canceled <- factor(test$is_canceled, levels = c(1,0))
    GBM_CM <- confusionMatrix(GBM.pred, test$is_canceled)
    GBM_result[nrow(GBM_result)+1, ] = c(GBM_CM$overall['Accuracy'], GBM_CM$byClass['Precision'],
                                         GBM_CM$byClass['Recall'], GBM_CM$byClass['F1'])
  }
  GBM_tuning2[nrow(GBM_tuning2)+1, ] = c(n, colMeans(GBM_result)[1], colMeans(GBM_result)[2],
                                       colMeans(GBM_result)[3], colMeans(GBM_result)[4])
  # print(n)
  # print(colMeans(GBM_result))
}
GBM_tuning2


## 5.4 Tuning bag.fraction ----
GBM_tuning3 <- data.frame(matrix(ncol=5, nrow=0))
colnames(GBM_tuning3) = c('bag.fraction', 'Accuracy', 'Precision', 'Recall', 'F1')
for (n in seq(0.3, 0.7, 0.1)){
  GBM_result <- data.frame(matrix(ncol=4, nrow=0))
  colnames(GBM_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
  for (i in 1:10){
    train <- resort_GBM[-folds[[i]],] 
    test <- resort_GBM[folds[[i]],] 
    
    GBM <- gbm(is_canceled~., data=train, distribution="bernoulli", n.trees=500, shrinkage=0.02, 
               n.minobsinnode=80, bag.fraction=n, verbose=FALSE)
    # summary(GBM) #compute relative inference of each variable
    
    GBM.pred <- predict.gbm(GBM, test, type='response', verbose=FALSE)
    GBM.pred[GBM.pred < 0.35] <- 0
    GBM.pred[GBM.pred >= 0.35] <- 1
    GBM.pred <- factor(GBM.pred, levels = c(1,0))
    
    test$is_canceled <- factor(test$is_canceled, levels = c(1,0))
    GBM_CM <- confusionMatrix(GBM.pred, test$is_canceled)
    GBM_result[nrow(GBM_result)+1, ] = c(GBM_CM$overall['Accuracy'], GBM_CM$byClass['Precision'],
                                         GBM_CM$byClass['Recall'], GBM_CM$byClass['F1'])
  }
  GBM_tuning3[nrow(GBM_tuning3)+1, ] = c(n, colMeans(GBM_result)[1], colMeans(GBM_result)[2],
                                         colMeans(GBM_result)[3], colMeans(GBM_result)[4])
}
GBM_tuning3

## 5.5 Final result ----
GBM_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(GBM_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
for (i in 1:10){
  train <- resort_GBM[-folds[[i]],] 
  test <- resort_GBM[folds[[i]],] 
  
  GBM <- gbm(is_canceled~., data=train, distribution="bernoulli", n.trees=500, shrinkage=0.02, 
             n.minobsinnode=80, bag.fraction=0.4, verbose=FALSE)
  # summary(GBM) #compute relative inference of each variable
  
  GBM.pred <- predict.gbm(GBM, test, type='response', verbose=FALSE)
  GBM.pred[GBM.pred < 0.35] <- 0
  GBM.pred[GBM.pred >= 0.35] <- 1
  GBM.pred <- factor(GBM.pred, levels = c(1,0))
  
  test$is_canceled <- factor(test$is_canceled, levels = c(1,0))
  GBM_CM <- confusionMatrix(GBM.pred, test$is_canceled)
  GBM_result[nrow(GBM_result)+1, ] = c(GBM_CM$overall['Accuracy'], GBM_CM$byClass['Precision'],
                                       GBM_CM$byClass['Recall'], GBM_CM$byClass['F1'])
}
colMeans(GBM_result)
#summary(GBM)