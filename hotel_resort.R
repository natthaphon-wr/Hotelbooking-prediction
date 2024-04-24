source('preprocess.R')
library(tree)
library(tidyverse) 
library(caret)
library(randomForest)
library(gbm)
library(xgboost)
library(lightgbm)
library(data.table)
library(catboost)


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
for (i in 1:1){
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
colMeans(DT_result)
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
set.seed(37)
RF_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(RF_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
RF_imp <- matrix(0, nrow = ncol(resort)-1, ncol=1)
rownames(RF_imp) <- colnames(resort %>% select(-is_canceled))
colnames(RF_imp) <- c('MeanDecreaseGini')
for (i in 1:10){
  print(i)
  train <- resort[-folds[[i]],] 
  test <- resort[folds[[i]],] 
  
  RF <- randomForest(is_canceled~., data=train, ntree=100, mtry=10, nodesize=1)
  RF_imp <- RF_imp + importance(RF)
  
  RF.pred <- predict(RF, test, type='response')
  RF_CM <- confusionMatrix(RF.pred, test$is_canceled)
  RF_result[nrow(RF_result)+1, ] = c(RF_CM$overall['Accuracy'], RF_CM$byClass['Precision'],
                                     RF_CM$byClass['Recall'], RF_CM$byClass['F1'])
}
colMeans(RF_result)

# Feature importance
RF_imp <- RF_imp/10
RF_imp <- data.frame('MeanDecreaseGini' = RF_imp)
RF_imp20 <- top_n(RF_imp, 20, MeanDecreaseGini)
ggplot(RF_imp20, aes(y=reorder(rownames(RF_imp20), MeanDecreaseGini), x=MeanDecreaseGini)) +
  geom_bar(stat = "identity") +
  ylab('Features') +
  ggtitle('Resort Hotel using Random Forest')


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
set.seed(37)
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


# 6. XGBoost ----
# Parameters Tuning
#   - eta = learning rate
#   - nrounds = max no. of boosting iterations
#   - gamma = for avoid overfitting 
#   - max_depth 
#   - min_child_weight
#   - subsample = subsample ratio for growing tree
#   - colsample_bytree = subsample ratio of columns

resort_xgb <- resort
resort_xgb$is_canceled <- unclass(resort_xgb$is_canceled)%%2 #Change variable type
train_control = trainControl(method = "cv", number = 10, search = "grid")

## 6.1 Initial nrounds and learning rate ----
xgbGrid1 <-  expand.grid(eta = c(0.01, 0.1, 0.3, 0.5), 
                         nrounds = c(50, 100, 500, 1000),
                         # fixed values below
                         max_depth = 6, 
                         min_child_weight = 10,
                         gamma = 0,
                         subsample = 1,
                         colsample_bytree = 1
                         )
xgbModel1 <- train(is_canceled~., 
                   data = resort,
                   method = "xgbTree", 
                   trControl = train_control, 
                   tuneGrid = xgbGrid1,
                   verbosity = 0)
print(xgbModel1)


## 6.2 Tune max_depth, min_child_weight ----
xgbGrid2 <-  expand.grid(max_depth = c(1, 5, 10, 15),
                         min_child_weight = c(1, 10, 50, 100),
                         # fixed values below
                         eta = 0.3,
                         nrounds = 500,
                         gamma = 0,
                         subsample = 1,
                         colsample_bytree = 1
                         )
xgbModel2 <- train(is_canceled~., 
                   data = resort,
                   method = "xgbTree", 
                   trControl = train_control, 
                   tuneGrid = xgbGrid2,
                   verbosity = 0)
print(xgbModel2)


## 6.3 Tune gamma ----
xgbGrid3 <-  expand.grid(gamma = c(0, 0.1, 0.2, 0.3, 0.4),
                         # fixed values below
                         eta = 0.3,
                         nrounds = 500,
                         max_depth = 10,
                         min_child_weight = 1,
                         subsample = 1,
                         colsample_bytree = 1
)
xgbModel3 <- train(is_canceled~., 
                   data = resort,
                   method = "xgbTree", 
                   trControl = train_control, 
                   tuneGrid = xgbGrid3,
                   verbosity = 0)
print(xgbModel3)

## 6.4 Tune subsample and colsample_bytree ----
xgbGrid4 <-  expand.grid(subsample = c(0.9, 1),
                         colsample_bytree = c(0.1, 0.3, 0.5), 
                         # fixed values below
                         gamma = 0.1,
                         eta = 0.3,
                         nrounds = 500,
                         max_depth = 10,
                         min_child_weight = 1
                         )
xgbModel4 <- train(is_canceled~., 
                   data = resort,
                   method = "xgbTree", 
                   trControl = train_control, 
                   tuneGrid = xgbGrid4,
                   verbosity = 0)
print(xgbModel4)


## 6.5 Adjust learning rate and rounds ----
xgbGrid5 <-  expand.grid(eta = c(0.01, 0.1, 0.3), 
                         nrounds = c(100, 500, 1000, 2000),
                         # fixed values below
                         max_depth = 10, 
                         min_child_weight = 1,
                         gamma = 0.1,
                         subsample = 1,
                         colsample_bytree = 0.5
)
xgbModel5 <- train(is_canceled~., 
                   data = resort,
                   method = "xgbTree", 
                   trControl = train_control, 
                   tuneGrid = xgbGrid5,
                   verbosity = 0)

## 6.6 Tune classification threshold ----
xgb_params <- list(eta=0.1, gamma=0.1, max_depth=10, min_child_weight=1, subsample=1, colsample_bytree=0.5,
                   booster = "gbtree", objective="binary:logistic", eval_metric="error")
xgb_threshold <- data.frame(matrix(ncol=5, nrow=0))
colnames(xgb_threshold) = c('Threshold', 'Accuracy', 'Precision', 'Recall', 'F1')
for (n in seq(0.6, 0.7, 0.1)){
  XGB_result <- data.frame(matrix(ncol=4, nrow=0))
  colnames(XGB_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
  for (i in 1:10){
    train <- resort_xgb[-folds[[i]],]
    test <- resort_xgb[folds[[i]],]
  
    dtrain = xgb.DMatrix(as.matrix(sapply(train %>% select(-is_canceled), as.numeric)),
                         label=train$is_canceled)
    dtest = xgb.DMatrix(as.matrix(sapply(test %>% select(-is_canceled), as.numeric)),
                        label=test$is_canceled)
    XGB <- xgboost(params = xgb_params, data = dtrain, nrounds = 1000, verbose=0)
  
    xgb.pred <- predict(XGB, newdata=dtest)
    xgb.pred <- ifelse (xgb.pred >= n, 1, 0)
    xgb.pred <- factor(xgb.pred, levels = c(1,0))
    
    test$is_canceled <- factor(test$is_canceled, levels = c(1,0))
    xgb.cm <- confusionMatrix(xgb.pred, test$is_canceled)
    XGB_result[nrow(XGB_result)+1, ] = c(xgb.cm$overall['Accuracy'], xgb.cm$byClass['Precision'],
                                         xgb.cm$byClass['Recall'], xgb.cm$byClass['F1'])
  }
  xgb_threshold[nrow(xgb_threshold)+1, ] = c(n, colMeans(XGB_result)[1], colMeans(XGB_result)[2],
                                             colMeans(XGB_result)[3], colMeans(XGB_result)[4])
  
}
xgb_threshold

## 6.7 Final result ----
set.seed(37)
xgb_params <- list(eta=0.1, gamma=0.1, max_depth=10, min_child_weight=1, subsample=1, colsample_bytree=0.5,
                   booster = "gbtree", objective="binary:logistic", eval_metric="error")
XGB_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(XGB_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
XGB_imp <- matrix(0, nrow = ncol(resort)-1, ncol=0)
XGB_imp <- cbind(XGB_imp, colnames(resort %>% select(-is_canceled)))
colnames(XGB_imp) <- c('Feature')
for (i in 1:10){
  print(i)
  train <- resort_xgb[-folds[[i]],]
  test <- resort_xgb[folds[[i]],]
  
  dtrain = xgb.DMatrix(as.matrix(sapply(train %>% select(-is_canceled), as.numeric)),
                       label=train$is_canceled)
  dtest = xgb.DMatrix(as.matrix(sapply(test %>% select(-is_canceled), as.numeric)),
                      label=test$is_canceled)
  XGB <- xgboost(params = xgb_params, data = dtrain, nrounds = 1000, verbose=0)
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
colMeans(XGB_result)

# Feature importance
XGB_imp[is.na(XGB_imp)] <- 0
XGB_imp$MeansGain <- rowMeans(XGB_imp[2:11])

XGB_imp20 <- top_n(XGB_imp, 20, MeansGain)
ggplot(XGB_imp20, aes(y=reorder(Feature, MeansGain), x=MeansGain)) +
  geom_bar(stat = "identity") +
  xlab('Gain') +
  ylab('Features') +
  ggtitle('Resort Hotel using XGBoost')


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

# params_dart <- list(objective='binary', data_sample_strategy='goss', boosting='dart', 
#                 num_iterations=100, learning_rate=0.1, num_leaves=31, 
#                 min_data_in_leaf=20, 
#                 feature_fraction=1, 
#                 lambda_l1=0, lambda_l2=0,
#                 top_rate=0.2, other_rate=0.1
#                 xgboost_dart_mode=true, drop_rate=0.1, max_drop=50,
#                 skip_drop=0.5, uniform_drop=false)

resort_lgmb <- resort
resort_lgmb$is_canceled <- unclass(resort_lgmb$is_canceled)%%2 #Change variable type

## 7.1 Boosting: gbdt ----

### 7.1.1 Initial tree ----
# num_iterations, learning_rate, num_leaves
grid1<- expand.grid(num_iterations = c(1000),
                    learning_rate = c(0.01),
                    num_leaves = c(100, 250, 500))
lgbm_cvtune1 <- data.frame(matrix(ncol=5, nrow=0))
colnames(lgbm_cvtune1) = c('CV_round', 'num_iterations', 'learning_rate', 'num_leaves', 'binary_logloss')

for (i in 1:10){
  train <- resort_lgmb[-folds[[i]],]
  test <- resort_lgmb[folds[[i]],]
  dtrain <- lgb.Dataset(as.matrix(sapply(train %>% select(-is_canceled), as.numeric)),
                        label=train$is_canceled)
  dtest <- lgb.Dataset.create.valid(dataset = dtrain, 
                                    data = as.matrix(sapply(test %>% select(-is_canceled), as.numeric)), 
                                    label = test$is_canceled)
  valids <- list(test = dtest)
  
  model <- list()
  loss <- numeric(nrow(grid1))
  for (j in 1:nrow(grid1)){
    model[[j]] <- lgb.train(params = list(num_iterations = grid1[j, 'num_iterations'],
                                          learning_rate = grid1[j, 'learning_rate'],
                                          num_leaves = grid1[j, 'num_leaves'],
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
  lgbm_cvtune1[nrow(lgbm_cvtune1)+1, ] = c(i,
                                           grid1[which.min(loss), "num_iterations"],
                                           grid1[which.min(loss), "learning_rate"],
                                           grid1[which.min(loss), "num_leaves"],
                                           min(loss))
  print(lgbm_cvtune1[i,])  
}
lgbm_cvtune1

### 7.1.2 Tuning tree parameters ----
# min_data_in_leaf
grid2<- expand.grid(min_data_in_leaf = c(1, 3, 5, 10))
lgbm_cvtune2 <- data.frame(matrix(ncol=3, nrow=0))
colnames(lgbm_cvtune2) = c('CV_round', 'min_data_in_leaf', 'binary_logloss')

for (i in 1:10){
  train <- resort_lgmb[-folds[[i]],]
  test <- resort_lgmb[folds[[i]],]
  dtrain <- lgb.Dataset(as.matrix(sapply(train %>% select(-is_canceled), as.numeric)),
                        label=train$is_canceled)
  dtest <- lgb.Dataset.create.valid(dataset = dtrain, 
                                    data = as.matrix(sapply(test %>% select(-is_canceled), as.numeric)), 
                                    label = test$is_canceled)
  valids <- list(test = dtest)
  
  model <- list()
  loss <- numeric(nrow(grid2))
  for (j in 1:nrow(grid2)){
    model[[j]] <- lgb.train(params = list(min_data_in_leaf = grid2[j, 'min_data_in_leaf'],   
                                          # fixed value below
                                          objective = 'binary', 
                                          data_sample_strategy = 'goss',
                                          boosting = 'gbdt', 
                                          num_iterations = 1000,
                                          learning_rate = 0.01,
                                          num_leaves = 250,
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
  lgbm_cvtune2[nrow(lgbm_cvtune2)+1, ] = c(i,
                                           grid2[which.min(loss), "min_data_in_leaf"],
                                           min(loss))
  print(lgbm_cvtune2[i,])  
}
lgbm_cvtune2

### 7.1.3 Tuning feature_fraction ----
# feature_fraction
grid3 <- expand.grid(feature_fraction = c(0.8, 0.9, 1))
lgbm_cvtune3 <- data.frame(matrix(ncol=3, nrow=0))
colnames(lgbm_cvtune3) = c('CV_round', 'feature_fraction', 'binary_logloss')
for (i in 1:10){
  train <- resort_lgmb[-folds[[i]],]
  test <- resort_lgmb[folds[[i]],]
  dtrain <- lgb.Dataset(as.matrix(sapply(train %>% select(-is_canceled), as.numeric)),
                        label=train$is_canceled)
  dtest <- lgb.Dataset.create.valid(dataset = dtrain, 
                                    data = as.matrix(sapply(test %>% select(-is_canceled), as.numeric)), 
                                    label = test$is_canceled)
  valids <- list(test = dtest)
  
  model <- list()
  loss <- numeric(nrow(grid3))
  for (j in 1:nrow(grid3)){
    model[[j]] <- lgb.train(params = list(feature_fraction = grid3[j, 'feature_fraction'], 
                                          # fixed value below
                                          objective = 'binary', 
                                          data_sample_strategy = 'goss',
                                          boosting = 'gbdt', 
                                          num_iterations = 1000,
                                          learning_rate = 0.01,
                                          num_leaves = 250,
                                          min_data_in_leaf = 1,
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
  lgbm_cvtune3[nrow(lgbm_cvtune3)+1, ] = c(i,
                                           grid3[which.min(loss), "feature_fraction"],
                                           min(loss))
  print(loss)
  print(lgbm_cvtune3[i,])  
}
lgbm_cvtune3

### 7.1.4 Tuning goss rates ----
# top_rate, other_rate 
grid4 <- expand.grid(top_rate = c(0.1, 0.2, 0.5),
                     other_rate = c(0.1, 0.2, 0.5))
lgbm_cvtune4 <- data.frame(matrix(ncol=4, nrow=0))
colnames(lgbm_cvtune4) = c('CV_round', 'top_rate', 'other_rate', 'binary_logloss')

for (i in 1:10){
  train <- resort_lgmb[-folds[[i]],]
  test <- resort_lgmb[folds[[i]],]
  dtrain <- lgb.Dataset(as.matrix(sapply(train %>% select(-is_canceled), as.numeric)),
                        label=train$is_canceled)
  dtest <- lgb.Dataset.create.valid(dataset = dtrain, 
                                    data = as.matrix(sapply(test %>% select(-is_canceled), as.numeric)), 
                                    label = test$is_canceled)
  valids <- list(test = dtest)
  
  model <- list()
  loss <- numeric(nrow(grid4))
  for (j in 1:nrow(grid4)){
    model[[j]] <- lgb.train(params = list(top_rate = grid4[j, 'top_rate'], 
                                          other_rate = grid4[j, 'other_rate'],
                                          # fixed value below
                                          objective = 'binary', 
                                          data_sample_strategy = 'goss',
                                          boosting = 'gbdt',
                                          num_iterations = 1000,
                                          learning_rate = 0.01,
                                          num_leaves = 250,
                                          min_data_in_leaf = 1,
                                          feature_fraction = 0.8, 
                                          lambda_l1 = 0, 
                                          lambda_l2 = 0
                                        ),
    data = dtrain,
    valids = valids,
    verbose = 0
    )
    loss[j] <- min(rbindlist(model[[j]]$record_evals$test$binary_logloss))
  }
  lgbm_cvtune4[nrow(lgbm_cvtune4)+1, ] = c(i,
                                           grid4[which.min(loss), "top_rate"],
                                           grid4[which.min(loss), "other_rate"],
                                           min(loss))
  print(loss)
  print(lgbm_cvtune4[i,])  
}
lgbm_cvtune4

### 7.1.5 Tuning regularization ----
# lambda_l1, lambda_l2
grid5 <- expand.grid(lambda_l1 = c(0.1, 0.2),
                     lambda_l2 = c(0.1, 0.2))
lgbm_cvtune5 <- data.frame(matrix(ncol=4, nrow=0))
colnames(lgbm_cvtune5) = c('CV_round', 'lambda_l1', 'lambda_l2', 'binary_logloss')

for (i in 1:10){
  train <- resort_lgmb[-folds[[i]],]
  test <- resort_lgmb[folds[[i]],]
  dtrain <- lgb.Dataset(as.matrix(sapply(train %>% select(-is_canceled), as.numeric)),
                        label=train$is_canceled)
  dtest <- lgb.Dataset.create.valid(dataset = dtrain, 
                                    data = as.matrix(sapply(test %>% select(-is_canceled), as.numeric)), 
                                    label = test$is_canceled)
  valids <- list(test = dtest)
  
  model <- list()
  loss <- numeric(nrow(grid5))
  for (j in 1:nrow(grid5)){
    model[[j]] <- lgb.train(params = list(lambda_l1 = grid5[j, 'lambda_l1'], 
                                          lambda_l2 = grid5[j, 'lambda_l2'],
                                          # fixed value below
                                          objective = 'binary', 
                                          data_sample_strategy = 'goss',
                                          boosting = 'gbdt',
                                          num_iterations = 1000,
                                          learning_rate = 0.01,
                                          num_leaves = 250,
                                          min_data_in_leaf = 1,
                                          feature_fraction = 0.8, 
                                          top_rate = 0.2, 
                                          other_rate = 0.5
                                        ),
    data = dtrain,
    valids = valids,
    verbose = 0
    )
    loss[j] <- min(rbindlist(model[[j]]$record_evals$test$binary_logloss))
  }
  lgbm_cvtune5[nrow(lgbm_cvtune5)+1, ] = c(i,
                                           grid5[which.min(loss), "lambda_l1"],
                                           grid5[which.min(loss), "lambda_l2"],
                                           min(loss))
  print(loss)
  print(lgbm_cvtune5[i,])  
}
lgbm_cvtune5


### 7.1.6 Tuning no.tree and learning rate again ----
# num_iterations, learning_rate, num_leaves
grid6 <- expand.grid(num_iterations = c(1000),
                    learning_rate = c(0.01, 0.05, 0.1),
                    num_leaves = c(250))
lgbm_cvtune6 <- data.frame(matrix(ncol=5, nrow=0))
colnames(lgbm_cvtune6) = c('CV_round', 'num_iterations', 'learning_rate', 'num_leaves', 'binary_logloss')

for (i in 6:10){
  train <- resort_lgmb[-folds[[i]],]
  test <- resort_lgmb[folds[[i]],]
  dtrain <- lgb.Dataset(as.matrix(sapply(train %>% select(-is_canceled), as.numeric)),
                        label=train$is_canceled)
  dtest <- lgb.Dataset.create.valid(dataset = dtrain, 
                                    data = as.matrix(sapply(test %>% select(-is_canceled), as.numeric)), 
                                    label = test$is_canceled)
  valids <- list(test = dtest)
  
  model <- list()
  loss <- numeric(nrow(grid6))
  for (j in 1:nrow(grid6)){
    model[[j]] <- lgb.train(params = list(num_iterations = grid6[j, 'num_iterations'],
                                          learning_rate = grid6[j, 'learning_rate'],
                                          num_leaves = grid6[j, 'num_leaves'],
                                          # fixed value below
                                          objective = 'binary', 
                                          data_sample_strategy = 'goss',
                                          boosting = 'gbdt', 
                                          min_data_in_leaf = 1, 
                                          feature_fraction = 0.8,
                                          top_rate = 0.2, 
                                          other_rate = 0.5,
                                          lambda_l1 = 0, 
                                          lambda_l2 = 0
                                        ),
    data = dtrain,
    valids = valids,
    verbose = 0
    )
    loss[j] <- min(rbindlist(model[[j]]$record_evals$test$binary_logloss))
  }
  lgbm_cvtune6[nrow(lgbm_cvtune6)+1, ] = c(i,
                                           grid6[which.min(loss), "num_iterations"],
                                           grid6[which.min(loss), "learning_rate"],
                                           grid6[which.min(loss), "num_leaves"],
                                           min(loss))
  print(loss)
  print(lgbm_cvtune6[i,])  
}
lgbm_cvtune6


## 7.2 Boosting: dart ----

### 7.2.1 Uniform drop ----
grid_d1 <- expand.grid(uniform_drop = c(TRUE, FALSE))
lgbm_cvtune_d1 <- data.frame(matrix(ncol=3, nrow=0))
colnames(lgbm_cvtune_d1) = c('CV_round', 'uniform_drop', 'binary_logloss')

for (i in 1:10){
  train <- resort_lgmb[-folds[[i]],]
  test <- resort_lgmb[folds[[i]],]
  dtrain <- lgb.Dataset(as.matrix(sapply(train %>% select(-is_canceled), as.numeric)),
                        label=train$is_canceled)
  dtest <- lgb.Dataset.create.valid(dataset = dtrain, 
                                    data = as.matrix(sapply(test %>% select(-is_canceled), as.numeric)), 
                                    label = test$is_canceled)
  valids <- list(test = dtest)
  
  model <- list()
  loss <- numeric(nrow(grid_d1))
  for (j in 1:nrow(grid_d1)){
    model[[j]] <- lgb.train(params = list(uniform_drop = grid_d1[j, "uniform_drop"],
                                          # fixed value below
                                          objective = 'binary', 
                                          data_sample_strategy = 'goss',
                                          boosting = 'dart',
                                          num_iterations = 1000,
                                          learning_rate = 0.01,
                                          num_leaves = 250,
                                          min_data_in_leaf = 1, 
                                          feature_fraction = 0.8,
                                          top_rate = 0.2, 
                                          other_rate = 0.5,
                                          lambda_l1 = 0, 
                                          lambda_l2 = 0,
                                          xgboost_dart_mode = TRUE,
                                          drop_rate = 0.1,
                                          max_drop = 50,
                                          skip_drop = 0.5
                                        ),
    data = dtrain,
    valids = valids,
    verbose = 0
    )
    loss[j] <- min(rbindlist(model[[j]]$record_evals$test$binary_logloss))
  }
  lgbm_cvtune_d1[nrow(lgbm_cvtune_d1)+1, ] = c(i,
                                           grid_d1[which.min(loss), "uniform_drop"],
                                           min(loss))
  print(loss)
  print(lgbm_cvtune_d1[i,])  
}
lgbm_cvtune_d1

### 7.2.2 Tune drop_rate ----
grid_d2 <- expand.grid(drop_rate = c(0.1, 0.2, 0.3))
lgbm_cvtune_d2 <- data.frame(matrix(ncol=3, nrow=0))
colnames(lgbm_cvtune_d2) = c('CV_round', 'drop_rate', 'binary_logloss')

for (i in 1:10){
  train <- resort_lgmb[-folds[[i]],]
  test <- resort_lgmb[folds[[i]],]
  dtrain <- lgb.Dataset(as.matrix(sapply(train %>% select(-is_canceled), as.numeric)),
                        label=train$is_canceled)
  dtest <- lgb.Dataset.create.valid(dataset = dtrain, 
                                    data = as.matrix(sapply(test %>% select(-is_canceled), as.numeric)), 
                                    label = test$is_canceled)
  valids <- list(test = dtest)
  
  model <- list()
  loss <- numeric(nrow(grid_d2))
  for (j in 1:nrow(grid_d2)){
    model[[j]] <- lgb.train(params = list(drop_rate = grid_d2[j, "drop_rate"],
                                          # fixed value below
                                          objective = 'binary', 
                                          data_sample_strategy = 'goss',
                                          boosting = 'dart',
                                          num_iterations = 1000,
                                          learning_rate = 0.01,
                                          num_leaves = 250,
                                          min_data_in_leaf = 1, 
                                          feature_fraction = 0.8,
                                          top_rate = 0.2, 
                                          other_rate = 0.5,
                                          lambda_l1 = 0, 
                                          lambda_l2 = 0,
                                          xgboost_dart_mode = TRUE,
                                          uniform_drop = TRUE,
                                          max_drop = 50,
                                          skip_drop = 0.5
    ),
    data = dtrain,
    valids = valids,
    verbose = 0
    )
    loss[j] <- min(rbindlist(model[[j]]$record_evals$test$binary_logloss))
  }
  lgbm_cvtune_d2[nrow(lgbm_cvtune_d2)+1, ] = c(i,
                                               grid_d2[which.min(loss), "drop_rate"],
                                               min(loss))
  print(loss)
  print(lgbm_cvtune_d2[i,])  
}
lgbm_cvtune_d2


### 7.2.3 Tune max_drop, skip_drop ----
grid_d3 <- expand.grid(max_drop = c(10, 50, 100),
                       skip_drop = c(0.25, 0.5, 0.75))
lgbm_cvtune_d3 <- data.frame(matrix(ncol=4, nrow=0))
colnames(lgbm_cvtune_d3) = c('CV_round', 'max_drop', 'skip_drop', 'binary_logloss')

for (i in 1:10){
  train <- resort_lgmb[-folds[[i]],]
  test <- resort_lgmb[folds[[i]],]
  dtrain <- lgb.Dataset(as.matrix(sapply(train %>% select(-is_canceled), as.numeric)),
                        label=train$is_canceled)
  dtest <- lgb.Dataset.create.valid(dataset = dtrain, 
                                    data = as.matrix(sapply(test %>% select(-is_canceled), as.numeric)), 
                                    label = test$is_canceled)
  valids <- list(test = dtest)
  
  model <- list()
  loss <- numeric(nrow(grid_d3))
  for (j in 1:nrow(grid_d3)){
    model[[j]] <- lgb.train(params = list(max_drop = grid_d3[j, "max_drop"],
                                          skip_drop = grid_d3[j, "skip_drop"],
                                          # fixed value below
                                          objective = 'binary', 
                                          data_sample_strategy = 'goss',
                                          boosting = 'dart',
                                          num_iterations = 1000,
                                          learning_rate = 0.01,
                                          num_leaves = 250,
                                          min_data_in_leaf = 1, 
                                          feature_fraction = 0.8,
                                          top_rate = 0.2, 
                                          other_rate = 0.5,
                                          lambda_l1 = 0, 
                                          lambda_l2 = 0,
                                          xgboost_dart_mode = TRUE,
                                          uniform_drop = TRUE,
                                          drop_rate = 0.1
                                          
                                        ),
    data = dtrain,
    valids = valids,
    verbose = 0
    )
    loss[j] <- min(rbindlist(model[[j]]$record_evals$test$binary_logloss))
  }
  lgbm_cvtune_d3[nrow(lgbm_cvtune_d3)+1, ] = c(i,
                                               grid_d3[which.min(loss), "max_drop"],
                                               grid_d3[which.min(loss), "skip_drop"],
                                               min(loss))
  print(loss)
  print(lgbm_cvtune_d3[i,])  
}
lgbm_cvtune_d3

## 7.3 Compare gbdt vs dart ----
set.seed(37)
lgbm_gbdt_params <- list(objective = 'binary', data_sample_strategy = 'goss', boosting = 'gbdt',
                         num_iterations = 1000, learning_rate = 0.01, num_leaves = 250,
                         min_data_in_leaf = 1, feature_fraction = 0.8,
                         top_rate = 0.2, other_rate = 0.5,
                         lambda_l1 = 0, lambda_l2 = 0)
lgbm_dart_params <- list(objective = 'binary', data_sample_strategy = 'goss', boosting = 'dart',
                         num_iterations = 1000, learning_rate = 0.01, num_leaves = 250,
                         min_data_in_leaf = 1, feature_fraction = 0.8,
                         top_rate = 0.2, other_rate = 0.5,
                         lambda_l1 = 0, lambda_l2 = 0,
                         xgboost_dart_mode = TRUE, uniform_drop = TRUE,
                         drop_rate = 0.1, max_drop = 10, skip_drop = 0.75)

loss_gbdt <- numeric(10)
loss_dart <- numeric(10)
for (i in 1:10){
  train <- resort_lgmb[-folds[[i]],]
  test <- resort_lgmb[folds[[i]],]
  dtrain <- lgb.Dataset(as.matrix(sapply(train %>% select(-is_canceled), as.numeric)),
                        label=train$is_canceled)
  dtest <- lgb.Dataset.create.valid(dataset = dtrain, 
                                    data = as.matrix(sapply(test %>% select(-is_canceled), as.numeric)), 
                                    label = test$is_canceled)
  valids <- list(test = dtest)
  
  model_gbdt <- lgb.train(params = lgbm_gbdt_params, data = dtrain, valids = valids, verbose = 0)
  model_dart <- lgb.train(params = lgbm_dart_params, data = dtrain, valids = valids, verbose = 0)
  
  loss_gbdt[i] <- min(rbindlist(model_gbdt$record_evals$test$binary_logloss))
  loss_dart[i] <- min(rbindlist(model_dart$record_evals$test$binary_logloss))
  
  print('Loss gbdt: ')
  print(loss_gbdt[i])
  print('Loss dart: ')
  print(loss_dart[i])
}


## 7.4 Tuning classification threshold ----
lgbm_params <- list(objective = 'binary', data_sample_strategy = 'goss', boosting = 'gbdt',
                    num_iterations = 1000, learning_rate = 0.1,num_leaves = 250,
                    min_data_in_leaf = 1, feature_fraction = 0.8,
                    top_rate = 0.2, other_rate = 0.5,
                    lambda_l1 = 0, lambda_l2 = 0)

lgbm_threshold <- data.frame(matrix(ncol=5, nrow=0))
colnames(lgbm_threshold) = c('Threshold', 'Accuracy', 'Precision', 'Recall', 'F1')

for (n in seq(0.3, 0.4, 0.1)){
  lgbm_result <- data.frame(matrix(ncol=4, nrow=0))
  colnames(lgbm_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
  for (i in 1:10){
    train <- resort_lgmb[-folds[[i]],]
    test <- resort_lgmb[folds[[i]],]
    dtrain <- lgb.Dataset(as.matrix(sapply(train %>% select(-is_canceled), as.numeric)),
                          label=train$is_canceled)
    dtest <- as.matrix(sapply(test %>% select(-is_canceled), as.numeric))
    
    LGBM <- lgb.train(params = lgbm_params,
                      data = dtrain,
                      verbose = 0
    )
    LGBM.pred <- predict(LGBM, newdata=dtest)
    LGBM.pred <- ifelse (LGBM.pred >= n, 1, 0)
    LGBM.pred <- factor(LGBM.pred, levels = c(1,0))
    
    test$is_canceled <- factor(test$is_canceled, levels = c(1,0))
    LGBM.cm <- confusionMatrix(LGBM.pred, test$is_canceled)
    lgbm_result[nrow(lgbm_result)+1, ] = c(LGBM.cm$overall['Accuracy'], LGBM.cm$byClass['Precision'],
                                           LGBM.cm$byClass['Recall'], LGBM.cm$byClass['F1'])
    
  }
  lgbm_threshold[nrow(lgbm_threshold)+1, ] = c(n, colMeans(lgbm_result)[1], colMeans(lgbm_result)[2],
                                               colMeans(lgbm_result)[3], colMeans(lgbm_result)[4])
  print(lgbm_threshold[nrow(lgbm_threshold), ])
}
lgbm_threshold

## 7.5 Final LGBM with gbdt boosting ----
set.seed(37)
lgbm_params <- list(objective = 'binary', data_sample_strategy = 'goss', boosting = 'gbdt',
                    num_iterations = 1000, learning_rate = 0.1,num_leaves = 250,
                    min_data_in_leaf = 1, feature_fraction = 0.8,
                    top_rate = 0.2, other_rate = 0.5,
                    lambda_l1 = 0, lambda_l2 = 0)
lgbm_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(lgbm_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
LGB_imp <- matrix(0, nrow = ncol(resort)-1, ncol=0)
LGB_imp <- cbind(LGB_imp, colnames(resort %>% select(-is_canceled)))
colnames(LGB_imp) <- c('Feature')
for (i in 1:10){
  print(i)
  train <- resort_lgmb[-folds[[i]],]
  test <- resort_lgmb[folds[[i]],]
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
colMeans(lgbm_result)

# Feature importance
LGB_imp[is.na(LGB_imp)] <- 0
LGB_imp$MeansGain <- rowMeans(LGB_imp[2:11])

LGB_imp20 <- top_n(LGB_imp, 20, MeansGain)
ggplot(LGB_imp20, aes(y=reorder(Feature, MeansGain), x=MeansGain)) +
  geom_bar(stat = "identity") +
  xlab('Gain') +
  ylab('Features') +
  ggtitle('Resort Hotel using LightGBM')


# 8. CatBoost ----
# Tuning parameters
# - depth = Tree Depth 
# - learning_rate
# - iterations = Number of trees
# - l2_leaf_reg = L2 regularization coefficient
# - rsm = The percentage of features to use at each iteration
# - border_count = The number of splits for numerical features

## Import new data ----
hotel_data2 <- preprocess(one_hot = FALSE)
resort2 <- hotel_data2[[2]]
sapply(resort2, class)
summary(resort2$is_canceled)
folds <- createFolds(resort2$is_canceled, k=10)

train_control = trainControl(method = "cv", number = 10, search = "grid", classProbs = TRUE)


## 8.1 Initial trees ----
# Tune iterations and learning_rate
catb_grid1 <- expand.grid(iterations = c(100, 500, 1000),
                          learning_rate = c(0.01, 0.05, 0.1, 0.3),
                          # fixed value
                          depth = 6,
                          l2_leaf_reg = 1e-6,
                          rsm = 0.9,
                          border_count = 255)
catb_Tune1 <- train(resort2 %>% select(-is_canceled), 
                    as.factor(make.names(resort2$is_canceled)),
                    method = catboost.caret,
                    tuneGrid = catb_grid1, 
                    trControl = train_control,
                    verbose = 0)
print(catb_Tune1)

## 8.2 Tune tree depth ----
catb_grid2 <- expand.grid(depth = c(4, 6, 8, 10, 12),
                          # fixed value
                          iterations = 1000,
                          learning_rate = 0.1,
                          l2_leaf_reg = 1e-6,
                          rsm = 0.9,
                          border_count = 255)
catb_Tune2 <- train(resort2 %>% select(-is_canceled), 
                    as.factor(make.names(resort2$is_canceled)),
                    method = catboost.caret,
                    tuneGrid = catb_grid2, 
                    trControl = train_control,
                    verbose = 0)
print(catb_Tune2)

## 8.3 Tune rsm ----
catb_grid3 <- expand.grid(rsm = c(0.6, 0.8, 1),
                          # fixed value
                          iterations = 1000,
                          learning_rate = 0.1,
                          depth = 8,
                          l2_leaf_reg = 1e-3,
                          border_count = 255)
catb_Tune3 <- train(resort2 %>% select(-is_canceled), 
                    as.factor(make.names(resort2$is_canceled)),
                    method = catboost.caret,
                    tuneGrid = catb_grid3, 
                    trControl = train_control,
                    verbose = 0)
print(catb_Tune3)

## 8.4 Tune border_count ----
catb_grid4 <- expand.grid(border_count = c(512),
                          # fixed value
                          iterations = 1000,
                          learning_rate = 0.1,
                          depth = 8,
                          rsm = 1,
                          l2_leaf_reg = 1e-3
                          )
catb_Tune4 <- train(resort2 %>% select(-is_canceled), 
                    as.factor(make.names(resort2$is_canceled)),
                    method = catboost.caret,
                    tuneGrid = catb_grid4, 
                    trControl = train_control,
                    verbose = 0)
print(catb_Tune4)

## 8.5 Tune L2 reg ----
catb_grid5 <- expand.grid(l2_leaf_reg = c(0.05),
                          # fixed value
                          iterations = 1000,
                          learning_rate = 0.1,
                          depth = 8,
                          rsm = 1,
                          border_count = 512
                          )
catb_Tune5 <- train(resort2 %>% select(-is_canceled), 
                    as.factor(make.names(resort2$is_canceled)),
                    method = catboost.caret,
                    tuneGrid = catb_grid5, 
                    trControl = train_control,
                    verbose = 0)
print(catb_Tune5)


## 8.6 Tune classification threshold ----
sapply(resort2, class)
# resort2$is_canceled <- unclass(resort2$is_canceled)%%2 #Change variable type

catb_params <- list(iterations = 1000, 
                    learning_rate = 0.1,
                    rsm = 1,
                    depth = 8,
                    border_count = 512,
                    l2_leaf_reg = 0.01,
                    logging_level = 'Silent'
                    )

catb_threshold <- data.frame(matrix(ncol=5, nrow=0))
colnames(catb_threshold) = c('Threshold', 'Accuracy', 'Precision', 'Recall', 'F1')
for (n in seq(0.55, 0.7, 0.05)){
  catb_result <- data.frame(matrix(ncol=4, nrow=0))
  colnames(catb_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
  
  for (i in 1:10){
    print(i)
    train <- resort2[-folds[[i]],]
    test <- resort2[folds[[i]],]
    
    train_pool <- catboost.load_pool(data = train %>% select(-is_canceled), 
                                     label = unclass(train$is_canceled)%%2)
    test_pool <- catboost.load_pool(data = test %>% select(-is_canceled), 
                                    label = unclass(test$is_canceled)%%2)
  
    catb_model <- catboost.train(train_pool, params = catb_params)
    
    catb.pred <- catboost.predict(catb_model, test_pool,prediction_type = 'Probability')
    catb.pred <- ifelse (catb.pred >= n, 1, 0)
    catb.pred <- factor(catb.pred, levels = c(1,0))
    
    test$is_canceled <- factor(test$is_canceled, levels = c(1,0))
    catb.cm <- confusionMatrix(catb.pred, test$is_canceled)
    catb_result[nrow(catb_result)+1, ] = c(catb.cm$overall['Accuracy'], catb.cm$byClass['Precision'],
                                           catb.cm$byClass['Recall'], catb.cm$byClass['F1'])
  }
  catb_threshold[nrow(catb_threshold)+1, ] = c(n, colMeans(catb_result)[1], colMeans(catb_result)[2],
                                               colMeans(catb_result)[3], colMeans(catb_result)[4])
}
catb_threshold

## 8.7 Final CatBoost ----
catb_params <- list(iterations = 1000, 
                    learning_rate = 0.1,
                    rsm = 1,
                    depth = 8,
                    border_count = 512,
                    l2_leaf_reg = 0.01,
                    logging_level = 'Silent')

catb_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(catb_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
catb_imp <- matrix(0, nrow = ncol(resort2)-1, ncol=0)
catb_imp <- cbind(catb_imp, colnames(resort2 %>% select(-is_canceled)))
colnames(catb_imp) <- c('Feature')

for (i in 1:10){
  print(i)
  train <- resort2[-folds[[i]],]
  test <- resort2[folds[[i]],]
  
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

# Feature importance
catb_imp$MeansFI <- rowMeans(catb_imp[2:11])
ggplot(catb_imp, aes(y=reorder(Feature, MeansFI), x=MeansFI)) +
  geom_bar(stat = "identity") +
  xlab('Importance') +
  ylab('Features') +
  ggtitle('Resort Hotel using CatBoost')


