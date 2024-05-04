source('preprocess.R')
source('onehot_encode.R')
source('rf_tuning.R')
source('gbm_tuning.R')
source('xgb_tuning.R')
source('catb_tuning.R')

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
## 1.1 Not one hot encoding ----
hotel_data <- preprocess(one_hot = FALSE, feature_select = TRUE)
city <- hotel_data[[3]]
sapply(city, class)
summary(city$is_canceled)

## 1.2 One hot encoding ----
hotel_data_1hot <- preprocess(one_hot = TRUE, feature_select = TRUE)
city_1hot <- hotel_data_1hot[[3]]
sapply(city_1hot, class)
summary(city_1hot$is_canceled)


# 2. Outer loop splitting ----
outer_folds <- createFolds(city$is_canceled, k=5)


# 3. Random forest ----
# Parameters Tuning
#   - ntree = no.of trees (row samples size = nrow(train))
#   - mtry = no. features that samples
#   - nodesize  = min size of terminal nodes

## 3.1 Tune ntree ----
RF_ntree_params <- list(ntree = seq(100, 500, 200),
                        mtry = 20,
                        nodesize = 1)
RF_ntree <- rf_tuning(data = city, 
                      outer_folds = outer_folds, 
                      Ntree = RF_ntree_params$ntree, 
                      Mtry = RF_ntree_params$mtry, 
                      Nodesize = RF_ntree_params$nodesize, 
                      param = 'ntree')
RF_ntree %>% 
  group_by(ntree) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))
write.csv(RF_ntree, ".\\result\\RF_city_ntree.csv", row.names=FALSE)


## 3.2 Tune mtry  ----
RF_mtry_params <- list(ntree = 500,
                       mtry = c(5, 10, 20, 30),
                       nodesize = 1)
RF_mtry <- rf_tuning(data = city, 
                     outer_folds = outer_folds, 
                     Ntree = RF_mtry_params$ntree, 
                     Mtry = RF_mtry_params$mtry, 
                     Nodesize = RF_mtry_params$nodesize, 
                     param = 'mtry')
RF_mtry %>% 
  group_by(mtry) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))
write.csv(RF_mtry, ".\\result\\RF_city_mtry.csv", row.names=FALSE)


## 3.3 Tune nodesize ----
RF_nodesize_params <- list(ntree = 500,
                           mtry = 20,
                           nodesize = c(1, 5, 10))
RF_nodesize <- rf_tuning(data = city, 
                         outer_folds = outer_folds, 
                         Ntree = RF_nodesize_params$ntree, 
                         Mtry = RF_nodesize_params$mtry, 
                         Nodesize = RF_nodesize_params$nodesize, 
                         param = 'nodesize')
RF_nodesize %>% 
  group_by(nodesize) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))
write.csv(RF_nodesize, ".\\result\\RF_city_nodesize.csv", row.names=FALSE)


## 3.4 Final RF ----
RF_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(RF_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
RF_imp <- matrix(0, nrow = ncol(city_1hot)-1, ncol=1)
rownames(RF_imp) <- colnames(city_1hot %>% select(-is_canceled))
colnames(RF_imp) <- c('MeanDecreaseGini')

for (i in 1:5){
  print(i)
  train <- city[-outer_folds[[i]],] 
  train <- onehot_encode(train)
  test <- city[outer_folds[[i]],] 
  test <- onehot_encode(test)
  
  RF <- randomForest(is_canceled~., data=train, ntree=500, mtry=20, nodesize=5)
  RF_imp <- RF_imp + importance(RF)
  
  RF.pred <- predict(RF, test, type='response')
  RF_CM <- confusionMatrix(RF.pred, test$is_canceled)
  RF_result[nrow(RF_result)+1, ] = c(RF_CM$overall['Accuracy'], RF_CM$byClass['Precision'],
                                     RF_CM$byClass['Recall'], RF_CM$byClass['F1'])
}
colMeans(RF_result)
write.csv(RF_result, ".\\result\\RF_city_Evaluation.csv", row.names=FALSE)

# Feature importance
RF_imp <- RF_imp/5
RF_imp <- data.frame('MeanDecreaseGini' = RF_imp)
RF_imp <- tibble::rownames_to_column(RF_imp, "Feature") 
RF_imp20 <- top_n(RF_imp, 20, MeanDecreaseGini)
ggplot(RF_imp20, aes(y=reorder(Feature, MeanDecreaseGini), x=MeanDecreaseGini)) +
  geom_bar(stat = "identity") +
  ylab('Features') +
  ggtitle('City Hotel using Random Forest')
write.csv(RF_imp, ".\\result\\RF_city_FI.csv", row.names=FALSE)


# 4. Gradient Boosting ----
# Parameters Tuning
#   - classification threshold 
#   - n.trees = total no. of tree
#   - shrinkage = learning rate
#   - n.minobsinnode = min no. of observations in terminal node
#   - bag.fraction = fraction of training set to be selected to build next tree

## GBM dataset ----
# city_GBM <- city
# city_GBM$is_canceled <- unclass(city_GBM$is_canceled)%%2 

## 4.1 Initial Tree,Tuning n.trees and shrinkage -----
GBM_intit_params <- list(n.trees = 500,
                         shrinkage = c(0.3, 0.4, 0.5),
                         n.minobsinnode = 100,
                         bag.fraction = 0.4,
                         threshold = 0.5)
GBM_tune1 <- gbm_tuning(data = city, 
                        outer_folds = outer_folds, 
                        Ntrees = GBM_intit_params$n.trees, 
                        Shrinkage = GBM_intit_params$shrinkage, 
                        Nminobsinnode = GBM_intit_params$n.minobsinnode,
                        Bagfraction = GBM_intit_params$bag.fraction,
                        Threshold = GBM_intit_params$threshold,
                        param = 'initial')
GBM_tune1 %>% 
  group_by(n.trees, shrinkage) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))
write.csv(GBM_tune1, ".\\result\\GBM_city_ntreelr.csv", row.names=FALSE)


## 4.2 Tuning n.minobsinnode -----
GBM_nmin_params <- list(n.trees = 500,
                        shrinkage = 0.5,
                        n.minobsinnode = c(10, 30),
                        bag.fraction = 0.4,
                        threshold = 0.5)
GBM_tune2 <- gbm_tuning(data = city, 
                       outer_folds = outer_folds, 
                       Ntrees = GBM_nmin_params$n.trees, 
                       Shrinkage = GBM_nmin_params$shrinkage, 
                       Nminobsinnode = GBM_nmin_params$n.minobsinnode,
                       Bagfraction = GBM_nmin_params$bag.fraction,
                       Threshold = GBM_nmin_params$threshold,
                       param = 'n.minobsinnode')
GBM_tune2 %>% 
  group_by(n.minobsinnode) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))
write.csv(GBM_tune2, ".\\result\\GBM_city_nmin2.csv", row.names=FALSE)


## 4.3 Tuning bag.fraction ----
GBM_bagfrac_params <- list(n.trees = 500,
                           shrinkage = 0.5,
                           n.minobsinnode = 50,
                           bag.fraction = c(0.4, 0.6, 0.8, 1),
                           threshold = 0.5)
GBM_tune3 <- gbm_tuning(data = city, 
                       outer_folds = outer_folds, 
                       Ntrees = GBM_bagfrac_params$n.trees, 
                       Shrinkage = GBM_bagfrac_params$shrinkage, 
                       Nminobsinnode = GBM_bagfrac_params$n.minobsinnode,
                       Bagfraction = GBM_bagfrac_params$bag.fraction,
                       Threshold = GBM_bagfrac_params$threshold,
                       param = 'bag.fraction')
GBM_tune3 %>% 
  group_by(bag.fraction) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))
write.csv(GBM_tune3, ".\\result\\GBM_city_bagfrac.csv", row.names=FALSE)


## 4.4 Tuning classify threshold -----
GBM_theshold_params <- list(n.trees = 500,
                            shrinkage = 0.5,
                            n.minobsinnode = 50,
                            bag.fraction = 0.6,
                            threshold = c(0.4, 0.5, 0.6))
GBM_tune4 <- gbm_tuning(data = city, 
                        outer_folds = outer_folds, 
                        Ntrees = GBM_theshold_params$n.trees, 
                        Shrinkage = GBM_theshold_params$shrinkage, 
                        Nminobsinnode = GBM_theshold_params$n.minobsinnode,
                        Bagfraction = GBM_theshold_params$bag.fraction,
                        Threshold = GBM_theshold_params$threshold,
                        param = 'threshold')
GBM_tune4 %>% 
  group_by(threshold) %>% 
  summarise(AVG_accuracy = mean(Accuracy),
            AVG_precision = mean(Precision),
            AVG_recall = mean(Recall),
            AVG_F1 = mean(F1))
write.csv(GBM_tune4, ".\\result\\GBM_city_theshold.csv", row.names=FALSE)


## 4.5 Final GBM ----
GBM_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(GBM_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
for (i in 1:5){
  print(i)
  train <- city[-outer_folds[[i]],] 
  train <- onehot_encode(train)
  train$is_canceled <- unclass(train$is_canceled)%%2 
  test <- city[outer_folds[[i]],] 
  test <- onehot_encode(test)
  test$is_canceled <- unclass(test$is_canceled)%%2 
  
  GBM <- gbm(is_canceled~., data=train, distribution="bernoulli", n.trees=500, 
             shrinkage=0.5, n.minobsinnode=50, bag.fraction=0.6, verbose=FALSE)
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
GBM_result
colMeans(GBM_result)
write.csv(GBM_result, ".\\result\\GBM_city_Evaluation.csv", row.names=FALSE)


# 5. XGBoost ----
# Parameters Tuning
#   - eta = learning rate
#   - nrounds = max no. of boosting iterations
#   - gamma = for avoid overfitting 
#   - max_depth 
#   - min_child_weight
#   - subsample = subsample ratio for growing tree
#   - colsample_bytree = subsample ratio of columns

## XGB dataset ---- 
# city_xgb <- city
# city_xgb$is_canceled <- unclass(city_xgb$is_canceled)%%2
xgb_trainCtr = trainControl(method = "LGOCV", p = 0.7, number = 1, search = "grid")

## 5.1 Initial nrounds and learning rate ----
xgbGrid1 <-  expand.grid(eta = c(0.01, 0.1, 0.2, 0.3, 0.5), 
                         nrounds = 500,
                         # fixed values below
                         max_depth = 5, 
                         min_child_weight = 1,
                         gamma = 0,
                         subsample = 1,
                         colsample_bytree = 1)
XGB_tune1 <- xgb_tuning(data = city, 
                        train_ctr = xgb_trainCtr, 
                        outer_folds = outer_folds, 
                        param_grid = xgbGrid1)
XGB_tune1
write.csv(XGB_tune1, ".\\result\\XGB_city_etanrounds.csv", row.names=FALSE)


## 5.2 Tune max_depth, min_child_weight ----
xgbGrid2 <-  expand.grid(max_depth = c(5, 10, 15),
                         min_child_weight = c(1, 5, 10),
                         # fixed values below
                         eta = 0.2,
                         nrounds = 500,
                         gamma = 0,
                         subsample = 1,
                         colsample_bytree = 1)
XGB_tune2 <- xgb_tuning(data = city, 
                        train_ctr = xgb_trainCtr, 
                        outer_folds = outer_folds, 
                        param_grid = xgbGrid2)
XGB_tune2
write.csv(XGB_tune2, ".\\result\\XGB_cit_depth.csv", row.names=FALSE)


## 5.3 Tune gamma ----
xgbGrid3 <-  expand.grid(gamma = c(0, 0.01, 0.1, 0.2),
                         # fixed values below
                         eta = 0.2,
                         nrounds = 500,
                         max_depth = 10,
                         min_child_weight = 1,
                         subsample = 1,
                         colsample_bytree = 1)
XGB_tune3 <- xgb_tuning(data = city, 
                        train_ctr = xgb_trainCtr, 
                        outer_folds = outer_folds, 
                        param_grid = xgbGrid3)
XGB_tune3
write.csv(XGB_tune3, ".\\result\\XGB_cit_gamma.csv", row.names=FALSE)


## 5.4 Tune subsample and colsample_bytree ----
xgbGrid4 <-  expand.grid(subsample = c(0.7, 0.9, 1),
                         colsample_bytree = c(0.7, 0.9, 1), 
                         # fixed values below
                         gamma = 0.2,
                         eta = 0.2,
                         nrounds = 500,
                         max_depth = 10,
                         min_child_weight = 1)
XGB_tune4 <- xgb_tuning(data = city, 
                        train_ctr = xgb_trainCtr, 
                        outer_folds = outer_folds, 
                        param_grid = xgbGrid4)
XGB_tune4
write.csv(XGB_tune4, ".\\result\\XGB_cit_sample.csv", row.names=FALSE)


## 5.5 Tune classification threshold ----
xgb_params <- list(eta = 0.2, 
                   gamma = 0.2, 
                   max_depth = 10, 
                   min_child_weight = 1, 
                   subsample = 0.9, 
                   colsample_bytree = 1,
                   booster = "gbtree", objective="binary:logistic", eval_metric="error")
xgb_threshold <- data.frame(matrix(ncol=6, nrow=0))
colnames(xgb_threshold) = c('Threshold', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')
for (threshold in seq(0.4, 0.6, 0.1)){
  for (i in 1:5){
    print(paste(threshold, i))
    train <- city[-outer_folds[[i]],]
    train_idx <- createDataPartition(train$is_canceled, p=0.7, list=FALSE)
    train_inner <- train[train_idx,]
    val_inner <- train[-train_idx,]
    train_inner <- onehot_encode(train_inner)
    val_inner <- onehot_encode(val_inner)
    train_inner$is_canceled <- unclass(train_inner$is_canceled)%%2
    val_inner$is_canceled <- unclass(val_inner$is_canceled)%%2
    
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
write.csv(xgb_threshold, ".\\result\\XGB_city_threshold.csv", row.names=FALSE)


## 5.6 Final result ----
xgb_params <- list(eta = 0.2, 
                   gamma = 0.2, 
                   max_depth = 10, 
                   min_child_weight = 1, 
                   subsample = 0.9, 
                   colsample_bytree = 1,
                   booster = "gbtree", objective="binary:logistic", eval_metric="error")
XGB_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(XGB_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
XGB_imp <- matrix(0, nrow = ncol(city_1hot)-1, ncol=0)
XGB_imp <- cbind(XGB_imp, colnames(city_1hot %>% select(-is_canceled)))
colnames(XGB_imp) <- c('Feature')

for (i in 1:5){
  print(i)
  train <- city[-outer_folds[[i]],]
  test <- city[outer_folds[[i]],]
  train <- onehot_encode(train)
  test <- onehot_encode(test)
  train$is_canceled <- unclass(train$is_canceled)%%2
  test$is_canceled <- unclass(test$is_canceled)%%2
  
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
write.csv(XGB_result, ".\\result\\XGB_city_Evaluation.csv", row.names=FALSE)

# Feature importance
XGB_imp[is.na(XGB_imp)] <- 0
XGB_imp$MeansGain <- rowMeans(XGB_imp[2:6])
XGB_imp20 <- top_n(XGB_imp, 20, MeansGain)
ggplot(XGB_imp20, aes(y=reorder(Feature, MeansGain), x=MeansGain)) +
  geom_bar(stat = "identity") +
  xlab('Gain') +
  ylab('Features') +
  ggtitle('City Hotel using XGBoost')
write.csv(XGB_imp, ".\\result\\XGB_city_FI.csv", row.names=FALSE)


# 6. LightGBM ----
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
city_lgbm <- city
city_lgbm$is_canceled <- unclass(city_lgbm$is_canceled)%%2

## 6.1 Tuning classification threshold ----
lgbm_params <- list(objective = 'binary', data_sample_strategy = 'goss', boosting = 'gbdt',
                    num_iterations = 500, learning_rate = 0.1, num_leaves = 250,
                    min_data_in_leaf = 10, feature_fraction = 1,
                    top_rate = 0.2, other_rate = 0.5,
                    lambda_l1 = 0, lambda_l2 = 0.01)
lgbm_threshold <- data.frame(matrix(ncol=6, nrow=0))
colnames(lgbm_threshold) = c('Threshold', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')

for (threshold in seq(0.4, 0.7, 0.1)){
  for (i in 1:5){
    print(paste(threshold, i))
    train <- city_lgbm[-outer_folds[[i]],]
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
write.csv(lgbm_threshold, ".\\result\\LGBM_city_threshold.csv", row.names=FALSE)

## 6.2 Final LGBM with gbdt boosting ----
lgbm_params <- list(objective = 'binary', data_sample_strategy = 'goss', boosting = 'gbdt',
                    num_iterations = 500, learning_rate = 0.1, num_leaves = 250,
                    min_data_in_leaf = 10, feature_fraction = 1,
                    top_rate = 0.2, other_rate = 0.5,
                    lambda_l1 = 0, lambda_l2 = 0.01)
lgbm_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(lgbm_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
LGB_imp <- matrix(0, nrow = ncol(city_lgbm)-1, ncol=0)
LGB_imp <- cbind(LGB_imp, colnames(city_lgbm %>% select(-is_canceled)))
colnames(LGB_imp) <- c('Feature')

for (i in 1:5){
  print(i)
  train <- city_lgbm[-outer_folds[[i]],]
  test <- city_lgbm[outer_folds[[i]],]
  
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
                   all.x = TRUE)
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
write.csv(lgbm_result, ".\\result\\LGBM_city_Evaluation1.csv", row.names=FALSE)

# Feature importance
LGB_imp[is.na(LGB_imp)] <- 0
LGB_imp$MeansGain <- rowMeans(LGB_imp[2:6])
LGB_imp20 <- top_n(LGB_imp, 20, MeansGain)
ggplot(LGB_imp20, aes(y=reorder(Feature, MeansGain), x=MeansGain)) +
  geom_bar(stat = "identity") +
  xlab('Gain') +
  ylab('Features') +
  ggtitle('City Hotel using LightGBM')
write.csv(LGB_imp, ".\\result\\LGBM_city_FI1.csv", row.names=FALSE)

# 8. CatBoost ----
# Tuning parameters
# - depth = Tree Depth 
# - learning_rate
# - iterations = Number of trees
# - l2_leaf_reg = L2 regularization coefficient
# - rsm = The percentage of features to use at each iteration
# - border_count = The number of splits for numerical features

## catBoost dataset ----
# hotel_catb <- preprocess(one_hot = FALSE, feature_select = TRUE)
# city_catb <- hotel_catb[[3]]
# sapply(city_catb, class)
# summary(city_catb$is_canceled)

catb_trainCR <- trainControl(method = 'LGOCV',  p = 0.7, number = 1, search = 'grid')

## 8.1 Initial trees ----
# Tune iterations and learning_rate
catbGrid1 <- expand.grid(iterations = 500,
                         learning_rate = c(0.01, 0.1, 0.2, 0.3),
                         # fixed value
                         depth = 8,
                         l2_leaf_reg = 0.1,
                         rsm = 0.8,
                         border_count = 512)
catB_tune1 <- catb_tuning(city, catb_trainCR, outer_folds, catbGrid1)
catB_tune1
write.csv(catB_tune1, ".\\result\\catB_city_iterlr.csv", row.names=FALSE)


## 8.2 Tune tree depth ----
catbGrid2 <- expand.grid(depth = c(4, 6, 8, 10, 12),
                         # fixed value
                         iterations = 500,
                         learning_rate = 0.1,
                         l2_leaf_reg = 0.1,
                         rsm = 0.8,
                         border_count = 512)
catB_tune2 <- catb_tuning(city, catb_trainCR, outer_folds, catbGrid2)
catB_tune2
write.csv(catB_tune2, ".\\result\\catB_city_depth.csv", row.names=FALSE)


## 8.3 Tune rsm ----
catbGrid3 <- expand.grid(rsm = c(0.6, 0.8, 1),
                         # fixed value
                         iterations = 500,
                         learning_rate = 0.1,
                         depth = 12,
                         l2_leaf_reg = 0.1,
                         border_count = 512)
catB_tune3 <- catb_tuning(city, catb_trainCR, outer_folds, catbGrid3)
catB_tune3
write.csv(catB_tune3, ".\\result\\catB_city_rsm.csv", row.names=FALSE)


## 8.4 Tune L2 reg ----
catbGrid4 <- expand.grid(l2_leaf_reg = c(0, 0.001, 0.01, 0.1),
                         # fixed value
                         iterations = 500,
                         learning_rate = 0.1,
                         depth = 12,
                         rsm = 0.8,
                         border_count = 512)
catB_tune4 <- catb_tuning(city, catb_trainCR, outer_folds, catbGrid4)
catB_tune4
write.csv(catB_tune4, ".\\result\\catB_city_l2.csv", row.names=FALSE)


## 8.5 Tune classification threshold ----
catb_params <- list(iterations = 500, 
                    learning_rate = 0.1,
                    rsm = 0.8,
                    depth = 12,
                    l2_leaf_reg = 0.1,
                    border_count = 512,
                    logging_level = 'Silent')
catb_threshold <- data.frame(matrix(ncol=6, nrow=0))
colnames(catb_threshold) = c('Threshold', 'fold', 'Accuracy', 'Precision', 'Recall', 'F1')
for (threshold in seq(0.4, 0.7, 0.1)){
  for (i in 1:5){
    print(paste(threshold, i))
    train <- city[-outer_folds[[i]],]
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
write.csv(catb_threshold, ".\\result\\catB_city_threshold.csv", row.names=FALSE)

## 8.6 Final CatBoost ----
catb_params <- list(iterations = 500, 
                    learning_rate = 0.1,
                    rsm = 0.8,
                    depth = 12,
                    l2_leaf_reg = 0.1,
                    border_count = 512,
                    logging_level = 'Silent')
catb_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(catb_result) = c('Accuracy', 'Precision', 'Recall', 'F1')
catb_imp <- matrix(0, nrow = ncol(city)-1, ncol=0)
catb_imp <- cbind(catb_imp, colnames(city %>% select(-is_canceled)))
colnames(catb_imp) <- c('Feature')

for (i in 1:5){
  print(i)
  train <- city[-outer_folds[[i]],]
  test <- city[outer_folds[[i]],]
  
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
write.csv(catb_result, ".\\result\\catB_city_Evaluation.csv", row.names=FALSE)

# Feature importance
catb_imp$MeansFI <- rowMeans(catb_imp[2:6])
ggplot(catb_imp, aes(y=reorder(Feature, MeansFI), x=MeansFI)) +
  geom_bar(stat = "identity") +
  xlab('Importance') +
  ylab('Features') +
  ggtitle('City Hotel using CatBoost')
write.csv(catb_imp, ".\\result\\catB_city_FI.csv", row.names=FALSE)
