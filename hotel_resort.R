source('preprocess.R')
library(tree)
library(tidyverse) 


# 1. Import data ----
hotel_data <- preprocess()
resort <- hotel_data[[2]]
sapply(resort, class)
summary(resort$is_canceled)


# 2. Do classification ----
set.seed(1)
folds <- createFolds(resort$is_canceled, k=10)
DT_result <- data.frame(matrix(ncol=4, nrow=0))
colnames(DT_result) = c('Accuracy', 'Precision', 'Recall', 'F1')

for (i in 1:10){
  train <- resort[-folds[[i]],] 
  test <- resort[folds[[i]],] 
  
  ## 2.1 Decision Tree ----
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
  CM <- confusionMatrix(tree.pred, test$is_canceled)
  DT_result[nrow(DT_result)+1, ] = c(CM$overall['Accuracy'], CM$byClass['Precision'],
                                     CM$byClass['Recall'], CM$byClass['F1'])
}


# 3. Conclude results ----
colMeans(DT_result)
summary(prune.trees)$used