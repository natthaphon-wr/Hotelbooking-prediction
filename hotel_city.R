source('preprocess.R')
source('split.R')
library(rpart)
library(rpart.plot)
library(RColorBrewer)

# 1. Import data ----
hotel_data <- preprocess()
city <- hotel_data[[3]]
city_split <- split(city, 0.9)
train <- city_split[[1]]
test <- city_split[[2]]

summary(train$is_canceled)
summary(test$is_canceled)

# 2. Decision Tree ----
tree <- rpart(is_canceled ~ ., data = train, method = "class")
summary(tree)
rpart.plot(tree)

princp(tree)
plotcp(tree)

predict(tree, test)