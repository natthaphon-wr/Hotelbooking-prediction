source('preprocess.R')
source('onehot_encode.R')
library(tidyverse) 
library(caret)
library(themis)

# Import data ----
data_not1hot <- preprocess(one_hot = FALSE, feature_select = TRUE)
resort <- data_not1hot[[2]]
sapply(resort, class)
summary(resort$is_canceled)

# SMOTE-NC sampling ----
resort_balance <- smotenc(resort, var = 'is_canceled', k = 5, over_ratio = 1)
summary(resort_balance$is_canceled)
sapply(resort_balance, class)

# 1 hot encoding ----
resort_balance$is_canceled <- unclass(resort_balance$is_canceled)%%2 
dummy <- dummyVars(" ~ .", data=resort_balance)
resort_balance_1hot <- data.frame(predict(dummy, newdata=resort_balance))
resort_balance_1hot$is_canceled <- factor(resort_balance_1hot$is_canceled, levels=c(1,0))
summary(resort_balance_1hot$is_canceled)
sapply(resort_balance_1hot, class)


# compare with previous ----
hotel_data <- preprocess(one_hot = TRUE, feature_select = TRUE)
resort2 <- hotel_data[[2]]
sapply(resort2, class)
summary(resort2$is_canceled)

# Test onehot_encode function ----
data <- preprocess(one_hot = FALSE, feature_select = TRUE)
resort <- data[[2]]
sapply(resort, class)
summary(resort$is_canceled)

resort_1hot <- onehot_encode(resort)
sapply(resort_1hot, class)
summary(resort_1hot$is_canceled)

