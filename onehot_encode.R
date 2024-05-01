onehot_encode <- function(df){
  library(dplyr)
  library(caret)
  
  df$is_canceled <- unclass(df$is_canceled)%%2 
  dummy <- dummyVars(" ~ .", data=df)
  df_1hot <- data.frame(predict(dummy, newdata=df))
  df_1hot$is_canceled <- factor(df_1hot$is_canceled, levels=c(1,0))
  
  return(df_1hot)
}