split <- function(data, train_ratio){
  set.seed(1)
  
  sample <- sample(c(TRUE, FALSE), nrow(data), replace=TRUE, prob=c(train_ratio, 1-train_ratio))
  train  <- data[sample, ]
  test   <- data[!sample, ]
  
  return(list(train, test))
}
