# Import library ----
library(ggplot2)
library(tidyr)
library(dplyr)
library(readr)

# Resort Hotel ----
## RF ----
resort_RF <- read_csv(".\\result\\V2\\Resort\\RF_resort_FI.csv")
resort_RF20 <- top_n(resort_RF, 20, MeanDecreaseGini)
ggplot(resort_RF20, aes(y=reorder(Feature, MeanDecreaseGini), x=MeanDecreaseGini)) +
  geom_bar(stat = "identity") +
  ylab('Features')

## XGB ----
resort_XGB <- read_csv(".\\result\\V2\\Resort\\XGB_resort_FI.csv")
resort_XGB20 <- top_n(resort_XGB, 20, MeansGain)
ggplot(resort_XGB20, aes(y=reorder(Feature, MeansGain), x=MeansGain)) +
  geom_bar(stat = "identity") +
  xlab('Gain') +
  ylab('Features')

## LGB ----
resort_LGBM <- read_csv(".\\result\\V2\\Resort\\LGBM_resort_FI.csv")
ggplot(resort_LGBM, aes(y=reorder(Feature, MeansGain), x=MeansGain)) +
  geom_bar(stat = "identity") +
  xlab('Gain') +
  ylab('Features')

## CatBoost ----
resort_catB <- read_csv(".\\result\\V2\\Resort\\catB_resort_FI.csv")
ggplot(resort_catB, aes(y=reorder(Feature, MeansFI), x=MeansFI)) +
  geom_bar(stat = "identity") +
  xlab('Importance') +
  ylab('Features')

# City Hotel ----
## RF ----
city_RF <- read_csv(".\\result\\V2\\City\\RF_city_FI.csv")
city_RF20 <- top_n(city_RF, 20, MeanDecreaseGini)
ggplot(city_RF20, aes(y=reorder(Feature, MeanDecreaseGini), x=MeanDecreaseGini)) +
  geom_bar(stat = "identity") +
  ylab('Features')

## XGB ----
city_XGB <- read_csv(".\\result\\V2\\City\\XGB_city_FI.csv")
city_XGB20 <- top_n(city_XGB, 20, MeansGain)
ggplot(city_XGB20, aes(y=reorder(Feature, MeansGain), x=MeansGain)) +
  geom_bar(stat = "identity") +
  xlab('Gain') +
  ylab('Features')

## LGB ----
city_LGBM <- read_csv(".\\result\\V2\\City\\LGBM_city_FI1.csv")
ggplot(city_LGBM, aes(y=reorder(Feature, MeansGain), x=MeansGain)) +
  geom_bar(stat = "identity") +
  xlab('Gain') +
  ylab('Features')

## CatBoost ----
city_catB <- read_csv(".\\result\\V2\\City\\catB_city_FI.csv")
ggplot(city_catB, aes(y=reorder(Feature, MeansFI), x=MeansFI)) +
  geom_bar(stat = "identity") +
  xlab('Importance') +
  ylab('Features')


