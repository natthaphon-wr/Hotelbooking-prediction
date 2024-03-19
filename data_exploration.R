library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
source('preprocess.R')


# 1. Import data ----
data <- preprocess()
hotel_all <- data[[1]]
resort <- data[[2]]
city <- data[[3]]


# 2. Compare resort, city hotel ----
ggplot(hotel_all) +
  geom_bar(aes(x = customer_type, fill=hotel), position = 'dodge')
ggplot(hotel_all) +
  geom_bar(aes(x = distribution_channel , fill=hotel), position = 'dodge')
ggplot(hotel_all) +
  geom_bar(aes(x = is_repeated_guest , fill=hotel), position = 'dodge')
ggplot(hotel_all) +
  geom_bar(aes(x = market_segment , fill=hotel), position = 'dodge')
ggplot(hotel_all) +
  geom_bar(aes(x = meal , fill=hotel), position = 'dodge')
ggplot(hotel_all) +
  geom_bar(aes(x=total_of_special_requests, fill=hotel), position = 'dodge')

ggplot(hotel_all) +
  geom_boxplot(aes(y=adr, fill=hotel)) +
  ylim(0,500)
ggplot(hotel_all) +
  geom_boxplot(aes(y=previous_cancellation_ratio, fill=hotel))
ggplot(hotel_all) +
  geom_boxplot(aes(y=stays_in_nights, fill=hotel)) +
  ylim(0,10)


# 3 Compare resort, city hotel on is_canceled ----
ggplot(hotel_all) +
  geom_bar(aes(x = customer_type, 
               fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
           position = 'dodge') +
  scale_fill_manual(values=c("#e48646", 
                             "#c85200", 
                             "#6b8ea4", 
                             "#366785")) +
  guides(fill=guide_legend(title="Hotel Type and Cancellation"))
ggplot(hotel_all) +
  geom_bar(aes(x = distribution_channel, 
               fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
           position = 'dodge') +
  scale_fill_manual(values=c("#e48646", 
                             "#c85200", 
                             "#6b8ea4", 
                             "#366785")) +
  guides(fill=guide_legend(title="Hotel Type and Cancellation"))
ggplot(hotel_all) +
  geom_bar(aes(x = is_repeated_guest, 
               fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
           position = 'dodge') +
  scale_fill_manual(values=c("#e48646", 
                             "#c85200", 
                             "#6b8ea4", 
                             "#366785")) +
  guides(fill=guide_legend(title="Hotel Type and Cancellation"))
ggplot(hotel_all) +
  geom_bar(aes(x = market_segment, 
               fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
           position = 'dodge') +
  scale_fill_manual(values=c("#e48646", 
                             "#c85200", 
                             "#6b8ea4", 
                             "#366785")) +
  guides(fill=guide_legend(title="Hotel Type and Cancellation"))
ggplot(hotel_all) +
  geom_bar(aes(x = meal, 
               fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
           position = 'dodge') +
  scale_fill_manual(values=c("#e48646", 
                             "#c85200", 
                             "#6b8ea4", 
                             "#366785")) +
  guides(fill=guide_legend(title="Hotel Type and Cancellation"))
ggplot(hotel_all) +
  geom_bar(aes(x = total_of_special_requests, 
               fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
           position = 'dodge') +
  scale_fill_manual(values=c("#e48646", 
                             "#c85200", 
                             "#6b8ea4", 
                             "#366785")) +
  guides(fill=guide_legend(title="Hotel Type and Cancellation"))

ggplot(hotel_all) +
  geom_boxplot(aes(y=adr, 
               fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
           position = 'dodge') +
  ylim(0,500) +
  scale_fill_manual(values=c("#e48646", 
                             "#c85200", 
                             "#6b8ea4", 
                             "#366785")) +
  guides(fill=guide_legend(title="Hotel Type and Cancellation"))
ggplot(hotel_all) +
  geom_boxplot(aes(y=previous_cancellation_ratio, 
                   fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
               position = 'dodge') +
  scale_fill_manual(values=c("#e48646", 
                             "#c85200", 
                             "#6b8ea4", 
                             "#366785")) +
  guides(fill=guide_legend(title="Hotel Type and Cancellation"))
ggplot(hotel_all) +
  geom_boxplot(aes(y=stays_in_nights, 
                   fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
               position = 'dodge') +
  ylim(0,10) +
  scale_fill_manual(values=c("#e48646", 
                             "#c85200", 
                             "#6b8ea4", 
                             "#366785")) +
  guides(fill=guide_legend(title="Hotel Type and Cancellation"))
