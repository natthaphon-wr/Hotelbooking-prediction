library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(stats)
library(reshape2)
library(heatmaply)
source('preprocess.R')


# Import data ----
data_1hot <- preprocess(one_hot = TRUE)
hotel_1hot <- data_1hot[[1]]
resort_1hot <- data_1hot[[2]]
city_1hot <- data_1hot[[3]]
summary(hotel_1hot)

data_cat <- preprocess(one_hot = FALSE)
hotel_cat <- data_cat[[1]]
resort_cat <- data_cat[[2]]
city_cat <- data_cat[[3]]
summary(hotel_cat)


# Plot all features ----
data_all <- preprocess(one_hot = FALSE, feature_select = FALSE)
hotels_all <- data_all[[1]]
resort_all <- data_all[[2]]
city_all <- data_all[[3]]
summary(hotels_all)

## Categorical data ----
### deposit type ----
ggplot(hotels_all) +
  geom_bar(aes(x = deposit_type, fill = is_canceled), 
           position = 'stack') +
  scale_fill_manual(values = c('#F2543D', '#38C477')) +
  theme(axis.text = element_text(size=10), 
        axis.title = element_text(size=14),
        legend.title = element_text(size=14),
        legend.text = element_text(size=10))

### arrival data ----
ggplot(hotels_all) +
  geom_bar(aes(x = arrival_date_year))

ggplot(hotels_all) +
  geom_bar(aes(x = arrival_date_month, fill = is_canceled), position = 'stack') 
ggplot(hotels_all) +
  geom_bar(aes(x = arrival_date_month, fill = hotel), position = 'dodge')


### customer_type ----
summary(hotels_all$customer_type)
ggplot(hotels_all) +
  geom_bar(aes(x = customer_type, fill = is_canceled))

### market_segment ----
summary(hotels_all$market_segment)
ggplot(hotels_all) +
  geom_bar(aes(x = market_segment, fill = is_canceled))

### distribution_channel ----
summary(hotels_all$distribution_channel)
ggplot(hotels_all) +
  geom_bar(aes(x = distribution_channel, fill = is_canceled))

### meal ----
summary(hotels_all$meal)
ggplot(hotels_all) +
  geom_bar(aes(x = meal, fill = is_canceled))



## Numerical data ----
ggplot(hotel_cat) +
  geom_histogram(aes(x = lead_time, fill = is_canceled), binwidth = 50, position = 'stack') 
ggplot(hotel_cat) +
  geom_histogram(aes(x = adr, fill = is_canceled), binwidth = 50, position = 'stack') +
  xlim(0, 500)
ggplot(hotel_cat) +
  geom_histogram(aes(x = arrival_date_week_number, fill = is_canceled), binwidth = 5, position = 'stack') 

ggplot(hotels_all) +
  geom_histogram(aes(x = lead_time, fill = is_canceled))
ggplot(resort_all) +
  geom_boxplot(aes(x = lead_time, y = is_canceled))

ggplot(hotels_all) +
  geom_bar(aes(x = arrival_date_week_number))
ggplot(hotels_all) +
  geom_bar(aes(x = arrival_date_week_number, fill = is_canceled), position = 'stack') 
ggplot(hotels_all) +
  geom_bar(aes(x = arrival_date_week_number, fill = hotel), position = 'dodge')

ggplot(hotels_all) +
  geom_boxplot(aes(y = stays_in_week_nights, x = hotel))
ggplot(hotels_all) +
  geom_boxplot(aes(y = stays_in_weekend_nights, x = hotel))
ggplot(hotels_all) +
  geom_boxplot(aes(y = stays_in_nights, x = hotel))
ggplot(resort_all) +
  geom_boxplot(aes(y = stays_in_week_nights, x = is_canceled))
ggplot(city_all) +
  geom_boxplot(aes(y = stays_in_weekend_nights, x = is_canceled))

ggplot(hotels_all) +
  geom_histogram(aes(x = days_in_waiting_list))

ggplot(hotels_all) +
  geom_histogram(aes(x = adr))

ggplot(hotels_all) +
  geom_bar(aes(x = required_car_parking_spaces, fill = is_canceled))
ggplot(resort_all) +
  geom_bar(aes(x = required_car_parking_spaces, fill = is_canceled))
ggplot(city_all) +
  geom_bar(aes(x = required_car_parking_spaces, fill = is_canceled))

ggplot(hotels_all) +
  geom_bar(aes(x = total_of_special_requests, fill = is_canceled))
ggplot(resort_all) +
  geom_bar(aes(x = total_of_special_requests, fill = is_canceled))
ggplot(city_all) +
  geom_bar(aes(x = total_of_special_requests, fill = is_canceled))

# Correlation b/w features ----
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}

## Both hotels ----
hotel_cat$is_canceled <- as.numeric(hotel_cat$is_canceled)%%2
hotel_imp <- hotel_cat %>% 
  select(is_canceled, 
         lead_time,
         adr, 
         arrival_week_number = arrival_date_week_number,
         prev_cancel_rate = previous_cancellation_ratio,
         special_requests = total_of_special_requests,
         require_parking_spaces = required_car_parking_spaces)
hotel_cormat <- round(cor(hotel_imp, method = "spearman"),2)
hotel_cor <- melt(get_lower_tri(hotel_cormat), na.rm = TRUE)
# write.csv(city_cor, '.\\hotels_cor.csv', row.names=FALSE)
ggplot(data = hotel_cor, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile() +
  geom_text(aes(label=value)) +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Spearman\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust = 0.5)) +
  coord_fixed() +
  theme(axis.text = element_text(size=10), 
        axis.title = element_text(size=12),
        legend.title = element_text(size=12),
        legend.text = element_text(size=10)) +
  labs(x = "Features",
       y = "Features") 

## City ----
city_cat$is_canceled <- as.numeric(city_cat$is_canceled)%%2
city_imp <- city_cat %>% 
  select(is_canceled, 
         lead_time,
         adr, 
         arrival_week_number = arrival_date_week_number,
         prev_cancel_rate = previous_cancellation_ratio,
         special_requests = total_of_special_requests,
         require_parking_spaces = required_car_parking_spaces)
city_cormat <- round(cor(city_imp, method = "spearman"),2)
city_cor <- melt(get_lower_tri(city_cormat), na.rm = TRUE)
# write.csv(city_cor, '.\\city_cor.csv', row.names=FALSE)
ggplot(data = city_cor, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile() +
  geom_text(aes(label=value)) +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Spearman\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust = 0.5)) +
  coord_fixed() +
  theme(axis.text = element_text(size=10), 
        axis.title = element_text(size=12),
        legend.title = element_text(size=12),
        legend.text = element_text(size=10)) +
  labs(x = "Features",
       y = "Features") 
  
## Resort ----
resort_cat$is_canceled <- as.numeric(resort_cat$is_canceled)%%2
resort_imp <- resort_cat %>% 
  select(is_canceled, 
         lead_time,
         adr, 
         arrival_week_number = arrival_date_week_number,
         prev_cancel_rate = previous_cancellation_ratio,
         special_requests = total_of_special_requests,
         require_parking_spaces = required_car_parking_spaces)
resort_cormat <- round(cor(resort_imp, method = "spearman"),2)
resort_cor <- melt(get_lower_tri(resort_cormat), na.rm = TRUE)
# write.csv(resort_cor, '.\\resort_cor.csv', row.names=FALSE)
ggplot(data = resort_cor, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile() +
  geom_text(aes(label=value)) +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Spearman\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust = 0.5)) +
  coord_fixed() +
  theme(axis.text = element_text(size=10), 
        axis.title = element_text(size=12),
        legend.title = element_text(size=12),
        legend.text = element_text(size=10)) +
  labs(x = "Features",
       y = "Features")


# Compare resort, city hotel ----
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


# Compare resort, city hotel on is_canceled ----
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
