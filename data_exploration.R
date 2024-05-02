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
ggplot(hotels_all) +
  geom_histogram(aes(x = lead_time, fill = is_canceled), binwidth = 50, position = 'dodge') 
ggplot(resort_all) +
  geom_histogram(aes(x = lead_time, fill = is_canceled), binwidth = 50, position = 'dodge') 
ggplot(city_all) +
  geom_histogram(aes(x = lead_time, fill = is_canceled), binwidth = 50, position = 'dodge') 
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

## City ----
city$is_canceled <- as.numeric(city$is_canceled)
city_imp <- city %>% 
  select(is_canceled, 
         adr, 
         adults,
         booking_changes,
         children,
         customer_type.Transient,
         customer_type.Transient.Party,
         distribution_channel.TA.TO,
         market_segment.Groups,
         market_segment.Online.TA,
         market_segment.Offline.TA.TO,
         previous_cancellation_ratio,
         stays_in_nights,
         total_of_special_requests)

city_cor <- melt(round(cor(city_imp, method = "spearman"),2))
ggplot(data = city_cor, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Spearman Correlation") +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust = 0.5)) +
  labs(title = "Correlation Matrix in City Hotel",
       x = "Features",
       y = "Features")
write.csv(city_cor, '.\\city_cor.csv', row.names=FALSE)
  
## Resort ----
resort$is_canceled <- as.numeric(resort$is_canceled)
resort_imp <- resort %>% 
  select(is_canceled, 
         adr, 
         adults,
         booking_changes,
         children,
         customer_type.Transient,
         customer_type.Transient.Party,
         distribution_channel.TA.TO,
         market_segment.Groups,
         market_segment.Online.TA,
         market_segment.Offline.TA.TO,
         previous_cancellation_ratio,
         stays_in_nights,
         total_of_special_requests)
resort_cor <- melt(round(cor(resort_imp, method = "spearman"),2))
ggplot(data = resort_cor, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Spearman Correlation") +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust = 0.5)) +
  labs(title = "Correlation Matrix in Resort Hotel",
       x = "Features",
       y = "Features")
write.csv(resort_cor, '.\\resort_cor.csv', row.names=FALSE)

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
