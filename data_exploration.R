library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)

# 1. Import data ----
hotel_all <- read_csv("hotel_bookings.csv")


# 2. Preprocess ----
hotel_all <- hotel_all %>% 
  mutate(previous_cancellation_ratio = case_when(previous_cancellations+previous_bookings_not_canceled==0 ~ 0,
                                                 previous_cancellations+previous_bookings_not_canceled>=0 ~ 
                                                   previous_cancellations/(previous_cancellations+
                                                                             previous_bookings_not_canceled)
                                                 ),
         stays_in_nights = stays_in_week_nights + stays_in_weekend_nights) %>% 
  select(hotel, is_canceled, adr, adults, babies, booking_changes, children, customer_type, days_in_waiting_list,
         distribution_channel, is_repeated_guest, market_segment, meal, previous_cancellation_ratio,
         stays_in_nights, total_of_special_requests) %>% 
  replace(is.na(.), 0) %>% 
  mutate_if(is.character, as.factor) 
hotel_all$is_canceled <- as.factor(hotel_all$is_canceled)
hotel_all$is_repeated_guest <- as.factor(hotel_all$is_repeated_guest)

sum(is.na(hotel_all)) 
summary(hotel_all)
sapply(hotel_all, class)


# 3. Compare resort, city hotel ----
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



# 4. Compare resort, city hotel on is_canceled ----
ggplot(hotel_all) +
  geom_bar(aes(x = customer_type, 
               fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
           position = 'dodge')
ggplot(hotel_all) +
  geom_bar(aes(x = distribution_channel, 
               fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
           position = 'dodge')
ggplot(hotel_all) +
  geom_bar(aes(x = is_repeated_guest, 
               fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
           position = 'dodge')
ggplot(hotel_all) +
  geom_bar(aes(x = market_segment, 
               fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
           position = 'dodge')
ggplot(hotel_all) +
  geom_bar(aes(x = meal, 
               fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
           position = 'dodge')+
ggplot(hotel_all) +
  geom_bar(aes(x = total_of_special_requests, 
               fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
           position = 'dodge')

ggplot(hotel_all) +
  geom_boxplot(aes(y=adr, 
               fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
           position = 'dodge') +
  ylim(0,500)
ggplot(hotel_all) +
  geom_boxplot(aes(y=previous_cancellation_ratio, 
                   fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
               position = 'dodge')
ggplot(hotel_all) +
  geom_boxplot(aes(y=stays_in_nights, 
                   fill=interaction(hotel, is_canceled, sep="-", lex.order=TRUE)), 
               position = 'dodge') +
  ylim(0,10)
