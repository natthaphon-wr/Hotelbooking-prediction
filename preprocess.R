preprocess <- function(){
  library(readr)
  library(dplyr)
  library(caret)
  
  hotel_all <- read_csv("hotel_bookings.csv")
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
  # hotel_all$is_canceled <- as.factor(hotel_all$is_canceled)
  hotel_all$is_repeated_guest <- as.factor(hotel_all$is_repeated_guest)
  
  # sum(is.na(hotel_all)) 
  # summary(hotel_all)
  # sapply(hotel_all, class)
  
  dummy <- dummyVars(" ~ .", data=hotel_all)
  hotel_1hot <- data.frame(predict(dummy, newdata=hotel_all))
  # sapply(hotel_1hot, class)
  
  resort <- hotel_1hot %>% 
    filter(hotel.Resort.Hotel == 1) %>% 
    select(-hotel.City.Hotel, -hotel.Resort.Hotel)
  city <- hotel_1hot %>% 
    filter(hotel.City.Hotel == 1) %>% 
    select(-hotel.City.Hotel, -hotel.Resort.Hotel)
  
  return(list(hotel_all, resort, city))
}
