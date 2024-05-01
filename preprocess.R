preprocess <- function(one_hot = TRUE, feature_select = TRUE){
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
           stays_in_nights = stays_in_week_nights + stays_in_weekend_nights,
           is_repeated_guest = factor(is_repeated_guest)) %>% 
    na.omit() %>% 
    mutate_if(is.character, as.factor)
  
  if (feature_select){
    hotel_all <- hotel_all %>% 
      select(hotel, is_canceled, adr, adults, arrival_date_week_number, babies,
             booking_changes, children, customer_type, days_in_waiting_list,
             distribution_channel, is_repeated_guest, lead_time, market_segment,
             meal, previous_cancellation_ratio, required_car_parking_spaces,
             stays_in_nights, total_of_special_requests) 
  }
    
  if (one_hot){
    dummy <- dummyVars(" ~ .", data=hotel_all)
    hotel_all <- data.frame(predict(dummy, newdata=hotel_all))
    hotel_all <- hotel_all %>% mutate(is_canceled = factor(is_canceled, levels=c(1,0)))
    resort <- hotel_all %>% 
      filter(hotel.Resort.Hotel == 1) %>% 
      select(-hotel.City.Hotel, -hotel.Resort.Hotel)
    city <- hotel_all %>% 
      filter(hotel.City.Hotel == 1) %>% 
      select(-hotel.City.Hotel, -hotel.Resort.Hotel)
  }else{
    hotel_all <- hotel_all %>% mutate(is_canceled = factor(is_canceled, levels=c(1,0)))
    resort <- hotel_all %>% 
      filter(hotel == 'Resort Hotel') %>% 
      select(-hotel)
    city <- hotel_all %>% 
      filter(hotel == 'City Hotel') %>% 
      select(-hotel)
  }

  return(list(hotel_all, resort, city))
}