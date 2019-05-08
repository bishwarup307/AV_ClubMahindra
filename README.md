# AV Club Mahindra DataOlympics

This repository contains the 2nd place solution for the [Club Mahindra DataOlympics](https://datahack.analyticsvidhya.com/contest/club-mahindra-dataolympics/) hackathon arranged by AnalyticsVidhya between 3rd May and 5th May, 2019.

Public Leaderboard | Private Leaderboard
------------------ | -------------------
94.9250599824 | 95.8431049355

### Problem Statement
Given around ~300,000 reservations across 32 different hotels/holiday homes of Club Mahindra the objective is to predict the average spend of customers per room per night stay at the hotel against the booking. A wide variety of attributes on the reservation were provided which includes
`booking_date`, `checkin_date`, `checkout_date` etc. Please visit the [competition homepage](https://datahack.analyticsvidhya.com/contest/club-mahindra-dataolympics/) for more information on the problem statement and the dataset.

### Approach
My approach is pretty straightforward which mainly revolves around feature engineering and feature selection. I tried many different combination of features and found the below three feature sets to be most useful for this contest.

1. **Features on `memberid`**
   - there were more than 100,000 unique member ids i.e. unique customers present in the whole dataset. But the train and test dataset didn't share any reservation for the same memberid. To put it in another way - the train and test dataset were split on `memberid`. Hence, it didn't make sense for me to use `memberid` itself as a variable in the model but a variety of different aggregated features on `memberid` proved to be the most important one later on. This also makes sense intuitively, as with customer level features the model could get additional information about the customers past and present behavior and more importantly relate similar customers in one way or the other. 

   - I created a pool of such features which include:
     - total number of reservations by a member
     - total number of reservation by a member on a particular resort
	 - total duration of holiday for each member
	 - average duration of each member for each trip taken
	 - average number of people the member has travelled with in the past (`total_pax`)
	 - reservations in different type of resorts (`resort_type_code`)	
	 - reservations in different type of rooms (`room_type_booked_code`)
	 - reservations in different holiday sessions (`season_holidayed_code`)
	 - reservations in different states (`state_code_resort`)
	 - reservations in different type of product categories (`main_product_code`)
	 - average of booking days in advance to checkin for each booking the member had in the past
	 - etc.

   - There were a significant number of reservations that had exact `booking_date`, `checkin_date` and `checkout_date` for a particular `memberid`. I thought of discarding those as duplicate rows but surpirisngly enough they had different target value `amount_spent_per_room_night_scaled` against them. I wasn't able to assess the quality of data discrepancy there but there seemed to be a good correlation of the target value inside those buckets. Adding a feature representing those buckets helped the score a bit.

   