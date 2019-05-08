# AV Club Mahindra DataOlympics

This repository contains the 2nd place solution for the [Club Mahindra DataOlympics](https://datahack.analyticsvidhya.com/contest/club-mahindra-dataolympics/) hackathon arranged by AnalyticsVidhya between 3rd May and 5th May, 2019.

Public Leaderboard | Private Leaderboard
------------------ | -------------------
94.9250599824 | 95.8431049355

## Problem Statement
Given around ~300,000 reservations across 32 different hotels/holiday homes of Club Mahindra the objective is to predict the average spend of customers per room per night stay at the hotel against the booking. A wide variety of attributes on the reservation were provided which includes
`booking_date`, `checkin_date`, `checkout_date` etc. Please visit the [competition homepage](https://datahack.analyticsvidhya.com/contest/club-mahindra-dataolympics/) for more information on the problem statement and the dataset.

## Approach

#### Feature Engineering
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

   - There were a significant number of reservations that had exact `booking_date`, `checkin_date` and `checkout_date` for a particular `memberid`. I thought of discarding those as duplicate rows but surpirisngly enough they had different target values `amount_spent_per_room_night_scaled` against them. I wasn't able to assess the quality of data discrepancy there but there seemed to be a good correlation of the target value inside those buckets. Adding a feature representing those buckets helped the score a bit.

2. **Temoral Features**
   - These are the second most imporant feature in my pool of features. Temporal features almost always helps boosted trees as most of the time these models can leverage the cross-sectional correlation of the data (e.g. interaction between different features at a given observation level) but there is no way for the model to tackle the time dimension (e.g. latent interaction between two or more observations recorded at different time points). by infusing these featues explicitly - the model can also learn the cross-time correlation e.g. how booking of member in the past affects the nature of booking of a member at present. This is very important.

   - The temporal features that I considered are:
     - The sequential booking number of a member
     - Days since last checkin
     - The sequential booking number of a member in a particular resort
     - Days since last checkin in a particular resort
     - Did you see a lot of booking happens on the same day by a member? It's normal in that when we plan a trip we tend to book the whole trip and a bunch of different hotels at the same time. A feature around that helped the model to learn if it's a continued trip and perhaps the spend in different hotels will be correlated in that trip.
     - That said, there were instances where the booking were done in place something like an extended stay. A feature around that also helped to dertermine the nature of spend.

3. **`resort_id` features**
   - The last main feature set was about aggregating different attributes on `resort` level e.g. total booking in the resort in different room category and in different holiday sessions etc. This was not as important as the above two and I also didn't do a good ablation study on it but it helped diversifying the feature set nonetheless.

4. **Ratio features**
  - I created a number of ratios between:
    - duration of a stay by number of days the booking was done in advance
    - number of children to number of adults travelling
    - number of adults to number of roomnights
    - etc.

#### Modeling
I had 3 models in total:
1. LightGBM with all features bagged 10 times
2. LightGBM with all features but `resort_id` features bagged 10 times
3. XgBoost with all features but `resort_id` features bagged 6 times

The first model itself was good for the 2nd place in public leaderborad but bagging helped to make the predictions a bit more robust. My final submission is a weighted average of the above 3 models.

## How to run the soultion
The above code is tested on python >= 3.6

1. Download the code or clone the repo using `git clone`
2. Download the dataset from the competition website and put it inside a folder called `input` in the main directory. (In case you put the dataset somewhere else you can modify the path in the `config.py` file.)
3. Create a directory called `submissions` in the main directory.
4. In a terminal window `cd` to the git directory and the `cd` to `src`
5. run `python train.py`

