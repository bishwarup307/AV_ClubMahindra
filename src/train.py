# -*- coding: utf-8 -*-
"""
__author__: bishwarup
"""
from __future__ import division, print_function
import os
import re
import random
import time
from pprint import pprint
import argparse
from tqdm import tqdm, tqdm_notebook
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb
from config import config

def date_to_integer(dt_time):
    return 10_000 * dt_time.year + 100 * dt_time.month + dt_time.day

def get_dual_features(df, feature_tuple, op = "ratio"):
    if op not in ("ratio", "difference"):
        raise ValueError("`op` must be one of `ratio` or `difference`")
    print(f"getting {op} features...")
    feature_dict = {}
    for pair in tqdm(feature_tuple, ncols = 80):
        f1, f2 = pair
        feature_name = f"ratio_{f1}_{f2}"
        if np.logical_or(f1 not in df.columns, f2 not in df.columns):
            raise ValueError(f"either of {f1}/{f2} not found...")
        ratio = df[f1] / df[f2] if op == 'ratio' else df[f1] - df[f2]
        feature_dict[feature_name] = ratio.values
    return pd.DataFrame(feature_dict)

def get_agg_features(df, feature_tuple):
    print("getting aggregated features...")
    feature_dict = {}
    for comb in tqdm(feature_tuple, ncols = 80):
        op = comb[2].__name__ if hasattr(comb[2], '__call__') else str(comb[2])
        feature_name = f"{op}_{comb[1]}_by_{'__'.join(comb[0])}"
        feature = df.groupby(comb[0])[comb[1]].transform(comb[2])
        feature_dict[feature_name] = feature.values
    return pd.DataFrame(feature_dict)

def nunique(x):
    return len(set(x))

def print_runtime(start_time):
    end_time = time.time()
    seconds_elapsed = end_time - start_time
    hours, rest = divmod(seconds_elapsed, 3600)
    minutes, seconds = divmod(rest, 60)
    print(f"done! elapsed {int(hours)}h {int(minutes)}m {seconds:.2f}s...")

def run_LGB(params, train, test, feature_names, n_folds = 10, seed = 0):
    skf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X, y = train[feature_names], train.amount_spent_per_room_night_scaled.values
    
    preds = np.zeros(test.shape[0])
    
    for i, (itr, icv) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[itr, :], X.iloc[icv, :]
        y_train, y_val = y[itr], y[icv]

        dtrain = lgb.Dataset(X_train, y_train)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain)

        bst = lgb.train(
                params                = params,
                train_set             = dtrain, 
                valid_sets            = [dtrain, dval],
                valid_names           = ['train', 'eval'],
                num_boost_round       = 30_000,
                verbose_eval          = 500,
                early_stopping_rounds = 200
            )

        score_, iter_ = bst.best_score['eval']['rmse'], bst.best_iteration
        test_preds = bst.predict(test[feature_names], num_iteration = iter_)
        preds += test_preds
    return preds / n_folds

def run_XGB(params, train, test, feature_names, n_folds = 10, seed = 0):
    skf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X, y = train[feature_names], train.amount_spent_per_room_night_scaled.values
    
    preds = np.zeros(test.shape[0])
    
    for i, (itr, icv) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[itr, :], X.iloc[icv, :]
        y_train, y_val = y[itr], y[icv]

        dtrain = xgb.DMatrix(data=X_train, label=y_train, missing=np.nan)
        dval = xgb.DMatrix(data = X_val, label=y_val, missing=np.nan)

        bst = xgb.train(
                params                = params,
                dtrain                = dtrain,
                num_boost_round       = 30_000,
                early_stopping_rounds = 200,
                evals                 = [(dtrain, 'train'), (dval, 'eval')],
                verbose_eval          = 500
            )

        score_, iter_ = bst.best_score, bst.best_iteration
        test_preds = bst.predict(xgb.DMatrix(test[feature_names]), num_iteration = iter_)
        preds += test_preds
    return preds / n_folds

def load_data(input_dir):
    print("reading data files...")
    train = pd.read_csv(os.path.join(input_dir, "train.csv"), parse_dates = ['booking_date', 'checkin_date', 'checkout_date'], 
                        dayfirst = True)
    test = pd.read_csv(os.path.join(input_dir, "test.csv"), parse_dates = ['booking_date', 'checkin_date', 'checkout_date'], 
                      dayfirst = True)
    print(f'train: {train.shape}')
    print(f'test: {test.shape}')

    test['amount_spent_per_room_night_scaled'] = -1
    D = pd.concat([train, test], ignore_index = True)
    del train, test

    print("extracting date features...")
    for date_col in ['booking_date', 'checkin_date', 'checkout_date']:
        D[f"{date_col}_year"] = D[date_col].dt.year
        D[f"{date_col}_mon"] = D[date_col].dt.month
        D[f"{date_col}_day"] = D[date_col].dt.day
        D[f"{date_col}_wday"] = D[date_col].dt.weekday

        D['booking_in_advance'] = (D['checkin_date'] - D['booking_date']).dt.days
        D['days_stayed'] = (D['checkout_date'] - D['checkin_date']).dt.days

        D.season_holidayed_code.fillna(-1, inplace = True)
        D.state_code_residence.fillna(-1, inplace = True)
        D['season_holidayed_code'] = D['season_holidayed_code'].astype(np.uint8)
        D['state_code_residence'] = D['state_code_residence'].astype(np.int32)

        D['n_people'] = D['numberofadults'] + D['numberofchildren']

    ratio_features_list = [
            ('numberofchildren', 'numberofadults'),
            ('numberofadults', 'total_pax'),
            ('days_stayed', 'booking_in_advance'),
            ('days_stayed', 'numberofadults'),
            ('numberofchildren', 'days_stayed'),
            ('roomnights', 'total_pax'),
            ('roomnights', 'numberofadults'),
            ('roomnights', 'days_stayed'),
            ('roomnights', 'days_stayed'),
            ('total_pax', 'days_stayed'),
            ('n_people', 'roomnights'),
            ('n_people', 'total_pax'),
            ('n_people', 'booking_in_advance'),
        ]

    agg_features_list = [
            (['memberid', 'resort_id'], 'reservation_id', 'size'),
            (['memberid'], 'reservation_id', 'size'),
            (['memberid'], 'days_stayed', 'sum'),
            (['memberid'], 'days_stayed', 'mean'),
            (['memberid'], 'season_holidayed_code', 'mean'),
            (['memberid'], 'resort_id', nunique),
            (['memberid'], 'room_type_booked_code', nunique),
            (['memberid'], 'state_code_resort', nunique),
            (['memberid'], 'roomnights', "mean"),
            (['memberid'], 'numberofadults', 'mean'),
            (['memberid'], 'numberofchildren', 'mean'),
            (['memberid'], 'total_pax', 'mean'),
            (['memberid'], 'booking_in_advance', 'mean'),
            (['memberid', 'room_type_booked_code'], 'reservation_id', 'size'),
            (['memberid', 'resort_type_code'], 'reservation_id', 'size'),
            (['memberid', 'season_holidayed_code'], 'reservation_id', 'size'),
            (['memberid', 'resort_type_code'], 'days_stayed', 'mean'),
            (['memberid', 'persontravellingid'], 'reservation_id', 'size'),
            (['memberid', 'booking_date'], 'reservation_id', 'size'),
            (['memberid', 'main_product_code'], 'reservation_id', 'size'),
            (['memberid', 'cluster_code'], 'reservation_id', 'size'),
            (['memberid', 'state_code_resort'], 'reservation_id', 'size'),
            (['memberid', 'resort_type_code'], 'roomnights', 'mean'),
            (['memberid', 'season_holidayed_code'], 'days_stayed', 'mean'),
            (['memberid', 'resort_region_code'], 'reservation_id', 'size'),
            (['memberid', 'resort_type_code', 'room_type_booked_code'], 'reservation_id', 'size'),
            (['memberid', 'resort_type_code', 'room_type_booked_code'], 'days_stayed', 'mean'),

            (['memberid', 'booking_date', 'checkin_date', 'checkout_date'], 'reservation_id', 'size'),

            (['resort_id', 'season_holidayed_code'], 'reservation_id', 'size'),
            (['resort_id', 'checkin_date_mon'], 'reservation_id', 'size'),
            (['resort_id'], 'reservation_id', 'size'),
            (['resort_id', 'season_holidayed_code'], 'reservation_id', 'size'),
            (['resort_id', 'room_type_booked_code'], 'reservation_id', 'size'),
            (['resort_id', 'checkin_date_mon', 'room_type_booked_code'], 'reservation_id', 'size'),
            (['resort_id', 'main_product_code'], 'reservation_id', 'size'),
            (['resort_id', 'main_product_code', 'channel_code'], 'reservation_id', 'size'),
            (['resort_id', 'main_product_code'], 'roomnights', 'mean'),
            (['resort_id', 'season_holidayed_code'], 'days_stayed', 'sum'),
            (['resort_id', 'state_code_residence'], 'reservation_id', 'size'),
            (['resort_id', 'state_code_residence', 'season_holidayed_code'], 'reservation_id', 'size'),
            (['resort_id', 'checkin_date'], 'reservation_id', 'size'),
            (['resort_id'], 'total_pax', 'mean'),
            (['resort_id'], 'numberofchildren', 'mean'),
            (['resort_id', 'member_age_buckets'], 'reservation_id', 'size'),
            (['resort_id', 'checkin_date_year', 'checkin_date_mon'], 'reservation_id', 'size'),
        ]

    print("getting ratio/aggregated features...(this may a minute or two)")
    ratio_features = get_dual_features(D, ratio_features_list)
    agg_features = get_agg_features(D, agg_features_list)
    print(f'ration features: {ratio_features.shape}')
    print(f'aggregate features: {agg_features.shape}')
    D = pd.concat([D, ratio_features, agg_features], axis = 1)
    del agg_features, ratio_features, ratio_features_list, agg_features_list
    
    print("getting temporal features...")
    D.sort_values(["memberid", "checkin_date"], inplace = True, ascending=True)
    D["booking_idx_memberid"] = D.groupby("memberid")["reservation_id"].cumcount() + 1
    D['days_since_last_checkin'] = D.groupby('memberid')['checkin_date'].diff().apply(lambda x: x.days)

    D.sort_values(["memberid", "resort_id", "checkin_date"], inplace = True, ascending=True)
    D["booking_idx_memberid__resort_id"] = D.groupby(["memberid", "resort_id"])["reservation_id"].cumcount() + 1
    D['days_since_last_checkin_resort'] = D.groupby(['memberid', 'resort_id'])['checkin_date'].diff().apply(lambda x: x.days)

    D.sort_values(["memberid", "booking_date", "checkin_date"], inplace = True, ascending=True)
    D["idx_continued_trip"] = D.groupby(["memberid", "booking_date", "checkin_date"])["reservation_id"].cumcount() + 1

    D.sort_values(["memberid", "resort_id", "checkout_date", "checkin_date"], inplace = True, ascending=True)
    D["idx_inplace_booking"] = D.groupby(["memberid", "resort_id", "checkout_date"])["reservation_id"].cumcount() + 1

    D.sort_values(["memberid", "checkin_date"], inplace = True, ascending=True)
    D["previous_checkout"]= D.groupby(["memberid"])["checkout_date"].shift()
    D["diff_checkin_checkout"] = (D["checkin_date"]  - D["previous_checkout"]).dt.days
    D["diff_checkin_checkout"].fillna(-9999, inplace = True)
    D.drop("previous_checkout", axis = 1, inplace = True)

    D['days_since_last_checkin'].fillna(-1, inplace = True)
    D['days_since_last_checkin'] = D['days_since_last_checkin'].astype(int)
    D['days_since_last_checkin_resort'].fillna(-1, inplace = True)
    D['days_since_last_checkin_resort'] = D['days_since_last_checkin_resort'].astype(int)

    print("transforming date and categorical features...")
    cat_cols = D.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        if col not in ['reservation_id']:
            lbl = LabelEncoder()
            D[col] = lbl.fit_transform(D[col])

    for col in ['booking_date', 'checkin_date', 'checkout_date']:
        D[col] = D[col].map(date_to_integer)

    features = [col for col in D.columns if col not in ['reservation_id', 'amount_spent_per_room_night_scaled', 'memberid']]
    print(f"total features: {len(features)}")
    pprint(features)

    train, test = D[D.amount_spent_per_room_night_scaled >= 0].reset_index(drop = True), \
                            D[D.amount_spent_per_room_night_scaled < 0].reset_index(drop = True)
    print(f'final train shape: {train.shape}')
    print(f'final test shape: {test.shape}')
    del D
    return train, test, features

def do_bag(train, test, features, params, seeds, model_type = "LGB"):
    if model_type not in ("LGB", "XGB"):
        raise ValueError("`model_type` must be either `LGB` or `XGB`")
    preds = np.zeros(test.shape[0])
    for i, seed in enumerate(seeds):
        print("#" * 18)
        print(f"RUN - {i+1} , SEED - {seed}")
        print("#" * 18)

        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"]=str(seed)
        if model_type == 'LGB':
            params.update({
                "feature_fraction_seed" : seed,
                "bagging_fraction_seed" : seed
            })
            preds_ = run_LGB(params, train, test, features, seed=seeds[i])
        else:
            params.update({"seed" : seed})
            preds_ = run_XGB(params, train, test, features, seed=seeds[i])
        preds += preds
    return preds / len(seeds)

if __name__ == '__main__':
    start_time = time.time()
    train, test, features = load_data(config["input_dir"])
    lgb_params = config["lgb_params"]
    xgb_params = config["xgb_params"]
    xgb_params["base_score"] = np.mean(train.amount_spent_per_room_night_scaled)
    lgb_seed_1, lgb_seed_2, xgb_seed = config["lgb_bag_1_seeds"], config["lgb_bag_2_seeds"], config["xgb_seeds"]
    test_ids = test["reservation_id"].values
    p1 = do_bag(train, test, features, lgb_params, lgb_seed_1)
    features = [col for col in features if "_by_resort_id" not in col]
    p2 = do_bag(train, test, features, lgb_params, lgb_seed_2)
    p3 = do_bag(train, test, features, xgb_params, xgb_seed, model_type = "XGB")
    final_preds = 0.5 * p1 + 0.4 * p2 + 0.1 * p3
    df = pd.DataFrame({"reservation_id" : test_ids, "amount_spent_per_room_night_scaled" : final_preds})
    df.to_csv("../submissions/submission.csv", index = False)
    print_runtime(start_time)