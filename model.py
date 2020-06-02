"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from category_encoders import *
from sklearn.decomposition import PCA
from geopy.distance import vincenty

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    input_df = pd.DataFrame.from_dict([feature_vector_dict])
    # Column name fixing
    input_df.columns = input_df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_').str.replace('=', '_')
    input_df.columns = input_df.columns.str.replace('__', '_')
    input_df.columns = input_df.columns.str.replace('(', '').str.replace(')', '')
    input_df.columns = input_df.columns.str.replace('__', '_')

    # Missing values and unique values
    def drop_columns(input_df, threshold, unique_threshold):
        for column in input_df.columns:
            if ((input_df[column].isna().mean() * 100) > threshold):
                input_df.drop(column, axis = 1, inplace = True)
            elif (input_df[column].nunique() < unique_threshold):
                input_df.drop(column, axis = 1, inplace = True)

            
    drop_columns(input_df, 70, 2)

    # Datetime convertion
    def convert_to_datetime(df):
        for col in df.columns:
            if col.endswith("time"):
                df[col] = pd.to_datetime(df[col])
        return df
    
    test = convert_to_datetime(input_df)

    # The function classifies the days of the month to their respective quarter of the month
    def quarter_of_month(x):
        if x >= 1 and x <= 7:
            return 1
        elif x > 7 and x <= 14:
            return 2
        elif x > 14 and x <= 21:
            return 3
        else:
            return 4
    
    #Applying quarter_of_month
    test["pick_up_quarter_month"] = test["pickup_day_of_month"].apply(quarter_of_month)

    # A function that classifies days as either falling on a weekend or not_weekend
    def weekend(x):
        if x <= 5:
            return 1
        else:
            return 0
    
    #Applying the not_weekend function to create a new feature
    test["weekend"] = test["pickup_weekday_mo_1"].apply(weekend)

    # test set
    test['placement_to_confirmation_diff'] = (test['confirmation_time'] - test['placement_time']).astype('timedelta64[s]')
    test['confirmation_to_arrivalpickup_diff'] = (test['arrival_at_pickup_time'] - test['confirmation_time']).astype('timedelta64[s]')
    test['arrivalpickup_to_pickup_diff'] = (test['pickup_time'] - test['arrival_at_pickup_time']).astype('timedelta64[s]')

    # Extracting the hour and minute component of pickup_time
    test['pickup_time'.split('_')[0] + '_hour'] = test['pickup_time'].dt.hour
    test['pickup_time'.split('_')[0] + '_minute'] = test['pickup_time'].dt.minute
    test['pickup_time'.split('_')[0] + '_second'] = test['pickup_time'].dt.second



    #Transforming Cyclic features
    test['pickup_day_of_month_sin'] = np.sin(test['pickup_day_of_month']*(2.*np.pi/31))
    test['pickup_day_of_month_cos'] = np.cos(test['pickup_day_of_month']*(2.*np.pi/31))
    test['pickup_weekday_sin'] = np.sin(test['pickup_weekday_mo_1']*(2.*np.pi/7))
    test['pickup_weekday_cos'] = np.cos(test['pickup_weekday_mo_1']*(2.*np.pi/7))
    test['pickup_hour_sin'] = np.sin(test['pickup_hour']*(2.*np.pi/23)) 
    test['pickup_hour_cos'] = np.cos(test['pickup_hour']*(2.*np.pi/23)) 
    test['pickup_minute_sin'] = np.sin(test['pickup_minute']*(2.*np.pi/60)) 
    test['pickup_minute_cos'] = np.cos(test['pickup_minute']*(2.*np.pi/60)) 
    test['pickup_second_sin'] = np.sin(test['pickup_second']*(2.*np.pi/60))
    test['pickup_second_cos'] = np.cos(test['pickup_second']*(2.*np.pi/60))

    time_cols = ['placement_time','arrival_at_pickup_time','pickup_time','confirmation_time']
    test.drop(time_cols, axis=1, inplace=True)



    more_test_cols = ['placement_day_of_month','confirmation_day_of_month',
                      'pickup_day_of_month', 'pickup_weekday_mo_1',
                      'arrival_at_pickup_day_of_month','arrival_at_pickup_weekday_mo_1',
                      'placement_weekday_mo_1','confirmation_weekday_mo_1',
                      'pickup_minute', 'pickup_hour','pickup_second']

    test.drop(more_test_cols, axis=1, inplace=True)

    #Creating new variables that will measure the Rider's Rating and Productivity
    test.loc[:, 'rating_factor'] = (test['average_rating'] * test['no_of_ratings'])
    test.loc[:, 'rider_productivity'] = (test['age'] / test['no_of_orders'])

    test['average_rider_speed'] = test['rider_id'].map(train.groupby('rider_id')['speed_meter_per_second'].mean())
    test['average_rider_speed'].fillna(train['speed_meter_per_second'].mean(), inplace=True)


    def haversine(pick_lat, pick_long, drop_lat, drop_long):
        """
        Calculate the circle distance between two points 
        on the earth (specified in decimal degrees)
        """
       # approximate radius of earth in km
        R = 6373.0

        # Converting degrees to radians
        pick_lat = np.deg2rad(pick_lat)                     
        pick_long = np.deg2rad(pick_long)     
        drop_lat = np.deg2rad(drop_lat)                       
        drop_long = np.deg2rad(drop_long)  

        dist = np.sin((drop_lat - pick_lat)/2)**2 + np.cos(pick_lat)*np.cos(drop_lat) * np.sin((drop_long - pick_long)/2)**2

        return 2 * R * np.arcsin(np.sqrt(dist))


    test.loc[:, 'distance_haversine'] = haversine(test['pickup_lat'].values, test['pickup_long'].values, test['destination_lat'].values, test['destination_long'].values)
    test["min_distance_pick_to_arrival"] = test.apply(lambda x: vincenty((x["pickup_lat"], x["pickup_long"]), (x["destination_lat"], x["destination_long"])).kilometers, axis = 1)
    test["excess_distance"] = test["distance_km"] - test["min_distance_pick_to_arrival"]

    def bearing_array(lat1, long1, lat2, long2):
        '''
        The function calculates the direction bearing of the given coordinates
        '''
        long_delta_rad = np.radians(long2 - long1)
        lat1, long1, lat2, long2 = map(np.radians, (lat1, long1, lat2, long2))
        y = np.sin(long_delta_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(long_delta_rad)
        angle = np.degrees(np.arctan2(y, x))
        return angle

    #Calculating the direction of the destination by applying bearing_array func on the coordinates
    test.loc[:, 'direction'] = bearing_array(test['pickup_lat'].values, test['pickup_long'].values, test['destination_lat'].values, test['destination_long'].values)

    coordinates = np.vstack((train[['pickup_lat', 'pickup_long']].values,
                        train[['destination_lat', 'destination_long']].values))

    pca = PCA(random_state=rs).fit(coordinates) #Instance
    # Train data transformation
    # Test data transformation
    test['pickup_pca_0'] = pca.transform(test[['pickup_lat', 'pickup_long']])[:, 0]
    test['pickup_pca_1'] = pca.transform(test[['pickup_lat', 'pickup_long']])[:, 1]
    test['dropoff_pca_0'] = pca.transform(test[['destination_lat', 'destination_long']])[:, 0]
    test['dropoff_pca_1'] = pca.transform(test[['destination_lat', 'destination_long']])[:, 1]

    #performing our own clustering on all the points pickup/destination in the data using KMeans
    Kmeans = MiniBatchKMeans(n_clusters=14, batch_size=1000,random_state=rs) #Instance
    Kmeans.fit(coordinates[np.arange(0, len(coordinates), 1)]) # fitting
    test['pickup_cluster'] = Kmeans.predict(test[['pickup_lat', 'pickup_long']])
    test['dropoff_cluster'] = Kmeans.predict(test[['destination_lat', 'destination_long']])
    test['center_latitude'] = (test['pickup_lat'].values + test['destination_lat'].values) / 2
    test['center_longitude'] = (test['pickup_long'].values + test['destination_long'].values) / 2

    # Dropping all the columns we will no longer need in the predictors (X) data

    test_cols_drop = ['order_no','user_id','no_of_orders','min_distance_pick_to_arrival', 'no_of_ratings', 
                      'age', 'average_rating', 'distance_km']

    test['platform_type'] = test['platform_type'].astype('category')

    X_test_data = test.drop(test_cols_drop, axis=1)
    X_test_order_no = test['order_no']

    #Feature Encoding
    oce = OneHotEncoder(cols=['platform_type','personal_or_business'])
    hce = TargetEncoder(cols=['rider_id'], smoothing = 40, min_samples_leaf = 7)
    X_test_data = oce.fit_transform(X_test_data)
    X_test_data = hce.fit_transform(X_test_data)
    X_test_data.drop('personal_or_business_2', axis=1, inplace=True)

    return test

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model_1, model_2):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction_1 = model_1.predict(prep_data)
    prediction_2 = model_2.predict(prep_data)
    prediction = 0.2828 * prediction_1 + (1 - 0.2828) * prediction_2
    # Format as list for output standerdisation.
    return prediction[0].tolist()
