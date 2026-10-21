import json
import pickle
import numpy as np
import pandas as pd

# Import the paths defined in config.py.
# This keeps model and JSON paths centralized.
import config


# =========================================================
# LOAD MODEL ONLY ONCE
# =========================================================

# The Random Forest model is loaded only once when
# Flask starts importing this file.
#
# This is important for performance because loading
# the .pkl file for every slider movement would make
# the web application slow.
with open(config.MODEL_FILE_PATH, "rb") as f:
    MODEL = pickle.load(f)


# Load the project feature information only once.
with open(config.JSON_FILE_PATH, "r") as f:
    PROJECT_DATA = json.load(f)


# Store the exact 41 feature names used during training.
# The same order must be used during prediction.
SELECTED_COLS = PROJECT_DATA["features"]


class TowerPowerPrediction:

    def __init__(
        self,
        max_active_users_dl,
        max_active_users_ul,
        max_rrc_users,
        rb_utilization,
        cqi_rank_1,
        cqi_rank_2,
        cqi_rank_3,
        cqi_rank_4,
        rrc_users,
        active_users_ul,
        active_users_dl,
        mimo_rank_dl,
        day,
        month,
        hour,
        minute,
        day_of_week,
        dl_throughput_mbps,
        ul_throughput_mbps,
        sector,
        site_id
    ):

        # Store the 5G user activity features.
        self.max_active_users_dl = max_active_users_dl
        self.max_active_users_ul = max_active_users_ul
        self.max_rrc_users = max_rrc_users
        self.rrc_users = rrc_users
        self.active_users_ul = active_users_ul
        self.active_users_dl = active_users_dl

        # Store the radio quality features.
        self.rb_utilization = rb_utilization
        self.cqi_rank_1 = cqi_rank_1
        self.cqi_rank_2 = cqi_rank_2
        self.cqi_rank_3 = cqi_rank_3
        self.cqi_rank_4 = cqi_rank_4
        self.mimo_rank_dl = mimo_rank_dl

        # Store the time-related features.
        self.day = day
        self.month = month
        self.hour = hour
        self.minute = minute
        self.day_of_week = day_of_week

        # Store throughput features.
        self.dl_throughput_mbps = dl_throughput_mbps
        self.ul_throughput_mbps = ul_throughput_mbps

        # Store categorical features.
        self.sector = sector
        self.site_id = site_id


    def get_prediction(self):

        # Create an array containing exactly the same
        # number of features used by the trained model.
        test_array = np.zeros(
            len(SELECTED_COLS)
        )


        # =================================================
        # USER ACTIVITY FEATURES
        # =================================================

        test_array[
            SELECTED_COLS.index(
                "5G max active users DL"
            )
        ] = self.max_active_users_dl


        test_array[
            SELECTED_COLS.index(
                "5G max active users UL"
            )
        ] = self.max_active_users_ul


        test_array[
            SELECTED_COLS.index(
                "5G max RRC users"
            )
        ] = self.max_rrc_users


        test_array[
            SELECTED_COLS.index(
                "5G RRC users"
            )
        ] = self.rrc_users


        test_array[
            SELECTED_COLS.index(
                "5G active users UL"
            )
        ] = self.active_users_ul


        test_array[
            SELECTED_COLS.index(
                "5G active users DL"
            )
        ] = self.active_users_dl


        # =================================================
        # RADIO QUALITY FEATURES
        # =================================================

        test_array[
            SELECTED_COLS.index(
                "5G RB utilization"
            )
        ] = self.rb_utilization


        test_array[
            SELECTED_COLS.index(
                "5G CQI rank 1"
            )
        ] = self.cqi_rank_1


        test_array[
            SELECTED_COLS.index(
                "5G CQI rank 2"
            )
        ] = self.cqi_rank_2


        test_array[
            SELECTED_COLS.index(
                "5G CQI rank 3"
            )
        ] = self.cqi_rank_3


        test_array[
            SELECTED_COLS.index(
                "5G CQI rank 4"
            )
        ] = self.cqi_rank_4


        test_array[
            SELECTED_COLS.index(
                "5G MIMO rank DL"
            )
        ] = self.mimo_rank_dl


        # =================================================
        # TIME FEATURES
        # =================================================

        test_array[
            SELECTED_COLS.index("Day")
        ] = self.day


        test_array[
            SELECTED_COLS.index("Month")
        ] = self.month


        test_array[
            SELECTED_COLS.index("Hour")
        ] = self.hour


        test_array[
            SELECTED_COLS.index("Minute")
        ] = self.minute


        test_array[
            SELECTED_COLS.index("Day_of_Week")
        ] = self.day_of_week


        # =================================================
        # THROUGHPUT FEATURES
        # =================================================

        test_array[
            SELECTED_COLS.index(
                "DL_Throughput_Mbps"
            )
        ] = self.dl_throughput_mbps


        test_array[
            SELECTED_COLS.index(
                "UL_Throughput_Mbps"
            )
        ] = self.ul_throughput_mbps


        # =================================================
        # SECTOR ONE-HOT ENCODING
        # =================================================

        sector_column = (
            "Sector_" +
            str(self.sector)
        )


        # Sector 1-4 are actual columns in the
        # final selected feature set.
        if sector_column in SELECTED_COLS:

            test_array[
                SELECTED_COLS.index(
                    sector_column
                )
            ] = 1


        # =================================================
        # BASE STATION ONE-HOT ENCODING
        # =================================================

        site_column = (
            "Base station_Site " +
            str(self.site_id)
        )


        # Set the selected base station column to 1.
        if site_column in SELECTED_COLS:

            test_array[
                SELECTED_COLS.index(
                    site_column
                )
            ] = 1

        else:

            raise ValueError(
                f"Site {self.site_id} is not available "
                f"in the final model features."
            )


        # =================================================
        # CREATE DATAFRAME
        # =================================================

        # Convert the input array into a DataFrame.
        #
        # The column names and order are exactly the same
        # as those used while training the Random Forest.
        test_df = pd.DataFrame(
            [test_array],
            columns=SELECTED_COLS
        )


        # =================================================
        # MAKE PREDICTION
        # =================================================

        # Use the already-loaded Random Forest model.
        # The pickle file is NOT loaded again.
        prediction = MODEL.predict(
            test_df
        )[0]


        return prediction