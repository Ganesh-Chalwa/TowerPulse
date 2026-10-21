from flask import Flask, render_template, request, jsonify

# Import the prediction class.
# The class prepares the 41 input features and
# uses the trained Random Forest model.
from project_app.Class import TowerPowerPrediction


# =========================================================
# CREATE FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/", methods=["GET"])
def home():

    # Display the main TowerPulse prediction interface.
    #
    # Prediction is generated automatically by JavaScript,
    # so there is no Predict button.
    return render_template(
        "index.html",
        prediction=None
    )


# =========================================================
# LIVE PREDICTION API
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # =================================================
        # USER ACTIVITY FEATURES
        # =================================================

        max_active_users_dl = float(
            request.form["max_active_users_dl"]
        )

        max_active_users_ul = float(
            request.form["max_active_users_ul"]
        )

        max_rrc_users = float(
            request.form["max_rrc_users"]
        )

        rrc_users = float(
            request.form["rrc_users"]
        )

        active_users_ul = float(
            request.form["active_users_ul"]
        )

        active_users_dl = float(
            request.form["active_users_dl"]
        )


        # =================================================
        # RADIO QUALITY FEATURES
        # =================================================

        rb_utilization = float(
            request.form["rb_utilization"]
        )

        cqi_rank_1 = float(
            request.form["cqi_rank_1"]
        )

        cqi_rank_2 = float(
            request.form["cqi_rank_2"]
        )

        cqi_rank_3 = float(
            request.form["cqi_rank_3"]
        )

        cqi_rank_4 = float(
            request.form["cqi_rank_4"]
        )

        mimo_rank_dl = float(
            request.form["mimo_rank_dl"]
        )


        # =================================================
        # TIME FEATURES
        # =================================================

        day = int(
            request.form["day"]
        )

        month = int(
            request.form["month"]
        )

        hour = int(
            request.form["hour"]
        )

        minute = int(
            request.form["minute"]
        )

        day_of_week = int(
            request.form["day_of_week"]
        )


        # =================================================
        # THROUGHPUT FEATURES
        # =================================================

        dl_throughput_mbps = float(
            request.form["dl_throughput_mbps"]
        )

        ul_throughput_mbps = float(
            request.form["ul_throughput_mbps"]
        )


        # =================================================
        # CATEGORICAL FEATURES
        # =================================================

        sector = int(
            request.form["sector"]
        )

        site_id = request.form["site_id"]


        # =================================================
        # CREATE MODEL INPUT
        # =================================================

        # Create the prediction object using all the
        # features required by the trained model.
        model_input = TowerPowerPrediction(

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
        )


        # =================================================
        # GENERATE PREDICTION
        # =================================================

        # Generate prediction using the trained
        # Random Forest model.
        prediction = model_input.get_prediction()


        # Round the prediction to two decimal places
        # for a clean UI display.
        prediction = round(
            float(prediction),
            2
        )


        # =================================================
        # RETURN JSON RESPONSE
        # =================================================

        # JavaScript receives this value and updates
        # the prediction card without refreshing the page.
        return jsonify({
            "success": True,
            "prediction": prediction
        })


    except Exception as error:

        # Return the error as JSON instead of crashing
        # the complete web application.
        return jsonify({
            "success": False,
            "error": str(error)
        }), 400


# =========================================================
# START FLASK
# =========================================================

if __name__ == "__main__":

    # Start the Flask development server.
    #
    # debug=True automatically reloads the server when
    # Python code changes during development.
    app.run(
        debug=True
    )