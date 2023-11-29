# Create API to host the model and web app

import os
from flask import Flask, request, jsonify, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from .model.predict_model import predict_class


# Create Flask app
app = Flask(
    __name__,
    template_folder="web_interface/templates",
    static_folder="web_interface/static",
)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the file from post request
        f = request.files["file"]
        print(f)

        #  Get the filename
        x = f.filename
        print(x)

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, "uploads", secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = predict_class(x)
        print(preds)
        return render_template(
            "index.html", prediction_text="The image is {}".format(f.filename)
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
