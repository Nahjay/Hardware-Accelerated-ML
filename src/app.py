# Create API to host the model and web app

import os
from flask import Flask, request, redirect, render_template, flash
from model.predict_model import predict_class

# Constants
UPLOAD_FOLDER = "web_interface/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# Create Flask app
app = Flask(
    __name__,
    template_folder="web_interface/templates",
    static_folder="web_interface/static",
)

# Configure Flask app
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]

        # If the user does not select a file, the browser will submit an empty part without a filename
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save the uploaded file to the upload folder
            filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filename)

            # Process the uploaded file using model
            prediction_result = predict_class.predict_class(filename)

            # Render the index.html template and add the prediction result to the template
            return render_template("index.html", prediction_result=prediction_result)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
