from flask import Flask, render_template, redirect, request, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

# Setup DB and login manager
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Ensure static folder exists
if not os.path.exists("static"):
    os.mkdir("static")

data = None  # Global variable

# ----------- AUTH ROUTES -----------

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed = generate_password_hash(password)



        if User.query.filter_by(email=email).first():
            flash('User already exists.')
            return redirect(url_for('signup'))

        new_user = User(email=email, password=hashed)
        db.session.add(new_user)
        db.session.commit()
        flash('Signup successful! Please log in.')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ----------- EDA ROUTES -----------

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    global data
    if request.method == "POST":
        file = request.files["file"]
        clean_option = request.form.get("clean_option")
        graph_option = request.form.getlist("graph_option")
        graph_size = request.form.get("graph_size")

        if file:
            if file.filename.endswith(".csv"):
                data = pd.read_csv(file)
            elif file.filename.endswith((".xlsx", ".xls")):
                data = pd.read_excel(file)
            else:
                return "Unsupported file format! Please upload CSV or Excel."

            # Data cleaning
            if clean_option == "drop_na":
                data = data.dropna()
            elif clean_option == "fill_mean":
                data = data.fillna(data.mean(numeric_only=True))
            elif clean_option == "fill_median":
                data = data.fillna(data.median(numeric_only=True))
            elif clean_option == "fill_mode":
                data = data.fillna(data.mode().iloc[0])

            summary = data.describe().to_html()
            null_info = data.isnull().sum().to_frame("Missing Values")
            null_info["Data Type"] = data.dtypes
            null_info_html = null_info.to_html()

            image_paths = []
            graph_size_dict = {"small": (8, 4), "medium": (10, 6), "large": (12, 8)}

            if "histogram" in graph_option:
                plt.figure(figsize=graph_size_dict[graph_size])
                data.hist(bins=20, figsize=graph_size_dict[graph_size], edgecolor="black")
                plt.tight_layout()
                path = "static/histogram.png"
                plt.savefig(path)
                plt.close()
                image_paths.append(path)

            if "heatmap" in graph_option:
                plt.figure(figsize=graph_size_dict[graph_size])
                numeric_data = data.select_dtypes(include=['float64', 'int64'])
                if not numeric_data.empty:
                    sns.heatmap(numeric_data.corr(), annot=True)
                    path = "static/heatmap.png"
                    plt.savefig(path)
                    plt.close()
                    image_paths.append(path)

            if "boxplot" in graph_option:
                plt.figure(figsize=graph_size_dict[graph_size])
                sns.boxplot(data=data.select_dtypes(include=['float64', 'int64']))
                path = "static/boxplot.png"
                plt.savefig(path)
                plt.close()
                image_paths.append(path)

            if "pairplot" in graph_option:
                pairplot = sns.pairplot(data.select_dtypes(include=['float64', 'int64']))
                path = "static/pairplot.png"
                pairplot.savefig(path)
                plt.close()
                image_paths.append(path)

            advanced_analysis = ""
            if "feature_importance" in graph_option:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100)
                data_numeric = data.select_dtypes(include=["float64", "int64"]).dropna()
                if not data_numeric.empty and data_numeric.shape[1] > 1:
                    X = data_numeric.iloc[:, :-1]
                    y = data_numeric.iloc[:, -1]
                    model.fit(X, y)
                    importances = pd.Series(model.feature_importances_, index=X.columns)
                    feature_html = importances.sort_values(ascending=False).to_frame("Importance").to_html()
                    advanced_analysis += f"<h2>Feature Importance:</h2>{feature_html}"

            if "regression" in graph_option:
                from sklearn.linear_model import LinearRegression
                data_numeric = data.select_dtypes(include=["float64", "int64"]).dropna()
                if not data_numeric.empty and data_numeric.shape[1] > 1:
                    model = LinearRegression()
                    X = data_numeric.iloc[:, :-1]
                    y = data_numeric.iloc[:, -1]
                    model.fit(X, y)
                    score = model.score(X, y)
                    advanced_analysis += f"<h2>Linear Regression R² Score: {score:.2f}</h2>"

            return render_template("result.html", summary=summary, null_info=null_info_html, image_paths=image_paths, advanced_analysis=advanced_analysis)

    return render_template("index.html")

@app.route("/download_cleaned")
@login_required
def download_cleaned():
    global data
    if data is not None:
        output = io.BytesIO()
        data.to_csv(output, index=False)
        output.seek(0)
        return send_file(output, as_attachment=True, download_name="cleaned_data.csv", mimetype="text/csv")
    return "No data to download."

# ---------- INIT DB ----------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=10000, debug=True)
