import csv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from flask import Flask, render_template, request, redirect, url_for, flash
from markupsafe import Markup
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from collections import defaultdict
from logging.handlers import RotatingFileHandler
import json
from collections import defaultdict
from flask_talisman import Talisman
from flask_wtf import FlaskForm
from wtforms import StringField, validators


# Initializes Flask application and setup security policies with Talisman.
app = Flask(__name__)
Talisman(app, content_security_policy=None)  


# Defines a FlaskForm for the anime input field.
class AnimeForm(FlaskForm):
    anime_title = StringField('Enter Anime Title:', [validators.Length(min=1, max=100), validators.DataRequired()])


# Custom error handler for 500 Internal Server Error.
@app.errorhandler(500)
def handle_500(error):
    return 'Internal Server Error', 500


# Converts the feedback_counts dictionary into a regular dictionary and then saves it as a JSON file.
def save_feedback():
    with open('feedback_data.json', 'w') as f:
        json.dump({k: v for k, v in feedback_counts.items()}, f)


# Reads feedback data from a JSON file and convert it back into a defaultdict structure. If the file does not exist, it initializes
# a new defaultdict with default feedback values.
def load_feedback():
    try:
        with open('feedback_data.json', 'r') as f:
            # Load data and convert it back to defaultdict.
            return defaultdict(lambda: {'good': 0, 'bad': 0}, json.load(f))
    except FileNotFoundError:
        # Return a new defaultdict if no data file exists.
        return defaultdict(lambda: {'good': 0, 'bad': 0})


# Initializes the app and configures the secret key for session management.
app = Flask(__name__)
app.config['SECRET_KEY'] = 'faouri-anime-recommendation-system'
app.config['SESSION_COOKIE_HTTPONLY'] = True   # Mitigates the risk of client side script accessing the protected cookie.
app.config['SESSION_COOKIE_SECURE'] = True   # Ensure cookies are only sent over HTTPS.


# Configures logging to write logs to a file with rotation, ensuring that the log files do not grow indefinitely.
logging.basicConfig(filename='app.log', level=logging.INFO)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


# Sets up rate limiter to limit the number of requests.
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)


# Returns a defaultdict with integer values, initializing new keys with a default value of 0.
def default_feedback():
    return defaultdict(int)  


# Loads the feedback data from a JSON file into a defaultdict.
def load_feedback():
    try:
        with open('feedback_data.json', 'r') as f:
            # Convert JSON back to defaultdict properly
            loaded_feedback = json.load(f)
            feedback_dict = defaultdict(default_feedback, {k: defaultdict(int, v) for k, v in loaded_feedback.items()})
            return feedback_dict
    except FileNotFoundError:
        return defaultdict(default_feedback)


# Initializes feedback counts from loaded feedback data.
feedback_counts = load_feedback()

# Load and preprocess the anime dataset
# This section reads the anime data from a CSV file, splits the genres into lists, and converts the anime names to lowercase for consistency in matching.
data = pd.read_csv('anime.csv')
data['Genres'] = data['Genres'].str.split(',')
data['Name'] = data['Name'].str.lower()


# Explode genres into individual rows and create genre vectors.
exploded_genres = data.explode('Genres')
all_genres = sorted(set(exploded_genres['Genres']))
data['Genre_vector'] = data['Genres'].apply(lambda genres: [1 if genre in genres else 0 for genre in all_genres])


# Normalizes the genre vectors for calculating cosine similarity accurately.
scaler = StandardScaler()
data['Genre_vector'] = list(scaler.fit_transform(pd.DataFrame(data['Genre_vector'].tolist())))



# Handles the requests for the home page. It displays a form for the user to input an anime title,  
# and provides recommendations based on genre similarity.
@app.route('/', methods=['GET', 'POST'])
def home():
    anime_titles = data['Name'].unique().tolist()
    if request.method == 'POST':
        anime_title = request.form['anime_title'].strip().lower()
        matching_anime = data[data['Name'] == anime_title]
        if not matching_anime.empty:
            user_vector = matching_anime.iloc[0]['Genre_vector']
            data['Similarity'] = data['Genre_vector'].apply(lambda x: cosine_similarity([x], [user_vector]).flatten()[0])
            top_recommendations = data.sort_values(by='Similarity', ascending=False).head(10)
            genre_chart = create_genre_chart(top_recommendations)
            similarity_chart = create_similarity_chart(top_recommendations)
            score_chart = create_score_chart(top_recommendations)
            return render_template('home.html', anime_title=anime_title, recommendations=top_recommendations[['Name', 'Score']].to_dict(orient='records'), 
                                   anime_titles=anime_titles, genre_chart=genre_chart, similarity_chart=similarity_chart, score_chart=score_chart)
        else:
            flash('Anime not found!', 'error')
    return render_template('home.html', anime_titles=anime_titles)


# Calculates the accuracy of the recommendations based on user feedback.
# It compares the number of good recommendations to the total number of feedback entries.
def calculate_and_save_accuracy():
    total_good = sum(feedback['recommendation good'] for feedback in feedback_counts.values())
    total_bad = sum(feedback['recommendation bad'] for feedback in feedback_counts.values())
    total = total_good + total_bad
    accuracy = (total_good / total) * 100 if total > 0 else 0

    with open('accuracy.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['total_good', 'total_bad', 'accuracy'])
        writer.writerow([total_good, total_bad, accuracy])

# Handles the recording of user feedback on the recommendations. It saves the feedback to both a JSON file and a CSV file for analysis.
@app.route('/rate_recommendations', methods=['POST'])
@limiter.limit("10 per minute")
def rate_recommendations():
    feedback = request.form['feedback'].lower()  # Converts feedback to lowercase.
    anime_title = request.form['anime_title']
    # Ensures the dictionary is initialized for unknown anime titles or feedback types.
    if anime_title not in feedback_counts:
        feedback_counts[anime_title] = default_feedback()
    feedback_counts[anime_title][feedback] += 1  # Increments the feedback count.
    save_feedback()

    # Calls the calculate and save accuracy function.
    calculate_and_save_accuracy()

    try:
        with open('feedback.csv', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([anime_title, feedback])
        flash('Feedback recorded successfully!', 'success')
    except Exception as e:
        flash(f'Error recording feedback: {e}', 'error')
    return redirect(url_for('home'))


# Function that uses Plotly to create a pie chart showing the distribution of genres among the top recommendations.
def create_genre_chart(df):
    fig = px.pie(df, names='Genres', title='<b>Genre Distribution</b>')
    return Markup(fig.to_html(full_html=False))


# Function that uses Plotly to create a scatter plot comparing the similarity scores to the anime ratings.
def create_similarity_chart(df):
    fig = px.scatter(df, x='Score', y='Similarity', title='<b>Recommendation Similarity vs Anime Rating</b>',
                     labels={'Score': 'Anime Rating', 'Similarity': 'Similarity Score'},
                     hover_data=['Name'])
    return Markup(fig.to_html(full_html=False))



# Function that uses Plotly to create a bar chart comparing the anime ratings and similarity scores side by side for each anime.
def create_score_chart(df):
    # Ensure that the data types are consistent and numeric.
    df['Similarity'] = pd.to_numeric(df['Similarity'], errors='coerce')  # Convert to numeric.
    df['Rating'] = pd.to_numeric(df['Score'], errors='coerce')  # Rename and convert 'Score' to 'Rating'.

    # Creates a plot with renamed metrics.
    fig = px.bar(df, x='Name', y=['Similarity', 'Rating'], title='<b>Anime Rating vs Similarity Analysis</b>',
                 labels={'value': 'Value', 'variable': 'Metrics'},
                 barmode='group',  
                 hover_data={'Name': False})  

    fig.update_layout(
        xaxis_title="Anime Name",
        yaxis_title="Rating and Similarity",
        legend_title="Metrics"
    )
    return Markup(fig.to_html(full_html=False))


# Function that is used to shut down the Flask server.
# It also ensures that the feedback data is saved before the server is shut down.
@app.route('/shutdown', methods=['POST'])
def shutdown():
    save_feedback()  # Save before shutting down.
    # Code to actually shut down the Flask server.
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'


# This starts the Flask application in debug mode, will be disabled in production.
if __name__ == '__main__':
    app.run(debug=True)
