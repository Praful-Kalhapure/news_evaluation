from flask import Flask, request, render_template, redirect, session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] ='sqlite:///database.db'

db = SQLAlchemy(app)
app.secret_key = 'secret key'

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, name, email, password):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html', error='Invalid user')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('dashboard.html', user=user)
    return render_template('login.html')




@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/login')


@app.route('/result')
def report():
    return render_template('result.html')


@app.route('/about')
def about():
    return render_template('about.html')






@app.route('/analyze', methods=['POST'])
def analyze():
    url_to_scrape = request.form['urlinput']

    # Run your Python code here
    result = analyze_sentiment(url_to_scrape)

    # Pass the result to the template and render result.html
    return render_template('result.html', result=result)

def analyze_sentiment(url_to_scrape):
    html_content = get_html_content(url_to_scrape)

    positive_words, negative_words = load_sentiment_words('list.csv')

    if html_content is not None and positive_words is not None and negative_words is not None:
        positive_count, negative_count, found_words = scrape_and_compare_sentiment(html_content, positive_words, negative_words)

        # Construct the result message
        result_message = f"<div style='text-align: center;'><b>Positive words count: </b>{positive_count}<br><b>Negative words count:</b> {negative_count}</div>"


        # Construct positive and negative word sections
        positive_words_section = f"<b>Positive Words:</b> {', '.join(set([word for word in found_words if word in positive_words]))}"
        negative_words_section = f"<b>Negative Words:</b> {', '.join(set([word for word in found_words if word in negative_words]))}"

        # Combine the result message, positive words section, and negative words section
        result = f"{result_message}<br><br><table border='1'>\
                    <tr>\
                        <th>Positive Words</th>\
                        <th>Negative Words</th>\
                    </tr>\
                    <tr>\
                        <td>{', '.join(set([word for word in found_words if word in positive_words]))}</td>\
                        <td>{', '.join(set([word for word in found_words if word in negative_words]))}</td>\
                    </tr>\
                  </table><br>"

        # Create a pie chart
        labels = ['Positive', 'Negative']
        sizes = [positive_count, negative_count]
        colors = ['green', 'red']
        explode = (0.1, 0)

        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        title_text = 'Analysis Pie Chart'

        # Create an underlined version using equal signs (or any character of your choice)
        underlined_title = title_text + '\n' + '----------------' * len(title_text)
        plt.title(underlined_title)
        

        # Save the plot to a BytesIO object
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_data = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()

        # Construct the HTML to display the image
        image_html = f'<img src="data:image/png;base64,{img_data}" alt="Sentiment Analysis Pie Chart">'

        # Add the image HTML to the result
        result += f"<br>{image_html}"

        return result
    else:
        return "Error in sentiment analysis"

def get_html_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return None

def load_sentiment_words(file_path):
    try:
        df = pd.read_csv(file_path)
        positive_words = df['positive'].tolist()
        negative_words = df['negative'].tolist()
        return positive_words, negative_words
    except Exception as e:
        print(f"Error loading sentiment words from CSV: {e}")
        return None, None

def scrape_and_compare_sentiment(html_content, positive_words, negative_words):
    if html_content is None or positive_words is None or negative_words is None:
        return None, None, None

    soup = BeautifulSoup(html_content, 'html.parser')

    text_content = soup.get_text()

    # Split the text content into words
    words = text_content.split()

    # Create sets to store unique positive and negative words found on the website
    unique_positive_words = set(word.lower() for word in words if word.lower() in positive_words)
    unique_negative_words = set(word.lower() for word in words if word.lower() in negative_words)

    # Count the unique positive and negative words
    positive_count = len(unique_positive_words)
    negative_count = len(unique_negative_words)

    # Combine the sets to get all unique found words
    found_words = list(unique_positive_words.union(unique_negative_words))

    return positive_count, negative_count, found_words

if __name__ == '__main__':
    app.run(debug=True)
