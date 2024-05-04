from flask import Flask, request, render_template, redirect, session
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from flask import jsonify
from sklearn.svm import SVC


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
    url_to_scrape = request.form['url']

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
        plt.figure(figsize=(4,4))
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



# Load the model
model = LogisticRegression()
dataset = pd.read_csv("NEW_WEB_DES.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
sc = StandardScaler()
X = sc.fit_transform(X)
model.fit(X, y)

@app.route('/analyzedes', methods=['POST'])
def analyzedes():
    if request.method == 'POST':
        layout = float(request.form['layout'])
        typography = float(request.form['typography'])
        color_scheme = float(request.form['color_scheme'])
        responsive_design = float(request.form['responsive_design'])
        
        # Scale the input
        input_data = sc.transform([[layout, typography, color_scheme, responsive_design]])
        
        # Predict using the model
        prediction = model.predict(input_data)
        
        # Determine the result label
        result = "Good Website Design [1]" if prediction[0] == 1 else "Bad Website Design [0]"
        
        # Construct the HTML response
        html_response = f"<p>The predicted result is: {result}</p>"
        
        return html_response

class WebsiteAnalyzer:
    def __init__(self):
        self.model_des = None
        self.model_fun = None
        self.model_soc = None
        self.sc_des = None
        self.sc_fun = None
        self.sc_soc = None
        self.load_models()

    def load_models(self):
        # Load the design model
        self.model_des = LogisticRegression()
        dataset_des = pd.read_csv("NEW_WEB_DES.csv")
        
        # Splitting the dataset into Independent and Dependent
        X_des = dataset_des.iloc[:, :-1].values
        y_des = dataset_des.iloc[:, -1].values
        
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X_des,y_des , test_size=0.2, random_state=42)
        
        self.sc_des = StandardScaler()
        X_train = self.sc_des.fit_transform(X_train)
        X_test=self.sc_des.transform(X_test)
        self.model_des.fit( X_train, y_train)
        y_pred = self.model_des.predict(X_test)
        print("\nACCURACY SCORE of Web Design:",accuracy_score(y_test,y_pred))
        cn=confusion_matrix(y_test,y_pred)
        print(cn)
        
        
        
        # Load the functionality model
        dataset_fun = pd.read_csv("NEW_WEB_FUNT.csv")
        X_fun = dataset_fun.iloc[:, :-1].values
        y_fun = dataset_fun.iloc[:, -1].values
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X_fun,y_fun, test_size=0.2, random_state=42)
        
        self.sc_fun = StandardScaler()
        X_train = self.sc_fun.fit_transform(X_train)
        X_test=self.sc_fun.transform(X_test)
        self.model_fun = LogisticRegression()
        self.model_fun.fit( X_train, y_train)
        y_pred = self.model_fun.predict(X_test)
        print("\nACCURACY SCORE of Web Functionality:",accuracy_score(y_test,y_pred))
        cn=confusion_matrix(y_test,y_pred)
        print(cn)
        
        

        # Load the social model
        dataset_soc = pd.read_csv("NEW_WEB_SOC.csv")
        X_soc = dataset_soc.iloc[:, :-1].values
        y_soc = dataset_soc.iloc[:, -1].values
        
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X_soc,y_soc, test_size=0.2, random_state=42)
        
        self.sc_soc = StandardScaler()
        X_train = self.sc_soc.fit_transform(X_train)
        X_test= self.sc_soc.transform(X_test)
        self.model_soc = LogisticRegression()
        self.model_soc.fit( X_train, y_train)
        y_pred = self.model_soc.predict(X_test)
        print("\nACCURACY SCORE of Social Media:",accuracy_score(y_test,y_pred))
        cn=confusion_matrix(y_test,y_pred)
        print(cn)
        

        
        #Load Overall WEBSITE model
        dataset_bi = pd.read_csv("Final_BI.csv")
        X_bi = dataset_bi.iloc[:, :-1].values
        y_bi = dataset_bi.iloc[:, -1].values
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X_bi,y_bi, test_size=0.2, random_state=42)
        
        self.sc_bi = StandardScaler()
        X_train = self.sc_bi.fit_transform(X_train)
        X_test= self.sc_bi.transform(X_test)
        self.model_bi = LogisticRegression()
        self.model_bi.fit( X_train, y_train)
        y_pred = self.model_bi.predict(X_test)
        print("\nACCURACY SCORE Overall Website:",accuracy_score(y_test,y_pred))
        cn=confusion_matrix(y_test,y_pred)
        print(cn)
    
        

    def analyze_design(self, layout, typography, color_scheme, responsive_design):
        input_data = self.sc_des.transform([[layout, typography, color_scheme, responsive_design]])
        prediction = self.model_des.predict(input_data)
        result2 = f"{'GOOD' if prediction[0] == 1 else 'BAD - Need Improvment'}"
        print("Result: ", result2)

        return result2
    



    def analyze_functionality(self, load_time, personalization, multimedia_support, accessibility):
        input_data = self.sc_fun.transform([[load_time, personalization, multimedia_support, accessibility]])
        prediction = self.model_fun.predict(input_data)
        result3 = f"{'GOOD' if prediction[0] == 1 else 'BAD - Need Improvment'}"
        print("Result: ", result3)
        return result3
    
    def analyze_social(self, shares, likes, comments, clicks):
        input_data = self.sc_soc.transform([[shares, likes, comments, clicks]])
        prediction = self.model_soc.predict(input_data)
        result4 = f"{'GOOD' if prediction[0] == 1 else 'BAD - Need Improvment'}"
        print("Result: ", result4)
        return result4
    

    def analyze_bi(self, result2, result3, result4):
        result2_binary = 1 if result2 == 'GOOD' else 0
        result3_binary = 1 if result3 == 'GOOD' else 0
        result4_binary = 1 if result4 == 'GOOD' else 0
        input_data = self.sc_bi.transform([[result2_binary, result3_binary, result4_binary]])
        prediction = self.model_bi.predict(input_data)
        result_5 = f"{'GOOD' if prediction[0] == 1 else 'BAD'}"
       
        return result_5

analyzer = WebsiteAnalyzer()
@app.route('/analyzeall', methods=['POST'])
def analyzeall():
    layout = float(request.form['layout'])
    typography = float(request.form['typography'])
    color_scheme = float(request.form['color_scheme'])
    responsive_design = float(request.form['responsive_design'])


    load_time = float(request.form['load_time'])
    personalization = float(request.form['personalization'])
    multimedia_support = float(request.form['multimedia_support'])
    accessibility = float(request.form['accessibility'])


    shares = float(request.form['shares'])
    likes = float(request.form['likes'])
    comments = float(request.form['comments'])
    clicks = float(request.form['clicks'])

    result2 = analyzer.analyze_functionality(load_time, personalization, multimedia_support, accessibility)
    result3 = analyzer.analyze_design(layout, typography, color_scheme, responsive_design)
    result4 = analyzer.analyze_social(shares, likes, comments, clicks)
    result5 = analyzer.analyze_bi(result2, result3, result4)

    #result5 = analyzer.analyze_bi(result2, result3, result4)
    #positive_count, negative_count, _ = analyze_sentiment(url_to_scrape)  # Assuming you have access to these values here
    #n_content = f"<div>{1 if positive_count > negative_count else 0}</div>"

    return render_template('final.html',result2=result2,result3=result3,result4=result4, result5=result5)

###############################################################################################################################################################



@app.route('/final.html', methods=['POST'])
def generate_final():
    result2 = request.form.get('result2')
    result3 = request.form.get('result3')
    result4 = request.form.get('result4')
    #result_5 = request.form.get('result5')
    img_data = request.form.get('img_data')
    result5 = analyzer.analyze_bi(result2, result3, result4)

   

    #n_content = request.form.get('n_content')  # Add this line to get the n_content value
    return render_template('final.html', result2=result2, result3=result3, result4=result4,result5=result5, img_data=img_data)




if __name__ == '__main__':
    app.run(debug=True)




