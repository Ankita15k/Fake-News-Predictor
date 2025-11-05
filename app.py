import pickle
import re
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- Setup and Initialization ---
app = Flask(__name__)

# NOTE: You will need to install nltk and download stopwords 
# if you haven't already done so on your deployment environment.
# You can add this line to app.py if it fails:
import nltk; nltk.download('stopwords')

port_stem = PorterStemmer()
# Load Model and Vectorizer
try:
    # 'rb' means read binary
    with open('vector.pkl', 'rb') as vector_file:
        vector_form = pickle.load(vector_file)
    with open('model.pkl', 'rb') as model_file:
        load_model = pickle.load(model_file)
except FileNotFoundError:
    print("FATAL ERROR: model.pkl or vector.pkl not found. Check file paths.")
    vector_form = None
    load_model = None

# --- Preprocessing Function (from your original code) ---
def stemming(content):
    """Cleans and stems the input text."""
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    # Filter out stopwords and stem
    # Ensure 'english' stopwords are available
    try:
        con = [port_stem.stem(word) for word in con if word not in stopwords.words('english')]
    except LookupError:
        # Fallback if stopwords are not downloaded on first run
        con = [port_stem.stem(word) for word in con]
    con = ' '.join(con)
    return con

# --- Prediction Function (from your original code) ---
def fake_news_predictor(news):
    """Predicts if the news is fake (1) or reliable (0) using the loaded model."""
    if load_model and vector_form:
        # 1. Preprocess:
        processed_news = stemming(news)
        
        # 2. Vectorize: Transform the single item into a feature vector
        # NOTE: Your vectorizer must have been fitted on training data already.
        vector_data = vector_form.transform([processed_news])
        
        # 3. Predict:
        prediction = load_model.predict(vector_data)
        
        # Returns the first element (0 or 1)
        return prediction[0] 
    return -1 # Error state

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    news_input = ""
    
    if request.method == 'POST':
        # Get the text input from the HTML form (name='news_article')
        news_input = request.form.get('news_article', '')
        
        if news_input:
            # Perform prediction
            prediction_class = fake_news_predictor(news_input)
            
            if prediction_class == 0:
                prediction_result = "✅ Reliable News 👍"
            elif prediction_class == 1:
                prediction_result = "❌ Fake or Unreliable News"
            else:
                prediction_result = "👎Error: Model not loaded correctly."
        else:
            prediction_result = "Please enter some news content to analyze."

    # Render the index.html template. 
    # Pass the prediction result and the original input back to the template.
    return render_template(
        'index.html', 
        result=prediction_result, 
        input_text=news_input
    )

# Run the app locally (Render ignores this block, but it's useful for testing)
if __name__ == '__main__':
    # You may need to run 'pip install nltk' and then 'python -c "import nltk; nltk.download('stopwords')"' 
    # locally before running this for the first time.
    app.run(debug=True)