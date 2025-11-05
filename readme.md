# 📰 Fake News Detection Classifier

This project is a **Machine Learning web application** designed to classify news articles as **Reliable** or **Fake/Unreliable** using a trained classification model. The application logic is implemented using **Flask** and is deployed via **Render**.

## 📂 Project Structure

The repository contains the following key files:

| File/Folder | Description |
| :--- | :--- |
| **`app.py`** | The main application file written in **Flask**, handling web routes and model loading. |
| **`model.pkl`** | The serialized **Trained Classification Model**. |
| **`vector.pkl`** | The serialized **TF-IDF Vectorizer** used to transform text into numerical features. |
| **`requirements.txt`** | Lists all necessary Python dependencies for deployment. |
| **`Procfile`** | Instructions for the Render web server on how to start the application. |
| **`templates/index.html`** | The HTML file providing the user interface for text input. |
| **`train/`** (Folder) | Likely contains scripts or data used during model training. |

---

## 🚀 Deployment & Running the App

### Local Testing

To run the application on your **local machine** for testing:

1.  **Activate Virtual Environment** (e.g., `source venv/bin/activate`).
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download NLTK Data (if needed):**
    ```bash
    python -c "import nltk; nltk.download('stopwords')"
    ```
4.  **Run Flask Server:**
    ```bash
    python app.py
    ```
    Access the app at: `http://127.0.0.1:5000/`

### Deployment (Render Configuration)

The application is configured for deployment using the following settings in Render:

* **Build Command:** `pip install -r requirements.txt && python -c "import nltk; nltk.download('stopwords')"`
* **Start Command:** `gunicorn app:app`

---

## 🛠️ Technology Stack

* **Backend Framework:** **Flask**
* **Machine Learning:** **Scikit-learn** (for model and vectorizer)
* **Text Processing:** **NLTK** (Stopword removal and Stemming)
* **Hosting:** **Render**