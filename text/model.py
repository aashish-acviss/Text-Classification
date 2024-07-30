import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

# Global variables to be used for prediction
fitted_vectorizer = None
m1 = None
m = None
id_to_category = None


def train_model():
    global fitted_vectorizer, m1, id_to_category, m

    # Load dataset
    dataset = pd.read_csv('website_classification.csv')
    df = dataset[['website_url', 'cleaned_website_text', 'Category']].copy()

    # Create a new column 'category_id' with encoded categories
    df['category_id'] = df['Category'].factorize()[0]
    category_id_df = df[['Category', 'category_id']].drop_duplicates()

    # Dictionaries for future use
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'Category']].values)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_website_text'], df['category_id'], test_size=0.25,
                                                        random_state=0)

    # Initialize and fit TfidfVectorizer
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')
    fitted_vectorizer = tfidf.fit(X_train)
    tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)

    # Train the LinearSVC model with dual='auto'
    m = LinearSVC(dual='auto').fit(tfidf_vectorizer_vectors, y_train)

    # Calibrate the model
    m1 = CalibratedClassifierCV(estimator=m, cv="prefit").fit(tfidf_vectorizer_vectors, y_train)

    # Transform the test data and make predictions
    X_test_tfidf = fitted_vectorizer.transform(X_test)
    y_pred = m1.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on X_test: {accuracy:.4f}")


def predict_category_m1(text):
    global fitted_vectorizer, m1, id_to_category

    # Transform the input text using the fitted vectorizer
    text_tfidf = fitted_vectorizer.transform([text])
    # Predict the category ID
    predicted_class_id = m1.predict(text_tfidf)[0]
    # Map the category ID to the category label
    predicted_class_label = id_to_category[predicted_class_id]
    return predicted_class_label

def predict_category_m(text):
    global fitted_vectorizer, m, id_to_category

    # Transform the input text using the fitted vectorizer
    text_tfidf = fitted_vectorizer.transform([text])
    # Predict the category ID
    predicted_class_id = m.predict(text_tfidf)[0]
    # Map the category ID to the category label
    predicted_class_label = id_to_category[predicted_class_id]
    return predicted_class_label


if __name__ == "__main__":
    train_model()
