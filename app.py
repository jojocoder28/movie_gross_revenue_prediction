# streamlit_app.py

import streamlit as st
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
# Configure GenAI API
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Title and description
st.title("Movie Gross Revenue Predictor")
st.write("This app predicts the gross revenue of a movie based on various features like genre, director, etc.")

# Load data
def load_data():
    df = pd.read_excel('./Movie_Dataset_Competition.xlsx')
    df['release_month'] = df['released'].apply(lambda x: x.split()[0])  # Extract release month
    df.drop(columns=['name', 'released'], inplace=True)  # Drop unnecessary columns
    df.fillna(0, inplace=True)
    return df

df = load_data()
new_df=df.copy()

from sklearn.cluster import KMeans

# Normalize Gross Revenue for better clustering
df['Gross Revenue (Normalized)'] = (df['Gross Revenue'] - df['Gross Revenue'].mean()) / df['Gross Revenue'].std()

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Revenue Cluster'] = kmeans.fit_predict(df[['Gross Revenue (Normalized)']])

# Label encoding for categorical columns (with "Unknown" class)
categorical_columns = ['release_month', 'genre', 'director', 'writer', 'star', 'country', 'company', 'rating']
label_encoders = {}

for col in categorical_columns:
    df[col] = df[col].astype(str)  # Ensure all values are strings before encoding
    label_encoders[col] = LabelEncoder()
    unique_classes = list(df[col].unique()) + ['Unknown']  # Add 'Unknown' to handle unseen labels
    label_encoders[col].fit(unique_classes)
    df[col] = label_encoders[col].transform(df[col])

X = df.drop(columns=['Gross Revenue','Gross Revenue (Normalized)','Revenue Cluster'])
y = df['Revenue Cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sidebar for user inputs
st.sidebar.header("Enter Movie Features:")
release_month = st.sidebar.selectbox("Release Month", new_df['release_month'].unique())
genre = st.sidebar.selectbox("Genre", new_df['genre'].unique())
director = st.sidebar.selectbox("Director", new_df['director'].unique())
writer = st.sidebar.selectbox("Writer", new_df['writer'].unique())
star = st.sidebar.selectbox("Star", new_df['star'].unique())
country = st.sidebar.selectbox("Country", new_df['country'].unique())
company = st.sidebar.selectbox("Company", new_df['company'].unique())
rating = st.sidebar.selectbox("Rating", new_df['rating'].unique())
runtime = st.sidebar.slider("Runtime (in minutes)", 60, 200, 120)
votes = st.sidebar.slider("Votes (in thousands)", 100, 100000, 5000)
budget = st.sidebar.number_input("Budget (in millions)", min_value=1, value=10000000)
year = st.sidebar.number_input("Year", min_value=1900, max_value=2025, value=2020)  # Adding 'year' feature

# Function to encode user input and handle unseen labels
def encode_value(col, value):
    if col in label_encoders:
        if value in label_encoders[col].classes_:
            return label_encoders[col].transform([value])[0]
        else:
            return label_encoders[col].transform(['Unknown'])[0]  # Map unseen labels to 'Unknown'
    return value
save_data = pd.DataFrame({
    'release_month': [release_month],
    'genre': [genre],
    'director': [director],
    'writer': [writer],
    'star': [star],
    'country': [country],
    'company': [company],
    'rating': [rating],
    'runtime': [runtime],
    'votes': [votes],
    'budget': [budget],
    'year': [year]  # Include 'year' in the input data
})
# Encode the user input
input_data = pd.DataFrame({
    'release_month': [encode_value('release_month', release_month)],
    'genre': [encode_value('genre', genre)],
    'director': [encode_value('director', director)],
    'writer': [encode_value('writer', writer)],
    'star': [encode_value('star', star)],
    'country': [encode_value('country', country)],
    'company': [encode_value('company', company)],
    'rating': [encode_value('rating', rating)],
    'runtime': [runtime],
    'votes': [votes],
    'budget': [budget],
    'year': [year]  # Include 'year' in the input data
})



# Make sure input data columns are in the same order as training data columns
input_data = input_data[X.columns]

# Predict the revenue and display it
if st.sidebar.button("Predict Revenue"):
    prediction = model.predict(input_data)[0]
    if prediction == 0:
        classpred ='Low'
    elif prediction == 1:
        classpred = 'Medium'
    else:
        classpred = 'High'
    st.success(f"The predicted Gross Revenue is {classpred}")
    feedback_prompt = f"This is a gross revenue prediction task from movie. Provide business recommendations based on the following data: {save_data} with the output being Predicted revenue: {round(prediction, 2)} "
    feedback_response = gemini_model.generate_content(feedback_prompt)
    feedback = feedback_response.text
    st.subheader("AI-enabled Feedback & Recommendation")
    st.write(feedback)

# Display evaluation metrics on test data
# y_pred = model.predict(X_test)
# st.subheader("Model Evaluation Metrics:")
# st.write(f"**Mean Absolute Error:** {mean_absolute_error(y_test, y_pred):.2f}")
# st.write(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.2f}")
# st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")
