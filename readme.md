# 📚 BookFinder

BookFinder is a personalized book recommendation web application built using Python, Streamlit, Pandas and Scikit-learn.

The system recommends books based on the user's interests, preferred genre, favourite book and maximum budget.

## Features

- User Registration and Login
- Secure password hashing
- Interest-based book recommendations
- Genre-based recommendations
- Favourite-book based recommendations
- Budget filtering
- AI-based similarity matching
- Match percentage
- Recommendation reason
- Book information from CSV dataset
- Simple and interactive Streamlit interface

## AI / NLP Technique

BookFinder uses Natural Language Processing for recommendations.

### TF-IDF

TF-IDF converts text information such as book titles, genres and descriptions into numerical vectors.

### Cosine Similarity

Cosine Similarity compares the user's interests with the book information and calculates how similar they are.

Books with higher similarity are ranked higher in the recommendations.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| Pandas | CSV and data handling |
| Scikit-learn | TF-IDF and Cosine Similarity |
| CSV | Book and user data storage |
| GitHub | Version control |
| Streamlit Community Cloud | Deployment |

## Project Structure

```text
BookFinder/
│
├── app.py
├── books.csv
├── users.csv
├── requirements.txt
└── README.md