# 📚 Book Recommender

A personalized book recommendation web application built using **Python, Streamlit, Pandas and Scikit-learn**.
It recommends books based on the user's **interests, favourite book, preferred genre and budget**.

## Features
- User Registration and Login
- Interest-based recommendations
- Favourite-book recommendations
- Genre-based recommendations
- Budget filtering
- Recommendation reasons

## Recommendation Logic

The system uses **content-based recommendation techniques** based on four user inputs:

- **Interests:** Matches selected interests with relevant concepts in book descriptions using predefined interest profiles and TF-IDF similarity.
- **Favourite Book:** Uses TF-IDF and Cosine Similarity to find books with similar title, author, genre and description content.
- **Genre:** Matches books with the user's selected preferred genre.
- **Budget:** Applies a hard price limit and excludes books above the user's maximum budget.

The system combines these factors to generate personalized book recommendations and explains the content-based reason for each recommendation.

## Technology

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF
- Cosine Similarity
- CSV