import os
import csv
import re
import hashlib
import html

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BOOKS_FILE = "books.csv"
USERS_FILE = "users.csv"

st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")

st.markdown("""
<style>
.main-title{font-size:42px;font-weight:800;margin:0}
.subtitle{color:#6b7280;font-size:16px;margin:4px 0 18px}
.hero{padding:24px;border-radius:18px;background:linear-gradient(135deg,#f7f2ff,#eef7ff);border:1px solid #e5e7eb;margin-bottom:18px}
.profile-box{padding:16px;border-radius:14px;background:#f7f8fa;border:1px solid #e7e9ed;margin:10px 0 18px}
.section-title{font-size:25px;font-weight:750;margin:14px 0 5px}
.book-card{border:1px solid #e5e7eb;border-radius:16px;padding:18px;margin:12px 0;background:white;box-shadow:0 3px 12px rgba(0,0,0,.045)}
.book-title{font-size:21px;font-weight:750}
.book-author{color:#666;margin:3px 0 10px}
.tag{display:inline-block;padding:4px 9px;border-radius:20px;background:#f0f1f3;margin:2px 4px 2px 0;font-size:13px}
.reason{background:#f7f8fa;border-radius:11px;padding:12px;margin-top:12px;line-height:1.5}
.evidence{font-size:13px;color:#666;margin-top:8px}
.small{font-size:13px;color:#777}
</style>
""", unsafe_allow_html=True)

# ----------------------------- AUTH -----------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def load_users():
    if not os.path.exists(USERS_FILE):
        pd.DataFrame(columns=["username", "password_hash"]).to_csv(USERS_FILE, index=False)
    try:
        df = pd.read_csv(USERS_FILE)
        if {"username", "password_hash"}.issubset(df.columns):
            return df
    except Exception:
        pass
    pd.DataFrame(columns=["username", "password_hash"]).to_csv(USERS_FILE, index=False)
    return pd.DataFrame(columns=["username", "password_hash"])


def register(username, password):
    username = username.strip()
    if not username or not password:
        return False, "Username and password are required."
    users = load_users()
    if not users.empty and username.lower() in users["username"].astype(str).str.lower().values:
        return False, "Username already exists."
    pd.DataFrame([[username, hash_password(password)]], columns=["username", "password_hash"]).to_csv(
        USERS_FILE, mode="a", header=not os.path.exists(USERS_FILE), index=False
    )
    return True, "Registration successful."


def authenticate(username, password):
    wanted = hash_password(password)
    return any(
        str(r["username"]).strip().lower() == username.strip().lower()
        and str(r["password_hash"]).strip() == wanted
        for _, r in load_users().iterrows()
    )

# ----------------------------- BOOK DATA -----------------------------
def load_books():
    if not os.path.exists(BOOKS_FILE):
        st.error("books.csv was not found in the same folder as app.py.")
        st.stop()

    rows = []
    try:
        with open(BOOKS_FILE, encoding="utf-8-sig", newline="") as f:
            for n, row in enumerate(csv.reader(f)):
                if n == 0 or len(row) < 5:
                    continue
                title = ",".join(row[:-4]).strip()
                author, genre, price, description = [x.strip() for x in row[-4:]]
                try:
                    price = float(price.replace("₹", "").replace(",", "").strip())
                except (ValueError, AttributeError):
                    continue
                rows.append([title, author, genre, price, description])
    except Exception as e:
        st.error(f"Could not read books.csv: {e}")
        st.stop()

    df = pd.DataFrame(rows, columns=["title", "author", "genre", "price", "description"])
    if df.empty:
        st.error("No valid books were found in books.csv.")
        st.stop()
    return df

# ----------------------------- RECOMMENDATION KNOWLEDGE -----------------------------
# A recommendation needs meaningful content evidence. Weak generic words
# are kept as supporting evidence, not as the only reason for a match.
INTEREST_PROFILES = {
    "AI": {
        "core": ["artificial intelligence", "machine learning", "deep learning", "neural network", "computer vision", "natural language processing", "robotics", "generative ai", "large language model"],
        "support": ["automation", "algorithm", "algorithms", "data science", "intelligent systems", "prediction", "training data", "ai models", "technology"],
        "profile": "artificial intelligence machine learning deep learning neural networks computer vision natural language processing robotics generative AI algorithms data science intelligent systems"
    },
    "Programming": {
        "core": ["programming", "python", "java", "javascript", "software development", "software engineering", "coding", "computer programming", "data structures", "algorithms", "web development"],
        "support": ["developer", "developers", "programmer", "programmers", "debugging", "database", "source code", "application development", "computer science"],
        "profile": "programming coding Python Java JavaScript software development software engineering algorithms data structures debugging developers computer science application development databases web development"
    },
    "Science": {
        "core": ["science", "scientific", "physics", "biology", "chemistry", "experiment", "experiments", "scientist", "scientists"],
        "support": ["research", "laboratory", "discovery", "evolution", "genetics", "matter", "energy", "nature"],
        "profile": "science scientific physics biology chemistry experiments scientists research laboratory discovery evolution genetics matter energy nature"
    },
    "Space": {
        "core": ["space", "astronomy", "astronaut", "astronauts", "universe", "galaxy", "galaxies", "planet", "planets", "cosmos", "cosmology"],
        "support": ["solar system", "stars", "star", "black hole", "black holes", "nasa", "orbit", "rocket", "mars", "moon"],
        "profile": "space astronomy astronauts universe galaxies planets cosmos cosmology solar system stars black holes NASA orbit rockets Mars Moon"
    },
    "Finance": {
        "core": ["finance", "financial", "money", "wealth", "investment", "investing", "investor", "investors", "personal finance"],
        "support": ["stock market", "stocks", "business", "economy", "economics", "saving", "savings", "budgeting", "assets", "income"],
        "profile": "finance money wealth investing investments investors personal finance financial decisions stocks stock market business economics saving budgeting assets income"
    },
    "Psychology": {
        "core": ["psychology", "psychological", "human behaviour", "human behavior", "cognitive", "cognition", "human mind", "mental processes"],
        "support": ["mind", "memory", "emotion", "emotions", "personality", "decision making", "social behaviour", "social behavior", "brain"],
        "profile": "psychology psychological human behavior human behaviour cognitive cognition mind memory emotions personality decision making social behavior brain mental processes"
    },
    "Romance": {
        "core": ["romance", "romantic", "love story", "romantic relationship", "romantic relationships"],
        "support": ["love", "relationship", "relationships", "couple", "couples", "marriage", "dating", "attraction", "heartbreak", "passion"],
        "profile": "romance romantic love stories relationships couples marriage dating attraction heartbreak passion romantic characters"
    },
    "Love": {
        "core": ["love", "love story", "love stories", "romance", "romantic relationship", "romantic relationships"],
        "support": ["relationship", "relationships", "couple", "couples", "marriage", "dating", "affection", "friendship", "family"],
        "profile": "love romance romantic relationships couples marriage dating affection friendship family emotional connections"
    },
    "Self Help": {
        "core": ["self help", "self-help", "personal growth", "personal development", "self improvement", "self-improvement", "motivation", "mindset"],
        "support": ["habits", "confidence", "discipline", "goals", "life advice", "success", "wellbeing", "growth"],
        "profile": "self help personal growth personal development self improvement motivation mindset habits confidence discipline goals life advice success wellbeing growth"
    },
    "Productivity": {
        "core": ["productivity", "time management", "focus", "deep work", "work habits", "organization"],
        "support": ["habits", "goals", "planning", "efficiency", "concentration", "workflow", "priorities", "routine"],
        "profile": "productivity time management focus deep work habits planning efficiency concentration workflow organization goals priorities routine work"
    },
    "Creativity": {
        "core": ["creativity", "creative", "creative thinking", "creative process", "artist", "artists", "design", "creative work"],
        "support": ["art", "ideas", "imagination", "innovation", "inspiration", "creator", "creators", "originality", "expression"],
        "profile": "creativity creative thinking creative process artists art design ideas imagination innovation inspiration creators originality expression creative work"
    },
    "History": {
        "core": ["history", "historical", "civilization", "civilizations", "ancient history", "world history"],
        "support": ["empire", "war", "wars", "culture", "revolution", "past", "historical events", "society"],
        "profile": "history historical civilizations ancient history world history empires wars culture revolution historical events society past"
    },
    "Mystery": {
        "core": ["mystery", "detective", "investigation", "investigator", "crime mystery", "murder mystery"],
        "support": ["crime", "murder", "clues", "suspect", "suspects", "case", "secrets", "hidden truth"],
        "profile": "mystery detective investigation investigators crime murder clues suspects cases secrets hidden truth mystery story"
    },
    "Thriller": {
        "core": ["thriller", "suspense", "psychological thriller", "crime thriller", "mystery thriller"],
        "support": ["danger", "murder", "crime", "investigation", "chase", "threat", "survival", "tension", "secrets"],
        "profile": "thriller suspense psychological thriller crime danger murder investigation chase threat survival tension secrets fast paced story"
    },
    "Children": {
        "core": ["children", "child", "kids", "children's", "young readers", "middle grade"],
        "support": ["family", "school", "adventure", "friendship", "fairy tale", "stories", "young"],
        "profile": "children kids young readers middle grade family school adventure friendship fairy tales stories young readers"
    }
}

# ----------------------------- TEXT / SCORING -----------------------------
def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(text).lower())).strip()


def contains(term, text):
    term, text = normalize(term), normalize(text)
    return bool(term) and (term in text if " " in term else term in text.split())


def book_text(row):
    return " ".join(str(row[c]) for c in ["title", "author", "genre", "description"])


def description_text(row):
    return normalize(row["description"])


def hits(terms, text):
    return [term for term in terms if contains(term, text)]


def interest_match(row, interest, vectorizer, matrix, positions):
    p = INTEREST_PROFILES[interest]
    desc = description_text(row)
    full = normalize(book_text(row))
    core_desc = hits(p["core"], desc)
    support_desc = hits(p["support"], desc)
    core_full = hits(p["core"], full)
    support_full = hits(p["support"], full)

    q = vectorizer.transform([p["profile"]])
    sim = float(cosine_similarity(q, matrix[positions[row.name]])[0][0]) * 100

    # Description evidence is deliberately much stronger than title/author evidence.
    score = min(55, len(core_desc) * 25) + min(20, len(support_desc) * 6)
    score += min(15, len(core_full) * 5)
    score += min(10, len(support_full) * 2)
    score += min(15, sim * 0.30)
    score = min(100, score)

    # A book must have actual content evidence. This prevents random matches
    # such as a thriller containing the word "code" from becoming Programming.
    genuine = bool(core_desc) or len(support_desc) >= 2 or (bool(core_full) and sim >= 22 and len(core_full) >= 2)

    # Programming is intentionally stricter because words like "code", "software"
    # and "development" can occur incidentally in unrelated descriptions.
    if interest == "Programming":
        strong = [
            "programming", "python", "java", "javascript", "software development",
            "software engineering", "coding", "computer programming", "data structures",
            "algorithms", "web development"
        ]
        genuine = bool(hits(strong, desc)) or (len(hits(strong, full)) >= 2 and sim >= 22)

    return {
        "match": genuine and score >= 25,
        "score": round(score, 1),
        "similarity": round(sim, 1),
        "core_desc": core_desc,
        "support_desc": support_desc,
        "core_full": core_full,
        "support_full": support_full
    }


def build_similarity(df, favourite):
    texts = df.apply(book_text, axis=1).tolist()
    if len(texts) < 2:
        return {}
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    fav_rows = df.index[df["title"].astype(str).str.lower() == favourite.lower()]
    if len(fav_rows) == 0:
        return {}
    pos = df.index.get_loc(fav_rows[0])
    scores = cosine_similarity(matrix[pos], matrix)[0]
    return {idx: round(float(scores[i]) * 100, 1) for i, idx in enumerate(df.index)}

# ----------------------------- RECOMMENDATION ENGINE -----------------------------
def recommend(df, interests, genre, favourite, budget):
    # The UI requires at least one interest, but keep the engine safe if it is
    # ever called directly with an empty list.
    if not interests:
        return {
            "overall": [],
            "per_interest": {},
            "favourite": [],
            "genre": [],
            "budget": []
        }

    affordable = df[df["price"] <= budget].copy()
    affordable = affordable[affordable["title"].astype(str).str.lower() != favourite.lower()].copy()
    if affordable.empty:
        return {"overall": [], "per_interest": {i: [] for i in interests}, "favourite": [], "genre": [], "budget": []}

    # One shared content space keeps every score comparable.
    texts = df.apply(book_text, axis=1).tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    positions = {idx: pos for pos, idx in enumerate(df.index)}
    fav_scores = build_similarity(df, favourite)

    items = []
    for _, row in affordable.iterrows():
        per_interest = {i: interest_match(row, i, vectorizer, matrix, positions) for i in interests}
        matched = [i for i, x in per_interest.items() if x["match"]]
        best_interest = max((x["score"] for x in per_interest.values() if x["match"]), default=0)
        coverage = len(matched) / max(1, len(interests)) * 100
        fav = fav_scores.get(row.name, 0)
        genre_match = normalize(row["genre"]) == normalize(genre)

        # Overall recommendation rewards genuine interest coverage, then content
        # quality, favourite similarity and genre fit. Price is only a hard filter.
        overall = coverage * 0.42 + best_interest * 0.28 + fav * 0.18 + (100 if genre_match else 0) * 0.12

        items.append({
            "row": row,
            "interest": per_interest,
            "matched": matched,
            "coverage": coverage,
            "fav": fav,
            "genre": genre_match,
            "overall": overall
        })

    per_interest = {}
    for interest in interests:
        candidates = [x for x in items if x["interest"][interest]["match"]]
        candidates.sort(key=lambda x: (x["interest"][interest]["score"], x["fav"], x["genre"]), reverse=True)
        per_interest[interest] = candidates[:3]

    favourite = [x for x in items if x["fav"] >= 20]
    favourite.sort(key=lambda x: x["fav"], reverse=True)

    genre_books = [x for x in items if x["genre"]]
    genre_books.sort(key=lambda x: (x["overall"], x["fav"]), reverse=True)

    # Budget is a filter, not a recommendation reason.  Prefer books that
    # genuinely match a selected interest; favourite/genre matches can appear
    # only after those relevant books.
    budget_books = [x for x in items if x["matched"] or x["fav"] >= 20 or x["genre"]]
    budget_books.sort(
        key=lambda x: (
            bool(x["matched"]),
            len(x["matched"]),
            x["overall"],
            x["fav"],
            x["genre"]
        ),
        reverse=True
    )

    return {
        "per_interest": per_interest,
        "favourite": favourite[:8],
        "genre": genre_books[:8],
        "budget": budget_books[:8]
    }

# ----------------------------- REASONS / UI -----------------------------
def interest_reason(item, interest):
    x = item["interest"][interest]
    if x["core_desc"]:
        evidence = ", ".join(x["core_desc"][:3])
        return f"The book's description directly discusses {evidence}, which is a strong content-level match for {interest}."
    if x["support_desc"]:
        evidence = ", ".join(x["support_desc"][:3])
        return f"The book's description contains related concepts such as {evidence}, giving it a genuine connection to {interest}."
    evidence = ", ".join(x["core_full"][:3])
    return f"The book information contains multiple {interest}-related concepts ({evidence}) and its overall text is also similar to the {interest} topic profile."


def favourite_reason(item, favourite):
    return f"It has {item['fav']:.0f}% content similarity with '{favourite}' based on the available title, author, genre and description text."


def show_book(item, reason, evidence=None):
    row = item["row"]
    tags = []
    if item["matched"]:
        tags.append("🎯 " + ", ".join(item["matched"]))
    if item["genre"]:
        tags.append("📚 Preferred genre")
    if item["fav"] >= 20:
        tags.append(f"❤️ Favourite similarity: {item['fav']:.0f}%")
    tag_html = "".join(f'<span class="tag">{html.escape(t)}</span>' for t in tags)
    evidence_html = f'<div class="evidence">Evidence: {html.escape(evidence)}</div>' if evidence else ""

    st.markdown(f"""
    <div class="book-card">
      <div class="book-title">📖 {html.escape(str(row['title']))}</div>
      <div class="book-author">by {html.escape(str(row['author']))}</div>
      <b>🏷️ {html.escape(str(row['genre']))}</b> &nbsp;&nbsp; <b>💰 ₹{row['price']:.0f}</b>
      <div style="margin:8px 0">{tag_html}</div>
      <div class="small">{html.escape(str(row['description']))}</div>
      <div class="reason"><b>Why recommended?</b><br>{html.escape(reason)}{evidence_html}</div>
    </div>
    """, unsafe_allow_html=True)


def search_books(df, query):
    q = normalize(query)
    if not q:
        return df
    mask = (
        df["title"].map(normalize).str.contains(q, regex=False)
        | df["author"].map(normalize).str.contains(q, regex=False)
        | df["genre"].map(normalize).str.contains(q, regex=False)
        | df["description"].map(normalize).str.contains(q, regex=False)
    )
    return df[mask]

# ----------------------------- LOGIN -----------------------------
def login_page():
    st.markdown('<div class="main-title">📚 Book Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Find books based on what you actually like.</div>', unsafe_allow_html=True)

    login, register_tab = st.tabs(["🔐 Login", "📝 Register"])
    with login:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", type="primary", use_container_width=True):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with register_tab:
        username = st.text_input("Create username", key="reg_user")
        password = st.text_input("Create password", type="password", key="reg_pass")
        confirm = st.text_input("Confirm password", type="password", key="reg_confirm")
        if st.button("Create Account", use_container_width=True):
            if password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, msg = register(username, password)
                (st.success if ok else st.error)(msg)

# ----------------------------- MAIN APP -----------------------------
def main_app():
    books = load_books()

    st.markdown("""
    <div class="hero">
      <div class="main-title">📚 Book Recommender</div>
      <div class="subtitle">Tell us what you like, your favourite book, genre and budget — then get clear, evidence-based recommendations.</div>
    </div>
    """, unsafe_allow_html=True)

    a, b = st.columns([5, 1])
    with a:
        st.success(f"Welcome back, **{st.session_state.username}** 👋")
    with b:
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    st.subheader("🎯 Build your reading profile")
    c1, c2 = st.columns(2)
    with c1:
        interests = st.multiselect("Your interests", list(INTEREST_PROFILES), help="Select one or more interests. Each selected interest gets its own recommendation section.")
        genres = sorted(books["genre"].dropna().astype(str).unique().tolist())
        genre = st.selectbox("Preferred genre", genres)
    with c2:
        favourite = st.selectbox("Favourite book", books["title"].astype(str).tolist())
        budget = st.number_input("Maximum budget (₹)", min_value=1, max_value=10000, value=499, step=50)

    st.caption("💡 Budget is a hard limit. A book above your maximum price is never recommended.")

    if st.button("✨ Find My Books", type="primary", use_container_width=True):
        if not interests:
            st.warning("Please select at least one interest.")
        else:
            st.session_state.results = recommend(books, interests, genre, favourite, budget)
            st.session_state.profile = {"interests": interests, "genre": genre, "favourite": favourite, "budget": budget}

    if "results" not in st.session_state:
        st.info("Choose your preferences and click **✨ Find My Books**.")
        return

    results = st.session_state.results
    profile = st.session_state.profile
    interests, genre = profile["interests"], profile["genre"]
    favourite, budget = profile["favourite"], profile["budget"]

    st.divider()
    st.subheader("📋 Your profile")
    st.markdown(f"""
    <div class="profile-box">
      <b>INTERESTS:</b> {html.escape(', '.join(interests))} &nbsp; | &nbsp;
      <b>GENRE:</b> {html.escape(genre)} &nbsp; | &nbsp;
      <b>FAVOURITE:</b> {html.escape(favourite)} &nbsp; | &nbsp;
      <b>MAX BUDGET:</b> ₹{budget:.0f}
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["🎯 Each Interest", "❤️ Favourite", "📚 Genre", "💰 Budget"])

    with tabs[0]:
        st.markdown('<div class="section-title">🎯 One recommendation set for every interest</div>', unsafe_allow_html=True)
        st.caption("Each interest is evaluated separately. A book is shown only when its content provides enough evidence for that interest.")
        for interest in interests:
            st.markdown(f"### {interest}")
            candidates = results["per_interest"].get(interest, [])
            if not candidates:
                st.info(f"No strong **{interest}** match was found within ₹{budget:.0f}.")
            else:
                for item in candidates:
                    x = item["interest"][interest]
                    evidence = ", ".join((x["core_desc"] or x["support_desc"] or x["core_full"])[:4])
                    show_book(item, interest_reason(item, interest), evidence)

    with tabs[1]:
        st.markdown(f'<div class="section-title">❤️ Similar to {html.escape(favourite)}</div>', unsafe_allow_html=True)
        st.caption("Similarity is calculated from the available book content, not from price alone.")
        if not results["favourite"]:
            st.info("No sufficiently similar book was found within your budget.")
        else:
            for item in results["favourite"]:
                show_book(item, favourite_reason(item, favourite))

    with tabs[2]:
        st.markdown(f'<div class="section-title">📚 More {html.escape(genre)} books</div>', unsafe_allow_html=True)
        if not results["genre"]:
            st.info(f"No {genre} book is available within ₹{budget:.0f}.")
        else:
            for item in results["genre"]:
                show_book(item, f"It belongs to your preferred genre '{genre}'.")

    with tabs[3]:
        st.markdown(f'<div class="section-title">💰 Relevant books under ₹{budget:.0f}</div>', unsafe_allow_html=True)
        st.caption("Only books relevant to at least one part of your profile are shown; cheap but irrelevant books are not promoted.")
        if not results["budget"]:
            st.info("No relevant book was found within this budget.")
        else:
            for item in results["budget"]:
                reason_parts = []
                if item["matched"]:
                    reason_parts.append("matches " + ", ".join(item["matched"][:3]))
                if item["genre"]:
                    reason_parts.append(f"fits your preferred genre '{genre}'")
                if item["fav"] >= 20:
                    reason_parts.append(f"has {item['fav']:.0f}% content similarity with '{favourite}'")
                reason = "Recommended because it " + "; it ".join(reason_parts) + "." if reason_parts else "It is relevant to your profile and within your budget."
                show_book(item, reason)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login_page()
