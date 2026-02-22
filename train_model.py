import pandas as pd
import pickle
import os, sys
sys.path.insert(0, 'src')
from preprocess import preprocess
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

# Load Data
df = pd.read_csv('data/spam.csv', encoding='latin-1')
df = df[['v1','v2']].rename(columns={'v1':'label','v2':'text'})
df['clean'] = df['text'].apply(preprocess)
df['label_num'] = (df['label'] == 'spam').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label_num'],
    test_size=0.2, random_state=42, stratify=df['label_num']
)

# TF-IDF
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)

# Models
models = {
    "Naive Bayes":         MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=5),
    "SVM":                 LinearSVC(C=1.0),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
}

print("=" * 55)
print("         MODEL COMPARISON RESULTS")
print("=" * 55)

results = {}
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    cv = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='f1').mean()
    print(f"\n{name}  (CV F1: {cv:.3f})")
    print(classification_report(y_test, preds, target_names=['ham','spam']))
    results[name] = {'model': model, 'cv_f1': cv}

# Save best model
os.makedirs('model', exist_ok=True)
best = max(results, key=lambda k: results[k]['cv_f1'])
print(f"\n✓ Best model: {best}")
pickle.dump(results[best]['model'], open('model/spam_model.pkl','wb'))
pickle.dump(tfidf, open('model/vectorizer.pkl','wb'))
print("✓ Model saved!")
