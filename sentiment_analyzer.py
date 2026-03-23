import re
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from collections import Counter

warnings.filterwarnings('ignore')

REAL_DATA_PATH = 'reviews.csv'   # set to your CSV if you have one

# ─────────────────────────────────────────────────────────────────────────────
# 0. ENGLISH STOPWORDS (built-in — no NLTK required)
# ─────────────────────────────────────────────────────────────────────────────
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than',
    'too','very','s','t','can','will','just','don','should','now','d','ll',
    'm','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn',
    'weren','won','wouldn','also','would','could','get','got','much','many',
    'even','still','though','although','however','yet','since','whether','else',
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_reviews(n_each=1000, seed=42):
    """Generate labelled movie reviews with realistic vocabulary."""
    rng = np.random.default_rng(seed)

    pos_phrases = [
        "absolutely brilliant film", "outstanding performance by the lead actor",
        "beautifully crafted story", "masterpiece of modern cinema",
        "kept me on the edge of my seat", "emotionally powerful and moving",
        "visually stunning with great cinematography", "one of the best films I have seen",
        "heartwarming and genuinely touching", "superb direction and writing",
        "the acting was phenomenal throughout", "a joy to watch from start to finish",
        "engaging and thought-provoking", "a must-watch for everyone",
        "incredibly well-paced and entertaining", "the plot twists were spectacular",
        "deeply moving and memorable experience", "excellent script and direction",
        "brought tears to my eyes in the best way", "a triumph of storytelling",
        "the chemistry between the actors was perfect", "gripping narrative",
        "wonderful performances all around", "highly recommend this movie",
        "creative and original story", "powerful message delivered brilliantly",
        "a rare gem in today's cinema", "outstanding soundtrack as well",
        "left me speechless and amazed", "exceeded all my expectations",
        "the best film of the year without doubt", "tremendously entertaining",
        "a delightful experience for the whole family", "inspiring and uplifting",
        "flawlessly executed in every department", "riveting from beginning to end",
        "genuinely funny and clever writing", "a beautiful piece of art",
        "captivating story with depth and heart", "the director's best work yet",
    ]
    neg_phrases = [
        "complete waste of time and money", "terrible acting throughout",
        "the plot made absolutely no sense", "one of the worst films ever made",
        "painfully boring and slow-paced", "the script was badly written",
        "wooden performances from the entire cast", "deeply disappointing film",
        "I nearly fell asleep watching it", "a predictable and cliché story",
        "the special effects were laughably bad", "could not connect with any character",
        "poorly directed mess of a film", "two hours I will never get back",
        "the dialogue was cringe-worthy and awful", "fell apart completely in the third act",
        "a soulless cash-grab of a movie", "no originality whatsoever",
        "incredibly frustrating and unsatisfying", "the editing was chaotic and confusing",
        "do not waste your time with this one", "an insult to good filmmaking",
        "the worst screenplay I have read", "hollow and meaningless film",
        "zero chemistry between the leads", "a confusing and incoherent mess",
        "nothing worked in this film at all", "the worst I have seen this year",
        "every scene felt forced and fake", "no redeeming qualities at all",
        "a dull and forgettable experience", "extremely poor production quality",
        "the story goes nowhere", "unconvincing and lazy performances",
        "not a single laugh in supposed comedy", "a deeply flawed and boring film",
        "the director clearly had no vision", "completely unlikeable characters",
        "regret watching this terrible film", "disappointing on every level possible",
    ]
    connectors = [
        "and also", "furthermore", "in addition", "I felt that", "I thought",
        "honestly", "to be honest", "in my opinion", "clearly", "surprisingly",
        "the story was", "the cast was", "the film was",
    ]

    reviews, labels = [], []
    for label, phrases in [(1, pos_phrases), (0, neg_phrases)]:
        for _ in range(n_each):
            n_phrases = rng.integers(2, 5)
            selected = rng.choice(phrases, n_phrases, replace=True).tolist()
            conn = rng.choice(connectors, n_phrases - 1).tolist()
            parts = [selected[0]]
            for c, p in zip(conn, selected[1:]):
                parts.append(c + " " + p)
            review = ". ".join(parts) + "."
            # add minor noise
            if rng.random() < 0.3:
                filler = rng.choice([
                    "Watched it last night.",
                    "Saw this with my family.",
                    "Would definitely recommend.",
                    "Would not recommend at all.",
                    "Overall my thoughts:",
                ])
                review = filler + " " + review
            reviews.append(review)
            labels.append(label)

    df = pd.DataFrame({'review': reviews, 'sentiment': labels})
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. NLP PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text):
    """
    NLP preprocessing pipeline:
      1. Lowercase
      2. Remove HTML tags
      3. Remove punctuation and special characters
      4. Remove digits
      5. Tokenise (split on whitespace)
      6. Remove stopwords
      7. Remove short tokens (length <= 2)
      8. Rejoin to string
    """
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)          # remove HTML tags
    text = re.sub(r'[^a-z\s]', ' ', text)         # remove non-alpha chars
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  Movie Review Sentiment Analyzer")
print("  CSA2001 – Fundamentals of AI and ML | BYOP")
print("  Aadi Jain | 25BCE10331 | VIT Bhopal")
print("=" * 65)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
if os.path.exists(REAL_DATA_PATH):
    df = pd.read_csv(REAL_DATA_PATH)
    df.columns = ['review', 'sentiment']
    print(f"\n[INFO] Loaded real dataset: {REAL_DATA_PATH}")
else:
    df = generate_reviews(n_each=1000)
    df.to_csv(REAL_DATA_PATH, index=False)
    print("\n[INFO] Generated synthetic dataset (2000 labelled reviews).")

print(f"\nDataset shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Class balance : {df['sentiment'].value_counts().to_dict()}")
print(f"\nSample reviews:")
for i in range(3):
    label = 'POSITIVE' if df.iloc[i]['sentiment'] == 1 else 'NEGATIVE'
    print(f"  [{label}] {df.iloc[i]['review'][:90]}...")

# ── EDA ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("STEP 1: Exploratory Data Analysis")
print("─" * 65)

df['review_length']  = df['review'].apply(lambda x: len(x.split()))
df['char_count']     = df['review'].apply(len)

print(f"\nAverage review length : {df['review_length'].mean():.1f} words")
print(f"Max review length     : {df['review_length'].max()} words")
print(f"Min review length     : {df['review_length'].min()} words")

os.makedirs('plots', exist_ok=True)

# Plot 1 – Class distribution + review length distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Exploratory Data Analysis', fontsize=14, fontweight='bold')

# Class balance
counts = df['sentiment'].value_counts()
bars = axes[0].bar(['Negative', 'Positive'], [counts[0], counts[1]],
                   color=['#e05252', '#4a90d9'], alpha=0.85, width=0.5)
axes[0].set_title('Class Balance', fontsize=12)
axes[0].set_ylabel('Number of Reviews')
for b, v in zip(bars, [counts[0], counts[1]]):
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                 str(v), ha='center', fontsize=11)

# Review length histogram
axes[1].hist(df[df['sentiment']==1]['review_length'], bins=25,
             alpha=0.65, color='#4a90d9', label='Positive')
axes[1].hist(df[df['sentiment']==0]['review_length'], bins=25,
             alpha=0.65, color='#e05252', label='Negative')
axes[1].set_title('Review Length Distribution', fontsize=12)
axes[1].set_xlabel('Number of Words')
axes[1].set_ylabel('Frequency')
axes[1].legend()

# Top words before preprocessing
all_words = ' '.join(df['review']).split()
word_freq = Counter(all_words).most_common(15)
words, freqs = zip(*word_freq)
axes[2].barh(words[::-1], freqs[::-1], color='#4a90d9', alpha=0.8)
axes[2].set_title('Top 15 Words (Before Preprocessing)', fontsize=12)
axes[2].set_xlabel('Frequency')

plt.tight_layout()
plt.savefig('plots/01_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] plots/01_eda.png")

# ── NLP PREPROCESSING ────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("STEP 2: NLP Preprocessing Pipeline")
print("─" * 65)

df['cleaned'] = df['review'].apply(clean_text)

print("\nPipeline steps: lowercase → remove HTML → remove punctuation")
print("             → tokenise → remove stopwords → remove short tokens")
print(f"\nExample (raw)    : {df.iloc[0]['review'][:80]}...")
print(f"Example (cleaned): {df.iloc[0]['cleaned'][:80]}...")
print(f"\nStopwords removed : {len(STOPWORDS)} words")

# TF-IDF Vectorisation
print("\n[INFO] Applying TF-IDF Vectorisation...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                              min_df=2, sublinear_tf=True)

X = vectorizer.fit_transform(df['cleaned'])
y = df['sentiment'].values

print(f"Vocabulary size   : {len(vectorizer.vocabulary_)} unique n-grams")
print(f"Feature matrix    : {X.shape[0]} reviews × {X.shape[1]} TF-IDF features")

# Plot 2 – Top words AFTER preprocessing
cleaned_words = ' '.join(df['cleaned']).split()
clean_freq = Counter(cleaned_words).most_common(15)
cwords, cfreqs = zip(*clean_freq)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('NLP Preprocessing: Top Words Before vs After', fontsize=14, fontweight='bold')

raw_words_freq = Counter(' '.join(df['review']).lower().split()).most_common(15)
rwords, rfreqs = zip(*raw_words_freq)
axes[0].barh(list(rwords)[::-1], list(rfreqs)[::-1], color='#e05252', alpha=0.8)
axes[0].set_title('Before Preprocessing', fontsize=12)
axes[0].set_xlabel('Frequency')

axes[1].barh(list(cwords)[::-1], list(cfreqs)[::-1], color='#4a90d9', alpha=0.8)
axes[1].set_title('After Preprocessing (Stopwords Removed)', fontsize=12)
axes[1].set_xlabel('Frequency')

plt.tight_layout()
plt.savefig('plots/02_preprocessing.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] plots/02_preprocessing.png")

# ── TRAIN-TEST SPLIT ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain : {X_train.shape[0]} samples  |  Test : {X_test.shape[0]} samples")

# ── MODEL TRAINING ────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("STEP 3: Training Models")
print("─" * 65)

models = {
    "Naïve Bayes (NLP Baseline)": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42, C=1.0),
    "MLP Neural Network": MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        learning_rate_init=0.001,
        alpha=0.0001,
    )
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n  Training: {name}...")
    model.fit(X_train, y_train)
    preds      = model.predict(X_test)
    proba      = model.predict_proba(X_test)[:, 1]
    acc        = accuracy_score(y_test, preds)
    auc        = roc_auc_score(y_test, proba)
    cv_scores  = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    report     = classification_report(y_test, preds, output_dict=True)
    cm         = confusion_matrix(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, proba)

    results[name] = {
        'model': model, 'preds': preds, 'proba': proba,
        'accuracy': acc, 'auc': auc,
        'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
        'report': report, 'cm': cm, 'fpr': fpr, 'tpr': tpr
    }
    print(f"    Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    AUC-ROC  : {auc:.4f}")
    print(f"    CV Acc   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  (5-fold)")

# ── RESULTS SUMMARY ───────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("MODEL COMPARISON SUMMARY")
print("─" * 65)
print(f"{'Model':<35} {'Accuracy':>10} {'AUC-ROC':>10} {'CV Acc':>12}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:<35} {r['accuracy']:>10.4f} {r['auc']:>10.4f} "
          f"{r['cv_mean']:>7.4f}±{r['cv_std']:.4f}")

best_name = max(results, key=lambda k: results[k]['accuracy'])
print(f"\n✓ Best model: {best_name}  (Accuracy = {results[best_name]['accuracy']*100:.2f}%)")

# ── VISUALISATIONS ────────────────────────────────────────────────────────────

# Plot 3 – Confusion matrices (all 3 models)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
cmap_list = ['Blues', 'Oranges', 'Greens']

for ax, (name, r), cmap in zip(axes, results.items(), cmap_list):
    sns.heatmap(r['cm'], annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                linewidths=0.5, cbar=False,
                annot_kws={'size': 13})
    ax.set_title(name, fontsize=10)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('plots/03_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[SAVED] plots/03_confusion_matrices.png")

# Plot 4 – ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#e05252', '#f5a623', '#4a90d9']
for (name, r), c in zip(results.items(), colors):
    ax.plot(r['fpr'], r['tpr'], color=c, linewidth=2.0,
            label=f"{name}  (AUC = {r['auc']:.3f})")
ax.plot([0,1], [0,1], 'k--', linewidth=1, alpha=0.5, label='Random classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves – All Models', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/04_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] plots/04_roc_curves.png")

# Plot 5 – Accuracy + AUC comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
names  = list(results.keys())
short  = ['Naïve Bayes', 'Logistic\nRegression', 'MLP Neural\nNetwork']
accs   = [r['accuracy'] for r in results.values()]
aucs   = [r['auc']      for r in results.values()]

bars1 = axes[0].bar(short, accs, color=['#e05252','#f5a623','#4a90d9'], alpha=0.85, width=0.5)
axes[0].set_title('Accuracy', fontsize=12)
axes[0].set_ylabel('Score')
axes[0].set_ylim(0.5, 1.05)
for b, v in zip(bars1, accs):
    axes[0].text(b.get_x()+b.get_width()/2, v+0.005, f'{v*100:.1f}%', ha='center', fontsize=11)

bars2 = axes[1].bar(short, aucs, color=['#e05252','#f5a623','#4a90d9'], alpha=0.85, width=0.5)
axes[1].set_title('AUC-ROC Score', fontsize=12)
axes[1].set_ylabel('Score')
axes[1].set_ylim(0.5, 1.05)
for b, v in zip(bars2, aucs):
    axes[1].text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.3f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('plots/05_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] plots/05_model_comparison.png")

# Plot 6 – MLP Neural Network architecture diagram (text-based visual)
mlp   = results["MLP Neural Network"]['model']
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')
fig.suptitle('MLP Neural Network Architecture', fontsize=14, fontweight='bold')

layers = [('Input\n5000 TF-IDF\nfeatures', 5000),
          ('Hidden 1\n256 neurons\nReLU', 256),
          ('Hidden 2\n128 neurons\nReLU', 128),
          ('Hidden 3\n64 neurons\nReLU', 64),
          ('Output\n2 neurons\nSoftmax', 2)]
colors_arch = ['#aad4f5','#7ab8f0','#4a90d9','#2c6fad','#1a4d8c']
x_positions = np.linspace(0.05, 0.95, len(layers))

for x, (label, size), c in zip(x_positions, layers, colors_arch):
    n_dots = min(int(np.log2(size+1))+1, 8)
    for j in range(n_dots):
        y = 0.5 + (j - n_dots/2) * 0.08
        ax.add_patch(plt.Circle((x, y), 0.022, color=c, ec='white', lw=1.5, zorder=3))
    ax.text(x, 0.08, label, ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

for i in range(len(layers)-1):
    ax.annotate('', xy=(x_positions[i+1]-0.01, 0.5),
                xytext=(x_positions[i]+0.01, 0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

ax.text(0.5, 0.97, f'Training accuracy: {mlp.best_validation_score_:.4f} | '
        f'Total layers: {len(mlp.hidden_layer_sizes)+2}',
        ha='center', va='top', transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f4f8', alpha=0.8))
plt.tight_layout()
plt.savefig('plots/06_nn_architecture.png', dpi=150, bbox_inches='tight')
plt.close()
print("[SAVED] plots/06_nn_architecture.png")

# ── PREDICTION DEMO ───────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("STEP 4: Live Prediction Demo")
print("─" * 65)

best_model = results[best_name]['model']
demo_reviews = [
    "An absolutely brilliant film. The performances were outstanding and the story was deeply moving.",
    "Terrible waste of time. The plot made no sense and the acting was painfully wooden throughout.",
    "A decent film with some good moments, though the pacing was a bit slow in the second half.",
]

print()
for rev in demo_reviews:
    cleaned   = clean_text(rev)
    vec       = vectorizer.transform([cleaned])
    pred      = best_model.predict(vec)[0]
    prob      = best_model.predict_proba(vec)[0]
    label     = "POSITIVE" if pred == 1 else "NEGATIVE"
    confidence = max(prob) * 100
    print(f"  Review    : \"{rev[:75]}...\"")
    print(f"  Prediction: {label}  (confidence: {confidence:.1f}%)")
    print()

# Detailed classification report for best model
print("─" * 65)
print(f"Detailed Classification Report — {best_name}")
print("─" * 65)
print(classification_report(y_test, results[best_name]['preds'],
                             target_names=['Negative', 'Positive']))

print("=" * 65)
print("  All steps completed. Plots saved in /plots/")
print("=" * 65)
