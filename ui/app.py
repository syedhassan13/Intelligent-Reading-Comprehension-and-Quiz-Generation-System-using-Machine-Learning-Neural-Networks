"""
╔══════════════════════════════════════════════════════════════════╗
║  RACE Reading Comprehension & Quiz Generation System             ║
║  Streamlit UI  —  app.py                                         ║
║                                                                  ║
║  HOW TO RUN:                                                     ║
║    pip install streamlit                                         ║
║    streamlit run app.py                                          ║
║                                                                  ║
║  MODELS EXPECTED (place in models/ folder):                      ║
║    models/lr_model.pkl                                           ║
║    models/svm_model.pkl                                          ║
║    models/tfidf_vectorizer.pkl                                   ║
║    models/kmeans_model.pkl                                       ║
║    models/best_bert_model.pt   (optional — heavy)                ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import time
import os
import io
import json
import random
from collections import Counter
from datetime import datetime
from scipy.sparse import hstack, csr_matrix

# ── Streamlit page config (MUST be first Streamlit call) ──────────
st.set_page_config(
    page_title="RACE Quiz System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL B INTEGRATION INTERFACE
# ══════════════════════════════════════════════════════════════════
# Your friend needs to implement the ModelBInterface class below.
# See model_b_interface.py for the full specification.
# ──────────────────────────────────────────────────────────────────
try:
    from model_b_interface import ModelBInterface
    MODEL_B_AVAILABLE = True
except ImportError:
    MODEL_B_AVAILABLE = False

class _FallbackModelB:
    """
    Temporary Model B used until your friend's real Model B is ready.
    Replace by dropping model_b_interface.py in the same folder as app.py.
    """
    STOPWORDS = {
        'the','a','an','is','are','was','were','be','been','being',
        'have','has','had','do','does','did','will','would','could',
        'should','may','might','shall','can','to','of','in','for',
        'on','with','at','by','from','that','this','it','its','as',
        'or','and','but','if','not','he','she','they','we','i','you',
        'said','says','also','about','but','when','there','their','then'
    }

    def generate_distractors(self, article: str, question: str,
                              correct_answer: str, n: int = 3) -> list:
        words = article.split()
        candidates = [
            w.strip('.,!?;:()') for w in words
            if len(w.strip('.,!?;:()')) > 2
            and w.strip('.,!?;:()').lower() not in self.STOPWORDS
            and w.strip('.,!?;:()').lower() not in correct_answer.lower()
        ]
        freq = Counter(candidates)
        top = [w for w, _ in freq.most_common(30)
               if w.lower() not in correct_answer.lower()]
        seen, distractors = set(), []
        for c in top:
            if c.lower() not in seen:
                seen.add(c.lower())
                distractors.append(c)
            if len(distractors) >= n:
                break
        while len(distractors) < n:
            distractors.append(f"None of the above (option {len(distractors)+1})")
        return distractors[:n]

    def generate_hints(self, article: str, question: str,
                       correct_answer: str) -> list:
        sentences = re.split(r'[.!?]', article)
        sentences = [s.strip() for s in sentences if len(s.split()) > 6]
        q_words = set(question.lower().split()) - self.STOPWORDS
        scored = []
        for s in sentences:
            s_words = set(s.lower().split())
            overlap = len(q_words & s_words)
            scored.append((overlap, s))
        scored.sort(key=lambda x: -x[0])
        top_sentences = [s for _, s in scored[:3]]
        hints = [
            f"💡 Hint 1: Focus on the topic of '{question.split()[1] if len(question.split()) > 1 else 'the passage'}'.",
            f"💡 Hint 2: Look at this part of the passage: \"{top_sentences[0][:80]}...\"" if top_sentences else "💡 Hint 2: Re-read the article carefully.",
            f"💡 Hint 3: The answer is closely related to: \"{top_sentences[1][:80]}...\"" if len(top_sentences) > 1 else f"💡 Hint 3: The answer is '{correct_answer[:20]}...'",
        ]
        return hints


# Instantiate Model B
if MODEL_B_AVAILABLE:
    model_b = ModelBInterface()
    MODEL_B_LABEL = "✅ Model B (Your Friend's)"
else:
    model_b = _FallbackModelB()
    MODEL_B_LABEL = "⚠️ Model B (Fallback — replace with real Model B)"


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL A LOADING
# ══════════════════════════════════════════════════════════════════
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

@st.cache_resource(show_spinner=False)
def load_model_a():
    """Load all Model A components from disk."""
    loaded = {}
    paths = {
        'lr'    : os.path.join(MODELS_DIR, 'lr_model.pkl'),
        'svm'   : os.path.join(MODELS_DIR, 'svm_model.pkl'),
        'tfidf' : os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'),
        'kmeans': os.path.join(MODELS_DIR, 'kmeans_model.pkl'),
    }
    for name, path in paths.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                loaded[name] = pickle.load(f)
        else:
            loaded[name] = None

    # Optional BERT
    bert_path = os.path.join(MODELS_DIR, 'best_bert_model.pt')
    loaded['bert_available'] = os.path.exists(bert_path)
    loaded['bert_path']      = bert_path
    return loaded


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — HELPER / INFERENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════
STOPWORDS = {
    'the','a','an','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','will','would','could',
    'should','may','might','shall','can','to','of','in','for',
    'on','with','at','by','from','that','this','it','its','as',
    'or','and','but','if','not','he','she','they','we','i','you'
}

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_combined_text_single(question: str, option: str) -> str:
    return clean_text(question) + ' sep ' + clean_text(option)

def lexical_overlap(t1: str, t2: str) -> float:
    s1, s2 = set(t1.split()), set(t2.split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def build_lex_features_single(article, question, option) -> np.ndarray:
    a, q, o = clean_text(article), clean_text(question), clean_text(option)
    return np.array([[
        lexical_overlap(q, o),
        lexical_overlap(a, o),
        lexical_overlap(a, q),
        len(o.split()),
        len(q.split()),
        min(len(a.split()), 500),
        int('_' in question or '____' in question)
    ]], dtype=np.float32)

def verify_options(article, question, options_dict, models):
    """
    Run LR + SVM on all 4 options.
    Returns per-option probability dicts.
    """
    results = {'LR': {}, 'SVM': {}}
    if not models.get('tfidf'):
        return results, {}

    for opt_id, opt_text in options_dict.items():
        text    = build_combined_text_single(question, opt_text)
        X_tfidf = models['tfidf'].transform([text])
        X_lex   = build_lex_features_single(article, question, opt_text)
        X_feat  = hstack([X_tfidf, csr_matrix(X_lex)])

        if models.get('lr'):
            p = models['lr'].predict_proba(X_feat)[0][1]
            results['LR'][opt_id] = float(p)

        if models.get('svm'):
            p = models['svm'].predict_proba(X_feat)[0][1]
            results['SVM'][opt_id] = float(p)

    # Ensemble average
    ensemble = {}
    for opt_id in options_dict:
        probs = [v[opt_id] for v in results.values() if opt_id in v]
        ensemble[opt_id] = np.mean(probs) if probs else 0.25

    return results, ensemble

def extract_candidate_sentences(article: str, top_k=5) -> list:
    sentences = re.split(r'[.!?]', article)
    sentences = [s.strip() for s in sentences if len(s.split()) > 8]
    return sorted(sentences, key=lambda s: len(s.split()), reverse=True)[:top_k]

WH_WORDS = ['what', 'who', 'where', 'when', 'why', 'how', 'which']

def apply_wh_templates(sentence: str) -> list:
    questions, words = [], sentence.strip().split()
    if not words:
        return questions
    questions.append(f"What is discussed about {' '.join(words[:4])}?")
    persons = [w for w in words if w[0].isupper() and len(w) > 2]
    if persons:
        questions.append(f"Who is {persons[0]} in the context of the passage?")
    questions.append(f"Why does the author mention '{' '.join(words[:5])}'?")
    if re.findall(r'\b\d+\b', sentence):
        questions.append("How many/much is mentioned in the passage regarding this topic?")
    if any(w.lower() in ['in','at','from','near','between'] for w in words):
        questions.append("Where does the described event or situation take place?")
    return questions

def score_questions(questions: list) -> str:
    if not questions:
        return "What is the main idea of the passage?"
    def score(q):
        words = q.lower().split()
        return (
            (3 if words[0] in WH_WORDS else 0) +
            min(len(words), 20) / 20 +
            len(set(words)) / max(len(words), 1)
        )
    return max(questions, key=score)

def generate_question_from_article(article: str) -> dict:
    candidates = extract_candidate_sentences(article)
    if not candidates:
        return {'question': 'What is the passage about?',
                'correct_answer': article.split('.')[0],
                'source_sentence': ''}
    all_qs = []
    for s in candidates:
        all_qs.extend(apply_wh_templates(s))
    best_q   = score_questions(all_qs)
    best_s   = candidates[0]
    ans_words = [w for w in best_s.split() if w[0].isupper() and len(w) > 2]
    answer   = ' '.join(ans_words[:3]) if ans_words else best_s.split()[0]
    return {'question': best_q, 'correct_answer': answer, 'source_sentence': best_s}


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        'current_mcq'     : None,
        'user_answer'     : None,
        'checked'         : False,
        'hints_used'      : 0,
        'session_log'     : [],
        'total_correct'   : 0,
        'total_attempted' : 0,
        'inference_times' : [],
        'model_score_log' : [],
        'active_tab'      : 'quiz',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — CUSTOM CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Global ── */
    .main { background-color: #0f1117; }
    h1, h2, h3 { color: #e2e8f0; }

    /* ── Cards ── */
    .card {
        background: #1e2130;
        border: 1px solid #2d3148;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }

    /* ── Option buttons ── */
    .opt-btn {
        width: 100%;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: 2px solid #3d4266;
        background: #1e2130;
        color: #c9d1d9;
        font-size: 1rem;
        text-align: left;
        cursor: pointer;
        margin-bottom: 0.5rem;
        transition: all 0.2s;
    }
    .opt-btn:hover { border-color: #7c83db; background: #252840; }
    .opt-correct   { border-color: #2ecc71 !important; background: #1a3a2a !important; color: #2ecc71 !important; }
    .opt-incorrect { border-color: #e74c3c !important; background: #3a1a1a !important; color: #e74c3c !important; }
    .opt-selected  { border-color: #7c83db !important; background: #252840 !important; }

    /* ── Result badges ── */
    .badge-correct   { background:#1a3a2a; color:#2ecc71; border-radius:8px; padding:0.4rem 0.8rem; font-weight:700; }
    .badge-incorrect { background:#3a1a1a; color:#e74c3c; border-radius:8px; padding:0.4rem 0.8rem; font-weight:700; }

    /* ── Metric cards ── */
    .metric-card {
        background: #252840;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #3d4266;
    }
    .metric-val { font-size: 2rem; font-weight: 800; color: #7c83db; }
    .metric-lbl { font-size: 0.8rem; color: #888; margin-top: 0.2rem; }

    /* ── Hint boxes ── */
    .hint-box {
        background: #1e2a1e;
        border-left: 4px solid #f39c12;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        color: #f5cba7;
        font-size: 0.95rem;
    }

    /* ── Prob bar ── */
    .prob-bar-wrap { background:#2d3148; border-radius:6px; height:10px; margin-top:4px; }
    .prob-bar-fill { background:#7c83db; border-radius:6px; height:10px; }

    /* ── Model B badge ── */
    .model-b-badge {
        font-size:0.75rem;
        background:#252840;
        border:1px solid #3d4266;
        border-radius:6px;
        padding:2px 8px;
        color:#aaa;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background: #161827 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — LOAD MODELS
# ══════════════════════════════════════════════════════════════════
with st.spinner("Loading Model A..."):
    models = load_model_a()

models_ready = any([models.get('lr'), models.get('svm')])


# ══════════════════════════════════════════════════════════════════
# SECTION 7 — SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 RACE Quiz System")
    st.caption("Reading Comprehension & Quiz Generation")
    st.divider()

    # Navigation
    page = st.radio(
        "Navigate",
        ["📝 Article Input", "🎯 Quiz", "💡 Hints", "📊 Analytics"],
        key="nav"
    )

    st.divider()

    # Model status
    st.markdown("**🔧 Model Status**")
    status_items = [
        ("Logistic Regression", models.get('lr') is not None),
        ("SVM",                 models.get('svm') is not None),
        ("TF-IDF",              models.get('tfidf') is not None),
        ("K-Means",             models.get('kmeans') is not None),
        ("BERT",                models.get('bert_available', False)),
        ("Model B",             MODEL_B_AVAILABLE),
    ]
    for name, ready in status_items:
        icon = "🟢" if ready else "🟡"
        st.markdown(f"{icon} {name}")

    st.divider()

    if not models_ready:
        st.warning("⚠️ No trained models found in `models/` folder.\n\nRun the notebook first to train and save models.")

    st.markdown(f"<small class='model-b-badge'>{MODEL_B_LABEL}</small>", unsafe_allow_html=True)
    st.caption("v1.0 · NUCES FAST 2026")


# ══════════════════════════════════════════════════════════════════
# SECTION 8 — PAGE: ARTICLE INPUT  (Screen 1)
# ══════════════════════════════════════════════════════════════════
if page == "📝 Article Input":
    st.markdown("# 📝 Article Input")
    st.caption("Paste a reading passage, load a RACE sample, or upload a CSV row.")

    col_input, col_guide = st.columns([3, 1])

    with col_input:
        # ── Input mode tabs ────────────────────────────────────────
        tab_paste, tab_sample, tab_race = st.tabs(
            ["✏️ Paste Article", "🎲 Random RACE Sample", "📂 Load from CSV"]
        )

        with tab_paste:
            article_text = st.text_area(
                "Paste your reading passage here",
                height=280,
                placeholder="The history of the internet dates back to research commissioned by the "
                            "United States federal government in the 1960s..."
            )
            use_orig_q = st.checkbox("Also paste original question (optional)", value=False)
            orig_q, orig_a, orig_opts = "", "", {}
            if use_orig_q:
                orig_q = st.text_input("Original Question")
                c1, c2, c3, c4 = st.columns(4)
                orig_opts = {
                    'A': c1.text_input("Option A"),
                    'B': c2.text_input("Option B"),
                    'C': c3.text_input("Option C"),
                    'D': c4.text_input("Option D"),
                }
                orig_a = st.selectbox("Correct Answer", ['A','B','C','D'])

        with tab_sample:
            st.info("Upload your `train.csv` below and click **Load Random Sample**.")
            csv_up = st.file_uploader("Upload train.csv", type=['csv'], key='csv_sample')
            if csv_up:
                df_sample = pd.read_csv(csv_up)
                if st.button("🎲 Load Random Sample"):
                    row = df_sample.sample(1).iloc[0]
                    st.session_state['sample_row'] = row.to_dict()
                    st.rerun()
            if 'sample_row' in st.session_state:
                row = st.session_state['sample_row']
                article_text = row.get('article', '')
                orig_q    = row.get('question', '')
                orig_opts = {o: row.get(o,'') for o in ['A','B','C','D']}
                orig_a    = row.get('answer', 'A')
                st.text_area("Loaded Article", value=article_text, height=200, disabled=True)
                st.write(f"**Original Q:** {orig_q}")
                st.write(f"**Correct:** {orig_a}. {orig_opts.get(orig_a,'')}")
                use_orig_q = True

        with tab_race:
            st.info("Upload a single-row CSV with columns: article, question, A, B, C, D, answer")
            single_csv = st.file_uploader("Upload single row CSV", type=['csv'], key='single_csv')
            if single_csv:
                df_row = pd.read_csv(single_csv)
                if len(df_row) > 0:
                    row = df_row.iloc[0]
                    article_text = row.get('article', '')
                    orig_q    = row.get('question', '')
                    orig_opts = {o: row.get(o,'') for o in ['A','B','C','D']}
                    orig_a    = row.get('answer','A')
                    use_orig_q = True
                    st.success("✅ Row loaded!")

        # ── Submit button ───────────────────────────────────────────
        st.divider()
        submit = st.button("🚀 Submit — Generate MCQ", type="primary", use_container_width=True,
                           disabled=not article_text.strip())

        if submit and article_text.strip():
            if not models_ready:
                st.error("❌ No trained models found. Train models in the notebook first.")
            else:
                with st.spinner("🔄 Running inference pipeline..."):
                    t_start = time.time()

                    # Step 1: Question + answer
                    if use_orig_q and orig_q.strip() and all(orig_opts.values()):
                        question       = orig_q
                        correct_answer = orig_opts[orig_a]
                        all_options    = orig_opts
                        answer_key     = orig_a
                    else:
                        qa = generate_question_from_article(article_text)
                        question       = qa['question']
                        correct_answer = qa['correct_answer']
                        # Step 2: Distractors from Model B
                        distractors = model_b.generate_distractors(
                            article_text, question, correct_answer, n=3
                        )
                        opts_list  = [correct_answer] + distractors
                        random.shuffle(opts_list)
                        labels     = ['A','B','C','D']
                        all_options = dict(zip(labels, opts_list))
                        answer_key  = labels[opts_list.index(correct_answer)]

                    # Step 3: Model A verification scores
                    model_scores, ensemble = verify_options(
                        article_text, question, all_options, models
                    )

                    # Step 4: Hints from Model B
                    hints = model_b.generate_hints(
                        article_text, question, correct_answer
                    )

                    t_elapsed = time.time() - t_start

                    # Store in session
                    st.session_state['current_mcq'] = {
                        'article'       : article_text,
                        'question'      : question,
                        'options'       : all_options,
                        'answer_key'    : answer_key,
                        'correct_answer': correct_answer,
                        'model_scores'  : model_scores,
                        'ensemble'      : ensemble,
                        'hints'         : hints,
                        'timestamp'     : datetime.now().strftime('%H:%M:%S'),
                    }
                    st.session_state['checked']    = False
                    st.session_state['user_answer'] = None
                    st.session_state['hints_used']  = 0
                    st.session_state['inference_times'].append(round(t_elapsed, 3))

                st.success(f"✅ MCQ generated in {t_elapsed:.2f}s — switch to **🎯 Quiz** tab!")

    with col_guide:
        st.markdown("""
        <div class='card'>
        <b>📌 How it works</b><br><br>
        1️⃣ Paste article (or load sample)<br><br>
        2️⃣ Click <b>Submit</b><br><br>
        3️⃣ Model A generates question + verifies answers<br><br>
        4️⃣ Model B generates distractors + hints<br><br>
        5️⃣ Go to <b>🎯 Quiz</b> tab to answer
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='card'>
        <b>🔧 Pipeline</b><br><br>
        <small>
        Article<br>→ Template Q-Gen<br>→ SVM Ranker<br>→ Model B Distractors<br>→ LR + SVM Verifier<br>→ Ensemble Score
        </small>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 9 — PAGE: QUIZ  (Screen 2)
# ══════════════════════════════════════════════════════════════════
elif page == "🎯 Quiz":
    st.markdown("# 🎯 Quiz")

    mcq = st.session_state.get('current_mcq')
    if not mcq:
        st.info("👈 Go to **📝 Article Input** first and submit an article.")
        st.stop()

    # ── Article preview ────────────────────────────────────────────
    with st.expander("📄 Reading Passage", expanded=False):
        st.write(mcq['article'][:1500] + ("..." if len(mcq['article']) > 1500 else ""))

    # ── Question card ──────────────────────────────────────────────
    st.markdown(f"""
    <div class='card'>
        <b style='font-size:1.15rem'>❓ {mcq['question']}</b>
    </div>
    """, unsafe_allow_html=True)

    # ── Options ────────────────────────────────────────────────────
    checked = st.session_state['checked']
    answer_key = mcq['answer_key']

    if not checked:
        choice = st.radio(
            "Select your answer:",
            options=list(mcq['options'].keys()),
            format_func=lambda k: f"{k}.  {mcq['options'][k]}",
            key="quiz_radio"
        )
        col_check, col_hint = st.columns([1, 4])
        with col_check:
            if st.button("✅ Check Answer", type="primary"):
                st.session_state['user_answer'] = choice
                st.session_state['checked']     = True
                # Log result
                is_correct = (choice == answer_key)
                st.session_state['total_attempted'] += 1
                if is_correct:
                    st.session_state['total_correct'] += 1
                st.session_state['session_log'].append({
                    'timestamp'     : mcq['timestamp'],
                    'question'      : mcq['question'][:60],
                    'user_answer'   : choice,
                    'correct_answer': answer_key,
                    'is_correct'    : is_correct,
                    'lr_score'      : mcq['model_scores'].get('LR', {}).get(choice, 'N/A'),
                    'svm_score'     : mcq['model_scores'].get('SVM', {}).get(choice, 'N/A'),
                })
                st.session_state['model_score_log'].append({
                    'scores': mcq['model_scores'],
                    'ensemble': mcq['ensemble'],
                    'answer_key': answer_key,
                    'user_answer': choice,
                    'is_correct': is_correct,
                    'inference_time': st.session_state['inference_times'][-1] if st.session_state['inference_times'] else 0,
                })
                st.rerun()

    else:
        user_ans = st.session_state['user_answer']
        is_correct = (user_ans == answer_key)

        for opt_id, opt_text in mcq['options'].items():
            if opt_id == answer_key:
                cls = "opt-correct"
                prefix = "✅"
            elif opt_id == user_ans and not is_correct:
                cls = "opt-incorrect"
                prefix = "❌"
            else:
                cls = "opt-btn"
                prefix = "  "
            st.markdown(
                f"<div class='opt-btn {cls}'>{prefix}  <b>{opt_id}.</b>  {opt_text}</div>",
                unsafe_allow_html=True
            )

        # Result badge
        if is_correct:
            st.markdown("<br><span class='badge-correct'>🎉 CORRECT! Well done!</span>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<br><span class='badge-incorrect'>❌ Incorrect. Correct answer: <b>{answer_key}. {mcq['options'][answer_key]}</b></span>",
                unsafe_allow_html=True
            )

        # ── Model probability breakdown ────────────────────────────
        st.divider()
        st.markdown("#### 🤖 Model A — Verification Scores")
        col_lr, col_svm, col_ens = st.columns(3)

        for col, model_name in zip([col_lr, col_svm, col_ens], ['LR', 'SVM', 'Ensemble']):
            with col:
                st.markdown(f"**{model_name}**")
                scores = mcq['model_scores'].get(model_name, mcq['ensemble']) if model_name != 'Ensemble' else mcq['ensemble']
                for opt_id in ['A','B','C','D']:
                    p = scores.get(opt_id, 0.25)
                    bar_w = int(p * 100)
                    color = "#2ecc71" if opt_id == answer_key else ("#e74c3c" if opt_id == user_ans and not is_correct else "#7c83db")
                    st.markdown(
                        f"<div style='margin:4px 0'><small><b>{opt_id}</b> {p:.3f}</small>"
                        f"<div class='prob-bar-wrap'><div class='prob-bar-fill' style='width:{bar_w}%;background:{color}'></div></div></div>",
                        unsafe_allow_html=True
                    )

        st.divider()
        if st.button("🔄 Try Another Question", use_container_width=True):
            st.session_state['current_mcq'] = None
            st.session_state['checked'] = False
            st.session_state['user_answer'] = None
            st.session_state['hints_used'] = 0
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# SECTION 10 — PAGE: HINTS  (Screen 3)
# ══════════════════════════════════════════════════════════════════
elif page == "💡 Hints":
    st.markdown("# 💡 Hint Panel")

    mcq = st.session_state.get('current_mcq')
    if not mcq:
        st.info("👈 Go to **📝 Article Input** first and submit an article.")
        st.stop()

    st.markdown(f"""
    <div class='card'>
        <b>❓ {mcq['question']}</b>
    </div>
    """, unsafe_allow_html=True)

    hints       = mcq.get('hints', [])
    hints_used  = st.session_state['hints_used']
    checked     = st.session_state['checked']

    if not hints:
        st.warning("No hints available for this question.")
    else:
        hint_labels = ["General clue", "More specific", "Near-explicit"]
        for i, (hint, label) in enumerate(zip(hints, hint_labels)):
            if i < hints_used:
                st.markdown(f"""
                <div class='hint-box'>
                    <small><b>Hint {i+1} — {label}</b></small><br>
                    {hint}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='card' style='opacity:0.4'>
                    <small>🔒 Hint {i+1} — {label} (locked)</small>
                </div>
                """, unsafe_allow_html=True)

        col_hint_btn, col_reveal = st.columns([1, 1])
        with col_hint_btn:
            if hints_used < len(hints):
                if st.button(f"💡 Reveal Hint {hints_used + 1}", type="secondary"):
                    st.session_state['hints_used'] += 1
                    st.rerun()
            else:
                st.success("All hints revealed!")

        with col_reveal:
            # Show "Reveal Answer" only after all hints used
            if hints_used >= len(hints) and not checked:
                if st.button("🔓 Reveal Answer", type="primary"):
                    st.session_state['checked']     = True
                    st.session_state['user_answer']  = mcq['answer_key']
                    st.info(f"✅ Correct Answer: **{mcq['answer_key']}. {mcq['options'][mcq['answer_key']]}**")

    if checked:
        user_ans   = st.session_state['user_answer']
        is_correct = (user_ans == mcq['answer_key'])
        if is_correct:
            st.success(f"✅ You got it right: **{mcq['answer_key']}. {mcq['options'][mcq['answer_key']]}**")
        else:
            st.error(f"❌ Correct answer was: **{mcq['answer_key']}. {mcq['options'][mcq['answer_key']]}**")

    st.divider()
    st.caption(f"Model B hints: {MODEL_B_LABEL}")


# ══════════════════════════════════════════════════════════════════
# SECTION 11 — PAGE: ANALYTICS DASHBOARD  (Screen 4)
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("# 📊 Developer / Analytics Dashboard")

    total_att = st.session_state['total_attempted']
    total_cor = st.session_state['total_correct']
    acc = (total_cor / total_att) if total_att > 0 else 0
    avg_lat = np.mean(st.session_state['inference_times']) if st.session_state['inference_times'] else 0

    # ── Top metrics ────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{total_att}</div><div class='metric-lbl'>Questions Attempted</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{total_cor}</div><div class='metric-lbl'>Correct Answers</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{acc*100:.1f}%</div><div class='metric-lbl'>Session Accuracy</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{avg_lat*1000:.0f}ms</div><div class='metric-lbl'>Avg Inference Latency</div></div>", unsafe_allow_html=True)

    st.divider()

    # ── Session log table ──────────────────────────────────────────
    st.markdown("### 📋 Session Log")
    log = st.session_state['session_log']
    if log:
        df_log = pd.DataFrame(log)
        # Style: color correct/incorrect
        def color_correct(val):
            return 'color: #2ecc71' if val else 'color: #e74c3c'
        st.dataframe(
            df_log.style.applymap(color_correct, subset=['is_correct']),
            use_container_width=True,
            height=250
        )

        # Export
        csv_bytes = df_log.to_csv(index=False).encode()
        st.download_button(
            "📥 Export Session Log (CSV)",
            data=csv_bytes,
            file_name=f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No session data yet. Answer some questions in the 🎯 Quiz tab.")

    st.divider()

    # ── Inference time chart ───────────────────────────────────────
    st.markdown("### ⚡ Inference Latency")
    times = st.session_state['inference_times']
    if times:
        df_times = pd.DataFrame({'Request': range(1, len(times)+1), 'Latency (s)': times})
        st.line_chart(df_times.set_index('Request'))
        st.caption(f"Min: {min(times):.3f}s  |  Max: {max(times):.3f}s  |  Avg: {np.mean(times):.3f}s")
    else:
        st.info("No inference requests yet.")

    st.divider()

    # ── Model A performance (historical from score log) ─────────────
    st.markdown("### 🤖 Model A — Verification Performance")
    score_log = st.session_state['model_score_log']
    if score_log:
        col_a, col_b = st.columns(2)

        lr_correct, svm_correct, total_scored = 0, 0, 0
        for entry in score_log:
            ak  = entry['answer_key']
            ens = entry['ensemble']
            lr_scores  = entry['scores'].get('LR', {})
            svm_scores = entry['scores'].get('SVM', {})
            if lr_scores:
                pred_lr = max(lr_scores, key=lr_scores.get)
                lr_correct += int(pred_lr == ak)
            if svm_scores:
                pred_svm = max(svm_scores, key=svm_scores.get)
                svm_correct += int(pred_svm == ak)
            total_scored += 1

        with col_a:
            st.markdown(f"""
            <div class='card'>
                <b>Logistic Regression</b><br><br>
                MCQ Accuracy: <b>{lr_correct/max(total_scored,1)*100:.1f}%</b><br>
                Correct: {lr_correct} / {total_scored}
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div class='card'>
                <b>SVM</b><br><br>
                MCQ Accuracy: <b>{svm_correct/max(total_scored,1)*100:.1f}%</b><br>
                Correct: {svm_correct} / {total_scored}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Answer questions to populate model performance stats.")

    st.divider()

    # ── Model B status ─────────────────────────────────────────────
    st.markdown("### 🔗 Model B Status")
    if MODEL_B_AVAILABLE:
        st.success("✅ Real Model B is connected (model_b_interface.py found).")
    else:
        st.warning("⚠️ Using fallback distractor generator. Your friend's Model B not yet connected.\n\n"
                   "**To connect:** place `model_b_interface.py` in the same folder as `app.py`. See the file for the required interface.")
