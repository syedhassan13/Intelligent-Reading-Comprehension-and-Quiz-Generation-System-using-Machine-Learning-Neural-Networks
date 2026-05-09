"""
╔══════════════════════════════════════════════════════════════════╗
║  MODEL B INTEGRATION INTERFACE                                   ║
║  File: model_b_interface.py                                      ║
║                                                                  ║
║  ─── FOR YOUR FRIEND (Model B developer) ───────────────────     ║
║                                                                  ║
║  HOW TO CONNECT:                                                 ║
║  1. Copy this file to the same folder as app.py                  ║
║  2. Fill in the two methods below with your real Model B logic   ║
║  3. Restart Streamlit — the app auto-detects the file            ║
║                                                                  ║
║  WHAT YOUR FRIEND MUST IMPLEMENT:                                ║
║  ┌─────────────────────────────────────────────────────────┐    ║
║  │  generate_distractors(article, question,                 │    ║
║  │                        correct_answer, n=3) → list[str]  │    ║
║  │                                                          │    ║
║  │  generate_hints(article, question,                       │    ║
║  │                 correct_answer)       → list[str]        │    ║
║  └─────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── Put your friend's imports here ────────────────────────────────
# import pickle
# import torch
# from transformers import ...
# from your_model_b_module import DistractorModel, HintModel

import os


class ModelBInterface:
    """
    Model B — Distractor & Hint Generator.

    Your friend must implement the two methods below.
    Everything else (loading, saving, caching) is their choice.

    The Streamlit app only ever calls:
        model_b.generate_distractors(article, question, correct_answer, n=3)
        model_b.generate_hints(article, question, correct_answer)
    """

    def __init__(self):
        """
        Load your trained Model B here.
        Examples:
            self.distractor_model = pickle.load(open('model_b_distractor.pkl', 'rb'))
            self.hint_model       = pickle.load(open('model_b_hint.pkl', 'rb'))
            self.tfidf            = pickle.load(open('model_b_tfidf.pkl', 'rb'))
        """
        # ── YOUR FRIEND FILLS THIS IN ──────────────────────────────
        # MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_b')
        #
        # Example (traditional ML):
        # with open(os.path.join(MODELS_DIR, 'distractor_lr.pkl'), 'rb') as f:
        #     self.distractor_model = pickle.load(f)
        # with open(os.path.join(MODELS_DIR, 'hint_lr.pkl'), 'rb') as f:
        #     self.hint_model = pickle.load(f)
        # with open(os.path.join(MODELS_DIR, 'model_b_tfidf.pkl'), 'rb') as f:
        #     self.tfidf = pickle.load(f)
        #
        # Example (neural):
        # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        # self.tokenizer = AutoTokenizer.from_pretrained('model_b_checkpoint/')
        # self.model     = AutoModelForSeq2SeqLM.from_pretrained('model_b_checkpoint/')
        # ───────────────────────────────────────────────────────────
        pass

    # ──────────────────────────────────────────────────────────────
    # METHOD 1: Distractor Generator
    # ──────────────────────────────────────────────────────────────
    def generate_distractors(
        self,
        article: str,
        question: str,
        correct_answer: str,
        n: int = 3
    ) -> list:
        """
        Generate n plausible but incorrect answer options (distractors).

        Parameters
        ----------
        article        : str  — The full reading passage
        question       : str  — The multiple-choice question
        correct_answer : str  — The correct answer text
        n              : int  — Number of distractors to generate (default 3)

        Returns
        -------
        list of str   — Exactly n distractor strings

        Requirements (from project spec):
        • Plausibility : look like legitimate answers to an uninformed reader
        • Incorrectness: definitively wrong w.r.t. the passage
        • Diversity    : the three distractors must NOT be trivially similar
        • Grammatical consistency: all options share the same syntactic form

        Example
        -------
        article = "Albert Einstein developed the theory of relativity..."
        question = "Who developed the theory of relativity?"
        correct_answer = "Albert Einstein"
        → returns ["Isaac Newton", "Nikola Tesla", "Thomas Edison"]
        """
        # ── YOUR FRIEND FILLS THIS IN ──────────────────────────────
        #
        # Traditional ML pipeline (example):
        # candidates = self._extract_candidates(article, correct_answer)
        # features   = self._featurize(candidates, correct_answer, question)
        # scores     = self.distractor_model.predict_proba(features)[:, 1]
        # ranked     = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
        # unique     = list(dict.fromkeys(ranked))  # deduplicate
        # return unique[:n]
        #
        # Neural pipeline (example):
        # prompt = f"Article: {article}\nQuestion: {question}\nAnswer: {correct_answer}\nGenerate 3 distractors:"
        # inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        # outputs = self.model.generate(**inputs, max_new_tokens=60, num_return_sequences=3)
        # return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        #
        # ───────────────────────────────────────────────────────────
        raise NotImplementedError(
            "ModelBInterface.generate_distractors() not implemented yet. "
            "Your friend must fill in this method."
        )

    # ──────────────────────────────────────────────────────────────
    # METHOD 2: Hint Generator
    # ──────────────────────────────────────────────────────────────
    def generate_hints(
        self,
        article: str,
        question: str,
        correct_answer: str
    ) -> list:
        """
        Generate a list of 3 graduated hints for the question.

        Parameters
        ----------
        article        : str  — The full reading passage
        question       : str  — The multiple-choice question
        correct_answer : str  — The correct answer text

        Returns
        -------
        list of str (length 3)
          hint[0] → most general clue (barely touches the answer)
          hint[1] → more specific (points to relevant passage section)
          hint[2] → near-explicit (almost gives it away)

        Example
        -------
        → [
            "💡 Hint 1: Think about who the passage mostly talks about.",
            "💡 Hint 2: Look at the second paragraph of the passage.",
            "💡 Hint 3: The answer is a person mentioned right at the start.",
          ]
        """
        # ── YOUR FRIEND FILLS THIS IN ──────────────────────────────
        #
        # Extractive approach (example):
        # sentences = self._split_sentences(article)
        # features  = self._score_sentences(sentences, question)
        # top3      = sorted(zip(features, sentences), reverse=True)[:3]
        # return [
        #     f"💡 Hint 1: {top3[2][1][:80]}",   # least relevant
        #     f"💡 Hint 2: {top3[1][1][:80]}",
        #     f"💡 Hint 3: {top3[0][1][:80]}",   # most relevant
        # ]
        #
        # ───────────────────────────────────────────────────────────
        raise NotImplementedError(
            "ModelBInterface.generate_hints() not implemented yet. "
            "Your friend must fill in this method."
        )


# ══════════════════════════════════════════════════════════════════
# HELPER: Quick smoke test
# Run: python model_b_interface.py
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Testing ModelBInterface...")
    m = ModelBInterface()

    article = (
        "Albert Einstein was a German-born theoretical physicist who developed the "
        "theory of relativity. He is best known for the mass-energy equivalence formula "
        "E=mc². Einstein received the Nobel Prize in Physics in 1921."
    )
    question       = "Who developed the theory of relativity?"
    correct_answer = "Albert Einstein"

    print("\n[1] Testing generate_distractors...")
    try:
        distractors = m.generate_distractors(article, question, correct_answer, n=3)
        print(f"    Distractors: {distractors}")
        assert len(distractors) == 3, "Must return exactly 3 distractors"
        assert all(isinstance(d, str) for d in distractors), "All distractors must be strings"
        print("    ✅ PASSED")
    except NotImplementedError:
        print("    ⚠️  Not implemented yet (expected at this stage)")
    except Exception as e:
        print(f"    ❌ FAILED: {e}")

    print("\n[2] Testing generate_hints...")
    try:
        hints = m.generate_hints(article, question, correct_answer)
        print(f"    Hints: {hints}")
        assert len(hints) == 3, "Must return exactly 3 hints"
        assert all(isinstance(h, str) for h in hints), "All hints must be strings"
        print("    ✅ PASSED")
    except NotImplementedError:
        print("    ⚠️  Not implemented yet (expected at this stage)")
    except Exception as e:
        print(f"    ❌ FAILED: {e}")

    print("\nDone.")
