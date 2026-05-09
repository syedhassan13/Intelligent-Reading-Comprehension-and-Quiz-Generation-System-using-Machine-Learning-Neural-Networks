"""
inference.py  —  Unified inference API for the Streamlit UI
"""
from __future__ import annotations
import os, re, time, pickle, random
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../models/model_b'))
from model_b_interface import ModelB, MODEL_B_IS_STUB

MODEL_DIR_A = os.path.join(os.path.dirname(__file__), '../models/model_a')

@dataclass
class MCQResult:
    question:        str
    options:         Dict[str, str]
    correct_answer:  str
    source_sentence: str
    is_race_original: bool = False

@dataclass
class VerifyResult:
    user_choice:      str
    predicted_answer: str
    is_correct:       bool
    lr_probs:         Dict[str, float]
    svm_probs:        Dict[str, float]
    ensemble_probs:   Dict[str, float]
    latency_ms:       float
    explanation:      str

@dataclass
class SessionStats:
    total: int = 0
    correct: int = 0
    latencies: list = field(default_factory=list)
    @property
    def accuracy(self): return self.correct / self.total if self.total else 0.0
    @property
    def avg_latency(self): return float(np.mean(self.latencies)) if self.latencies else 0.0

_STOPWORDS = {'the','a','an','is','are','was','were','be','been','have','has','had','do','does','did','to','of','in','for','on','with','at','by','from','that','this','it','as','or','and','but','not','he','she','they','we','i','you','his','her','their','its','our','my','your','which','what','who','where','when','why','how'}
WH_WORDS = ['what', 'who', 'where', 'when', 'why', 'how', 'which']

def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def lexical_overlap(t1, t2):
    s1,s2 = set(t1.split()),set(t2.split())
    if not s1 or not s2: return 0.0
    return len(s1&s2)/len(s1|s2)

def build_lexical_features(df):
    feats=[]
    for _,row in df.iterrows():
        feats.append([lexical_overlap(row['question'],row['option']),lexical_overlap(row['article'],row['option']),lexical_overlap(row['article'],row['question']),len(row['option'].split()),len(row['question'].split()),len(row['article'].split()),int('_' in row.get('question_raw',(''))) ])
    return np.array(feats,dtype=np.float32)

def build_combined_text(df): return df['question']+' sep '+df['option']

def _extract_candidate_sentences(article,top_k=5):
    sents=[s.strip() for s in re.split(r'[.!?]',article) if len(s.split())>8]
    return sorted(sents,key=lambda s:len(s.split()),reverse=True)[:top_k]

def _apply_wh_templates(sentence):
    words=sentence.strip().split()
    if not words: return []
    qs=[f"What is discussed about {' '.join(words[:4])}?"]
    pw=[w for w in words if w[0].isupper() and len(w)>2]
    if pw: qs.append(f"Who is {pw[0]} mentioned in the passage?")
    qs.append(f"Why does the passage mention: '{sentence[:50]}...'?")
    if re.findall(r'\b\d+\b',sentence): qs.append("According to the passage, what is the relevant number or amount?")
    return qs

def _rank_questions(questions):
    if not questions: return "What is the main idea of the passage?"
    def score(q):
        w=q.lower().split()
        return (1 if w and w[0] in WH_WORDS else 0)*3+len(w)/20+len(set(w))/max(len(w),1)
    return max(questions,key=score)

def generate_question_from_article(article):
    cands=_extract_candidate_sentences(article)
    if not cands: return {'question':'What is the passage about?','correct_answer':article.split('.')[0],'source_sentence':''}
    all_q=[]
    for s in cands: all_q.extend(_apply_wh_templates(s))
    best_q=_rank_questions(all_q)
    bs=cands[0]
    aw=[w for w in bs.split() if w[0].isupper() and len(w)>2]
    ans=' '.join(aw[:3]) if aw else bs.split()[0]
    return {'question':best_q,'correct_answer':ans,'source_sentence':bs}

class InferenceEngine:
    def __init__(self):
        self.lr_model=None; self.svm_model=None; self.tfidf=None
        self.bert_model=None; self.bert_tokenizer=None
        self.model_b=ModelB(); self.device='cpu'
        self.is_ready=False; self.model_b_is_stub=MODEL_B_IS_STUB

    def load(self, model_dir=MODEL_DIR_A):
        errors=[]
        for attr,fname in [('lr_model','lr_model.pkl'),('svm_model','svm_model.pkl'),('tfidf','tfidf_vectorizer.pkl')]:
            p=os.path.join(model_dir,fname)
            if os.path.exists(p):
                with open(p,'rb') as f: setattr(self,attr,pickle.load(f))
            else: errors.append(f'{fname} not found')
        bert_path=os.path.join(model_dir,'best_bert_model.pt')
        if os.path.exists(bert_path):
            try:
                import torch
                from transformers import BertForSequenceClassification,BertTokenizerFast
                self.device='cuda' if torch.cuda.is_available() else 'cpu'
                self.bert_tokenizer=BertTokenizerFast.from_pretrained('bert-base-uncased')
                self.bert_model=BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
                self.bert_model.load_state_dict(torch.load(bert_path,map_location=self.device))
                self.bert_model.to(self.device); self.bert_model.eval()
            except Exception as e: errors.append(f'BERT: {e}')
        try: self.model_b.load()
        except Exception as e: errors.append(f'ModelB: {e}')
        self.is_ready=(self.lr_model is not None or self.svm_model is not None)
        return errors

    def _build_features(self,article,question,options):
        rows=[{'article':clean_text(article),'question':clean_text(question),'option':clean_text(t),'option_id':l,'question_raw':question,'label':0} for l,t in options.items()]
        df=pd.DataFrame(rows)
        X=hstack([self.tfidf.transform(build_combined_text(df)),csr_matrix(build_lexical_features(df))])
        return X,list(options.keys())

    def verify(self,article,question,user_choice,options):
        t0=time.time()
        X,labels=self._build_features(article,question,options)
        lr_p={l:0.0 for l in labels}; svm_p={l:0.0 for l in labels}
        if self.lr_model: lr_p=dict(zip(labels,self.lr_model.predict_proba(X)[:,1].tolist()))
        if self.svm_model: svm_p=dict(zip(labels,self.svm_model.predict_proba(X)[:,1].tolist()))
        active=[d for d in [lr_p,svm_p] if any(v>0 for v in d.values())]
        ens={l:float(np.mean([d[l] for d in active])) for l in labels} if active else {l:0.25 for l in labels}
        pred=max(ens,key=ens.get); correct=(user_choice==pred); lat=(time.time()-t0)*1000
        exp=f"✅ Correct! Models predict {pred} ({ens[pred]:.0%})." if correct else f"❌ Models predict {pred} ({ens[pred]:.0%}), you chose {user_choice}."
        return VerifyResult(user_choice,pred,correct,lr_p,svm_p,ens,lat,exp)

    def build_mcq(self,article):
        qa=generate_question_from_article(article)
        q,ca,ss=qa['question'],qa['correct_answer'],qa['source_sentence']
        dr=self.model_b.get_distractors(article,q,ca,n=3)
        opts=[ca]+dr.distractors; random.shuffle(opts)
        labels=['A','B','C','D']; options=dict(zip(labels,opts))
        cl=[l for l,t in options.items() if t==ca][0]
        return MCQResult(q,options,cl,ss,False)

    def mcq_from_race_row(self,row):
        return MCQResult(row['question'],{'A':row['A'],'B':row['B'],'C':row['C'],'D':row['D']},row['answer'],'',True)

    def get_hints(self,article,question,correct_answer):
        return self.model_b.get_hints(article,question,correct_answer)