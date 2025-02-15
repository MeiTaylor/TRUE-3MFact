from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np
import json

def compute_rouge_bleu(reference_str, hypothesis_str):
    reference = " ".join(reference_str.strip().split())
    hypothesis = " ".join(hypothesis_str.strip().split())
    
    rouge = Rouge()
    
    rouge_scores = rouge.get_scores(hyps=hypothesis, refs=reference, avg=True)
    
    smoothing_fn = SmoothingFunction().method1
    
    bleu_1 = sentence_bleu(
        references=[reference.split(' ')],
        hypothesis=hypothesis.split(' '),
        weights=(1, 0, 0, 0),
        smoothing_function=smoothing_fn
    )
    
    bleu_2 = sentence_bleu(
        references=[reference.split(' ')],
        hypothesis=hypothesis.split(' '),
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothing_fn
    )
    
    bleu_3 = sentence_bleu(
        references=[reference.split(' ')],
        hypothesis=hypothesis.split(' '),
        weights=(0.33, 0.33, 0.33, 0),
        smoothing_function=smoothing_fn
    )
    
    bleu_4 = sentence_bleu(
        references=[reference.split(' ')],
        hypothesis=hypothesis.split(' '),
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing_fn
    )
    
    # 构建结果字典
    result = {
        'ROUGE-1': rouge_scores['rouge-1']['f'],
        'ROUGE-2': rouge_scores['rouge-2']['f'],
        'ROUGE-L': rouge_scores['rouge-l']['f'],
        'BLEU-1': bleu_1,
        'BLEU-2': bleu_2,
        'BLEU-3': bleu_3,
        'BLEU-4': bleu_4
    }
    
    return result

