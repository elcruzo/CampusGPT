"""
Individual evaluator classes for specific metrics
Modular evaluation components for CampusGPT assessment
"""

import numpy as np
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import re
from collections import Counter

from src.utils.text_processing import extract_keywords, calculate_similarity
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")
    
    @abstractmethod
    def evaluate(self, references: List[str], predictions: List[str], **kwargs) -> Dict[str, Any]:
        """
        Evaluate predictions against references
        
        Args:
            references: Ground truth texts
            predictions: Model predictions
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        pass
    
    def _validate_inputs(self, references: List[str], predictions: List[str]) -> None:
        """Validate input data"""
        if len(references) != len(predictions):
            raise ValueError(f"Length mismatch: {len(references)} references vs {len(predictions)} predictions")
        
        if not references:
            raise ValueError("Empty input data provided")


class BLEUEvaluator(BaseEvaluator):
    """BLEU score evaluator with n-gram matching"""
    
    def __init__(self, max_n: int = 4):
        super().__init__("BLEU")
        self.max_n = max_n
    
    def evaluate(self, references: List[str], predictions: List[str], **kwargs) -> Dict[str, Any]:
        """Compute BLEU scores"""
        self._validate_inputs(references, predictions)
        
        scores = []
        detailed_scores = []
        
        for ref, pred in zip(references, predictions):
            score, details = self._compute_bleu_score(ref, pred)
            scores.append(score)
            detailed_scores.append(details)
        
        return {
            'score': np.mean(scores),
            'scores': scores,
            'detailed_scores': detailed_scores,
            'metric': 'BLEU',
            'n_grams': self.max_n
        }
    
    def _compute_bleu_score(self, reference: str, prediction: str) -> tuple[float, Dict[str, Any]]:
        """Compute BLEU score for single pair"""
        if not prediction.strip():
            return 0.0, {'precision': [0.0] * self.max_n, 'bp': 0.0}
        
        ref_tokens = reference.lower().split()
        pred_tokens = prediction.lower().split()
        
        if not ref_tokens:
            return 1.0 if not pred_tokens else 0.0, {'precision': [1.0] * self.max_n, 'bp': 1.0}
        
        # Compute n-gram precisions
        precisions = []
        for n in range(1, self.max_n + 1):
            precision = self._compute_ngram_precision(ref_tokens, pred_tokens, n)
            precisions.append(precision)
        
        # Brevity penalty
        bp = self._compute_brevity_penalty(len(ref_tokens), len(pred_tokens))
        
        # BLEU score (geometric mean of precisions * brevity penalty)
        if all(p > 0 for p in precisions):
            bleu = bp * (np.prod(precisions) ** (1.0 / len(precisions)))
        else:
            bleu = 0.0
        
        return bleu, {'precision': precisions, 'bp': bp}
    
    def _compute_ngram_precision(self, ref_tokens: List[str], pred_tokens: List[str], n: int) -> float:
        """Compute n-gram precision"""
        if len(pred_tokens) < n:
            return 0.0
        
        # Generate n-grams
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        
        if not pred_ngrams:
            return 0.0
        
        # Count matches (with clipping for reference counts)
        matches = 0
        for ngram, count in pred_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
        
        return matches / sum(pred_ngrams.values())
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Generate n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        return Counter(ngrams)
    
    def _compute_brevity_penalty(self, ref_len: int, pred_len: int) -> float:
        """Compute brevity penalty"""
        if pred_len >= ref_len:
            return 1.0
        return np.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0


class ROUGEEvaluator(BaseEvaluator):
    """ROUGE score evaluator (ROUGE-1, ROUGE-2, ROUGE-L)"""
    
    def __init__(self, rouge_types: Optional[List[str]] = None):
        super().__init__("ROUGE")
        self.rouge_types = rouge_types or ['rouge-1', 'rouge-2', 'rouge-l']
    
    def evaluate(self, references: List[str], predictions: List[str], **kwargs) -> Dict[str, Any]:
        """Compute ROUGE scores"""
        self._validate_inputs(references, predictions)
        
        results = {}
        for rouge_type in self.rouge_types:
            scores = []
            for ref, pred in zip(references, predictions):
                score = self._compute_rouge_score(ref, pred, rouge_type)
                scores.append(score)
            
            results[rouge_type] = {
                'f1': np.mean([s['f1'] for s in scores]),
                'precision': np.mean([s['precision'] for s in scores]),
                'recall': np.mean([s['recall'] for s in scores]),
                'scores': scores
            }
        
        # Overall ROUGE score (average of ROUGE-1 and ROUGE-2 F1)
        overall_score = np.mean([
            results.get('rouge-1', {}).get('f1', 0.0),
            results.get('rouge-2', {}).get('f1', 0.0)
        ])
        
        return {
            'score': overall_score,
            'rouge_scores': results,
            'metric': 'ROUGE'
        }
    
    def _compute_rouge_score(self, reference: str, prediction: str, rouge_type: str) -> Dict[str, float]:
        """Compute specific ROUGE score"""
        ref_tokens = reference.lower().split()
        pred_tokens = prediction.lower().split()
        
        if rouge_type == 'rouge-1':
            return self._rouge_n_score(ref_tokens, pred_tokens, 1)
        elif rouge_type == 'rouge-2':
            return self._rouge_n_score(ref_tokens, pred_tokens, 2)
        elif rouge_type == 'rouge-l':
            return self._rouge_l_score(ref_tokens, pred_tokens)
        else:
            raise ValueError(f"Unknown ROUGE type: {rouge_type}")
    
    def _rouge_n_score(self, ref_tokens: List[str], pred_tokens: List[str], n: int) -> Dict[str, float]:
        """Compute ROUGE-N score"""
        if not pred_tokens or not ref_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        ref_ngrams = set(self._get_ngrams_list(ref_tokens, n))
        pred_ngrams = set(self._get_ngrams_list(pred_tokens, n))
        
        if not pred_ngrams:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        overlap = len(ref_ngrams & pred_ngrams)
        precision = overlap / len(pred_ngrams)
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0
        
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def _rouge_l_score(self, ref_tokens: List[str], pred_tokens: List[str]) -> Dict[str, float]:
        """Compute ROUGE-L score based on longest common subsequence"""
        if not pred_tokens or not ref_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        lcs_length = self._longest_common_subsequence(ref_tokens, pred_tokens)
        
        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(ref_tokens)
        
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def _get_ngrams_list(self, tokens: List[str], n: int) -> List[tuple]:
        """Get n-grams as list of tuples"""
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]


class AccuracyEvaluator(BaseEvaluator):
    """Content accuracy evaluator using keyword and semantic matching"""
    
    def __init__(self, keyword_weight: float = 0.6, semantic_weight: float = 0.4):
        super().__init__("Accuracy")
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
    
    def evaluate(self, references: List[str], predictions: List[str], **kwargs) -> Dict[str, Any]:
        """Compute accuracy scores"""
        self._validate_inputs(references, predictions)
        
        keyword_scores = []
        semantic_scores = []
        combined_scores = []
        
        for ref, pred in zip(references, predictions):
            keyword_acc = self._compute_keyword_accuracy(ref, pred)
            semantic_acc = self._compute_semantic_accuracy(ref, pred)
            combined_acc = (self.keyword_weight * keyword_acc + 
                          self.semantic_weight * semantic_acc)
            
            keyword_scores.append(keyword_acc)
            semantic_scores.append(semantic_acc)
            combined_scores.append(combined_acc)
        
        return {
            'score': np.mean(combined_scores),
            'keyword_accuracy': np.mean(keyword_scores),
            'semantic_accuracy': np.mean(semantic_scores),
            'scores': combined_scores,
            'metric': 'Accuracy'
        }
    
    def _compute_keyword_accuracy(self, reference: str, prediction: str) -> float:
        """Compute accuracy based on keyword overlap"""
        ref_keywords = set(extract_keywords(reference))
        pred_keywords = set(extract_keywords(prediction))
        
        if not ref_keywords:
            return 1.0 if not pred_keywords else 0.5
        
        if not pred_keywords:
            return 0.0
        
        overlap = len(ref_keywords & pred_keywords)
        return overlap / len(ref_keywords)
    
    def _compute_semantic_accuracy(self, reference: str, prediction: str) -> float:
        """Compute semantic similarity-based accuracy"""
        return calculate_similarity(reference, prediction)


class FluencyEvaluator(BaseEvaluator):
    """Evaluate response fluency and naturalness"""
    
    def __init__(self):
        super().__init__("Fluency")
        
        # Common fluency issues patterns
        self.fluency_patterns = {
            'repetition': r'\b(\w+)\s+\1\b',  # Word repetition
            'incomplete_sentence': r'[A-Z][^.!?]*$',  # Sentence without ending
            'grammar_issues': r'\b(a)\s+[aeiou]\w+',  # Basic grammar check
        }
    
    def evaluate(self, references: List[str], predictions: List[str], **kwargs) -> Dict[str, Any]:
        """Compute fluency scores"""
        self._validate_inputs(references, predictions)
        
        scores = []
        detailed_analysis = []
        
        for pred in predictions:
            score, analysis = self._compute_fluency_score(pred)
            scores.append(score)
            detailed_analysis.append(analysis)
        
        return {
            'score': np.mean(scores),
            'scores': scores,
            'detailed_analysis': detailed_analysis,
            'metric': 'Fluency'
        }
    
    def _compute_fluency_score(self, text: str) -> tuple[float, Dict[str, Any]]:
        """Compute fluency score for single text"""
        if not text.strip():
            return 0.0, {'issues': ['empty_text'], 'word_count': 0}
        
        score = 1.0
        issues = []
        
        # Check for repetition
        if re.search(self.fluency_patterns['repetition'], text, re.IGNORECASE):
            score -= 0.2
            issues.append('repetition')
        
        # Check sentence completeness
        sentences = re.split(r'[.!?]+', text.strip())
        if sentences and not text.strip()[-1] in '.!?':
            score -= 0.1
            issues.append('incomplete_ending')
        
        # Check average sentence length (too short or too long indicates issues)
        word_count = len(text.split())
        sentence_count = len([s for s in sentences if s.strip()])
        
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
            if avg_sentence_length < 5:  # Very short sentences
                score -= 0.15
                issues.append('very_short_sentences')
            elif avg_sentence_length > 40:  # Very long sentences
                score -= 0.1
                issues.append('very_long_sentences')
        
        # Check for basic grammar patterns
        if re.search(self.fluency_patterns['grammar_issues'], text, re.IGNORECASE):
            score -= 0.1
            issues.append('grammar_issues')
        
        score = max(0.0, score)
        
        analysis = {
            'issues': issues,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length if sentence_count > 0 else 0
        }
        
        return score, analysis


class RelevanceEvaluator(BaseEvaluator):
    """Evaluate response relevance to the given instruction"""
    
    def __init__(self):
        super().__init__("Relevance")
    
    def evaluate(self, references: List[str], predictions: List[str], 
                instructions: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Compute relevance scores"""
        self._validate_inputs(references, predictions)
        
        if instructions and len(instructions) != len(predictions):
            raise ValueError("Instructions length must match predictions length")
        
        scores = []
        
        for i, (ref, pred) in enumerate(zip(references, predictions)):
            instruction = instructions[i] if instructions else ""
            score = self._compute_relevance_score(instruction, ref, pred)
            scores.append(score)
        
        return {
            'score': np.mean(scores),
            'scores': scores,
            'metric': 'Relevance'
        }
    
    def _compute_relevance_score(self, instruction: str, reference: str, prediction: str) -> float:
        """Compute relevance score for single example"""
        if not prediction.strip():
            return 0.0
        
        # Extract key concepts from instruction and reference
        instruction_keywords = set(extract_keywords(instruction)) if instruction else set()
        reference_keywords = set(extract_keywords(reference))
        prediction_keywords = set(extract_keywords(prediction))
        
        # Combine instruction and reference keywords as relevant concepts
        relevant_keywords = instruction_keywords | reference_keywords
        
        if not relevant_keywords:
            return 0.5  # Neutral if no reference concepts
        
        if not prediction_keywords:
            return 0.0
        
        # Calculate keyword overlap with relevant concepts
        overlap = len(relevant_keywords & prediction_keywords)
        keyword_relevance = overlap / len(relevant_keywords)
        
        # Boost score if prediction addresses the instruction directly
        instruction_boost = 0.0
        if instruction:
            instruction_similarity = calculate_similarity(instruction, prediction)
            instruction_boost = instruction_similarity * 0.3
        
        # Final relevance score
        relevance = min(1.0, keyword_relevance + instruction_boost)
        
        return relevance