"""
Evaluation metrics for CampusGPT model
Comprehensive evaluation suite with multiple metrics
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import re
from pathlib import Path
import json

from src.utils.logger import get_logger
from src.utils.text_processing import extract_keywords, calculate_similarity

logger = get_logger(__name__)


def compute_metrics(eval_preds) -> Dict[str, float]:
    """
    Compute metrics for HuggingFace trainer
    
    Args:
        eval_preds: Tuple of (predictions, labels) from trainer
        
    Returns:
        Dictionary of computed metrics
    """
    predictions, labels = eval_preds
    
    # For now, return simple perplexity-like metric
    # In practice, this would compute more sophisticated metrics
    return {
        'eval_loss': np.mean(predictions),  # Simplified
        'perplexity': np.exp(np.mean(predictions))
    }


class EvaluationSuite:
    """Comprehensive evaluation suite for CampusGPT"""
    
    def __init__(self, model_path: str, test_data_path: str):
        """
        Initialize evaluation suite
        
        Args:
            model_path: Path to the model to evaluate
            test_data_path: Path to test dataset
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.test_data = self._load_test_data()
        
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """Load test dataset"""
        test_file = Path(self.test_data_path)
        if not test_file.exists():
            logger.warning(f"Test data not found: {self.test_data_path}")
            return []
        
        data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line.strip())
                    data.append(example)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(data)} test examples")
        return data
    
    def run_all_metrics(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation with all metrics
        
        Returns:
            Complete evaluation results
        """
        logger.info("Running comprehensive evaluation suite...")
        
        # Generate predictions (mock for now)
        predictions = self._generate_predictions()
        
        results = {
            'dataset_info': {
                'total_examples': len(self.test_data),
                'categories': self._get_category_distribution()
            },
            'metrics': {}
        }
        
        # Compute all metrics
        evaluators = [
            BLEUEvaluator(),
            ROUGEEvaluator(),
            AccuracyEvaluator(),
            SemanticSimilarityEvaluator(),
            CategoryAccuracyEvaluator()
        ]
        
        for evaluator in evaluators:
            logger.info(f"Computing {evaluator.__class__.__name__}...")
            metric_result = evaluator.evaluate(self.test_data, predictions)
            results['metrics'][evaluator.metric_name] = metric_result
        
        # Overall summary
        results['summary'] = self._generate_overall_summary(results['metrics'])
        
        return results
    
    def _generate_predictions(self) -> List[str]:
        """Generate model predictions for test data"""
        # Mock predictions for now
        # In practice, this would use the actual model
        predictions = []
        
        for example in self.test_data:
            instruction = example.get('instruction', '')
            category = example.get('category', 'general')
            
            # Generate category-appropriate responses
            if 'dining' in instruction.lower() or 'meal' in instruction.lower():
                pred = "The dining halls are open Monday-Friday 7:00 AM to 9:00 PM, and weekends 8:00 AM to 8:00 PM. The Late Night Grill is available until midnight on weekdays."
            elif 'register' in instruction.lower() or 'class' in instruction.lower():
                pred = "To register for classes: 1) Meet with your academic advisor, 2) Clear any holds on your account, 3) Use the online course catalog, 4) Register during your priority time."
            elif 'financial aid' in instruction.lower():
                pred = "Financial aid options include federal grants, scholarships, work-study, and loans. Complete the FAFSA by February 15th for priority consideration."
            elif 'library' in instruction.lower() or 'computer' in instruction.lower():
                pred = "For IT support, visit the Help Desk in the library basement, call (555) 123-HELP, or submit an online ticket. Free support for all students."
            elif 'housing' in instruction.lower() or 'dorm' in instruction.lower():
                pred = "Housing applications are due by May 1st. Contact the Housing Office for room assignments, meal plans, and residence hall information."
            else:
                pred = "Please contact the appropriate campus office for detailed information about your specific question."
            
            predictions.append(pred)
        
        return predictions
    
    def _get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of categories in test data"""
        categories = defaultdict(int)
        for example in self.test_data:
            category = example.get('category', 'unknown')
            categories[category] += 1
        return dict(categories)
    
    def _generate_overall_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall performance summary"""
        summary = {
            'overall_score': 0.0,
            'performance_level': 'unknown',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Calculate overall score (weighted average)
        weights = {
            'bleu': 0.25,
            'rouge': 0.25,
            'accuracy': 0.3,
            'semantic_similarity': 0.2
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                metric_score = metrics[metric_name].get('overall_score', 0.0)
                total_score += metric_score * weight
                total_weight += weight
        
        summary['overall_score'] = total_score / total_weight if total_weight > 0 else 0.0
        
        # Performance level
        if summary['overall_score'] >= 0.9:
            summary['performance_level'] = 'excellent'
        elif summary['overall_score'] >= 0.8:
            summary['performance_level'] = 'good'
        elif summary['overall_score'] >= 0.7:
            summary['performance_level'] = 'acceptable'
        else:
            summary['performance_level'] = 'needs_improvement'
        
        # Analyze strengths and weaknesses
        for metric_name, metric_result in metrics.items():
            score = metric_result.get('overall_score', 0.0)
            if score >= 0.85:
                summary['strengths'].append(f"High {metric_name} score ({score:.2f})")
            elif score < 0.7:
                summary['weaknesses'].append(f"Low {metric_name} score ({score:.2f})")
        
        # Generate recommendations
        if summary['overall_score'] < 0.8:
            summary['recommendations'].append("Consider additional fine-tuning on domain-specific data")
        
        if 'accuracy' in metrics and metrics['accuracy']['overall_score'] < 0.75:
            summary['recommendations'].append("Improve factual accuracy through knowledge base integration")
        
        return summary
    
    def generate_report(self, results: Dict[str, Any], output_path: str) -> None:
        """Generate detailed HTML evaluation report"""
        html_content = self._generate_html_report(results)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML evaluation report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CampusGPT Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .score {{ font-size: 1.5em; font-weight: bold; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>CampusGPT Evaluation Report</h1>
                <p>Model: {self.model_path}</p>
                <p>Test Examples: {results['dataset_info']['total_examples']}</p>
                <p>Overall Score: <span class="score">{results['summary']['overall_score']:.2f}</span></p>
                <p>Performance Level: {results['summary']['performance_level'].title()}</p>
            </div>
        """
        
        # Add metrics
        for metric_name, metric_result in results['metrics'].items():
            score = metric_result.get('overall_score', 0.0)
            css_class = 'good' if score >= 0.8 else 'warning' if score >= 0.7 else 'error'
            
            html += f"""
            <div class="metric">
                <h3>{metric_name.replace('_', ' ').title()}</h3>
                <p>Overall Score: <span class="score {css_class}">{score:.3f}</span></p>
            """
            
            # Category breakdown if available
            if 'category_breakdown' in metric_result:
                html += "<h4>Category Breakdown:</h4><table>"
                html += "<tr><th>Category</th><th>Score</th></tr>"
                for category, cat_score in metric_result['category_breakdown'].items():
                    html += f"<tr><td>{category}</td><td>{cat_score:.3f}</td></tr>"
                html += "</table>"
            
            html += "</div>"
        
        # Summary
        html += f"""
            <div class="metric">
                <h3>Summary & Recommendations</h3>
                <h4>Strengths:</h4>
                <ul>{"".join(f"<li>{s}</li>" for s in results['summary']['strengths'])}</ul>
                <h4>Areas for Improvement:</h4>
                <ul>{"".join(f"<li>{w}</li>" for w in results['summary']['weaknesses'])}</ul>
                <h4>Recommendations:</h4>
                <ul>{"".join(f"<li>{r}</li>" for r in results['summary']['recommendations'])}</ul>
            </div>
        """
        
        html += "</body></html>"
        return html


class BaseEvaluator:
    """Base class for evaluation metrics"""
    
    def __init__(self):
        self.metric_name = "base"
    
    def evaluate(self, test_data: List[Dict[str, Any]], predictions: List[str]) -> Dict[str, Any]:
        """Evaluate predictions against test data"""
        raise NotImplementedError
    
    def _compute_category_breakdown(
        self, 
        test_data: List[Dict[str, Any]], 
        scores: List[float]
    ) -> Dict[str, float]:
        """Compute per-category breakdown of scores"""
        category_scores = defaultdict(list)
        
        for example, score in zip(test_data, scores):
            category = example.get('category', 'unknown')
            category_scores[category].append(score)
        
        return {
            category: np.mean(scores) 
            for category, scores in category_scores.items()
        }


class BLEUEvaluator(BaseEvaluator):
    """BLEU score evaluator"""
    
    def __init__(self):
        super().__init__()
        self.metric_name = "bleu"
    
    def evaluate(self, test_data: List[Dict[str, Any]], predictions: List[str]) -> Dict[str, Any]:
        """Compute BLEU scores"""
        scores = []
        
        for example, prediction in zip(test_data, predictions):
            reference = example.get('output', '')
            bleu_score = self._compute_bleu(reference, prediction)
            scores.append(bleu_score)
        
        return {
            'overall_score': np.mean(scores),
            'category_breakdown': self._compute_category_breakdown(test_data, scores),
            'individual_scores': scores
        }
    
    def _compute_bleu(self, reference: str, prediction: str) -> float:
        """Compute simplified BLEU score"""
        if not prediction.strip():
            return 0.0
        
        ref_tokens = reference.lower().split()
        pred_tokens = prediction.lower().split()
        
        if not ref_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        # 1-gram precision
        ref_1grams = set(ref_tokens)
        pred_1grams = set(pred_tokens)
        
        if not pred_1grams:
            return 0.0
        
        overlap = len(ref_1grams & pred_1grams)
        precision = overlap / len(pred_1grams)
        
        # Brevity penalty
        bp = min(1.0, len(pred_tokens) / len(ref_tokens))
        
        return precision * bp


class ROUGEEvaluator(BaseEvaluator):
    """ROUGE score evaluator"""
    
    def __init__(self):
        super().__init__()
        self.metric_name = "rouge"
    
    def evaluate(self, test_data: List[Dict[str, Any]], predictions: List[str]) -> Dict[str, Any]:
        """Compute ROUGE scores"""
        scores = []
        
        for example, prediction in zip(test_data, predictions):
            reference = example.get('output', '')
            rouge_score = self._compute_rouge(reference, prediction)
            scores.append(rouge_score)
        
        return {
            'overall_score': np.mean(scores),
            'category_breakdown': self._compute_category_breakdown(test_data, scores),
            'individual_scores': scores
        }
    
    def _compute_rouge(self, reference: str, prediction: str) -> float:
        """Compute ROUGE-1 F1 score"""
        ref_tokens = set(reference.lower().split())
        pred_tokens = set(prediction.lower().split())
        
        if not ref_tokens and not pred_tokens:
            return 1.0
        
        if not ref_tokens or not pred_tokens:
            return 0.0
        
        overlap = len(ref_tokens & pred_tokens)
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)


class AccuracyEvaluator(BaseEvaluator):
    """Response accuracy evaluator"""
    
    def __init__(self):
        super().__init__()
        self.metric_name = "accuracy"
    
    def evaluate(self, test_data: List[Dict[str, Any]], predictions: List[str]) -> Dict[str, Any]:
        """Compute accuracy scores"""
        scores = []
        
        for example, prediction in zip(test_data, predictions):
            reference = example.get('output', '')
            category = example.get('category', 'general')
            accuracy = self._compute_accuracy(reference, prediction, category)
            scores.append(accuracy)
        
        return {
            'overall_score': np.mean(scores),
            'category_breakdown': self._compute_category_breakdown(test_data, scores),
            'individual_scores': scores
        }
    
    def _compute_accuracy(self, reference: str, prediction: str, category: str) -> float:
        """Compute contextual accuracy score"""
        ref_keywords = extract_keywords(reference)
        pred_keywords = extract_keywords(prediction)
        
        if not ref_keywords:
            return 1.0 if not pred_keywords else 0.5
        
        # Keyword overlap accuracy
        overlap = len(set(ref_keywords) & set(pred_keywords))
        keyword_accuracy = overlap / len(ref_keywords)
        
        # Category-specific bonus
        category_keywords = {
            'academic': ['register', 'course', 'advisor', 'grade', 'credit'],
            'student_life': ['dining', 'housing', 'activity', 'meal', 'dorm'],
            'administrative': ['financial', 'aid', 'form', 'office', 'deadline'],
            'campus_services': ['library', 'help', 'support', 'computer', 'wifi']
        }
        
        category_bonus = 0.0
        if category in category_keywords:
            for keyword in category_keywords[category]:
                if keyword in prediction.lower():
                    category_bonus += 0.05
        
        return min(1.0, keyword_accuracy + category_bonus)


class SemanticSimilarityEvaluator(BaseEvaluator):
    """Semantic similarity evaluator"""
    
    def __init__(self):
        super().__init__()
        self.metric_name = "semantic_similarity"
    
    def evaluate(self, test_data: List[Dict[str, Any]], predictions: List[str]) -> Dict[str, Any]:
        """Compute semantic similarity scores"""
        scores = []
        
        for example, prediction in zip(test_data, predictions):
            reference = example.get('output', '')
            similarity = calculate_similarity(reference, prediction)
            scores.append(similarity)
        
        return {
            'overall_score': np.mean(scores),
            'category_breakdown': self._compute_category_breakdown(test_data, scores),
            'individual_scores': scores
        }


class CategoryAccuracyEvaluator(BaseEvaluator):
    """Category-specific accuracy evaluator"""
    
    def __init__(self):
        super().__init__()
        self.metric_name = "category_accuracy"
    
    def evaluate(self, test_data: List[Dict[str, Any]], predictions: List[str]) -> Dict[str, Any]:
        """Evaluate category-specific accuracy"""
        category_results = defaultdict(list)
        
        for example, prediction in zip(test_data, predictions):
            category = example.get('category', 'unknown')
            reference = example.get('output', '')
            
            # Category-specific accuracy assessment
            accuracy = self._assess_category_accuracy(category, reference, prediction)
            category_results[category].append(accuracy)
        
        # Compute category averages
        category_scores = {
            category: np.mean(scores) 
            for category, scores in category_results.items()
        }
        
        overall_score = np.mean([
            score for scores in category_results.values() 
            for score in scores
        ]) if category_results else 0.0
        
        return {
            'overall_score': overall_score,
            'category_breakdown': category_scores,
            'category_details': dict(category_results)
        }
    
    def _assess_category_accuracy(self, category: str, reference: str, prediction: str) -> float:
        """Assess accuracy for specific category"""
        # Category-specific assessment criteria
        criteria = {
            'academic': ['advisor', 'register', 'course', 'credit', 'grade'],
            'student_life': ['dining', 'meal', 'housing', 'activity', 'hours'],
            'administrative': ['financial', 'aid', 'form', 'deadline', 'office'],
            'campus_services': ['library', 'help', 'computer', 'support', 'contact']
        }
        
        if category not in criteria:
            return 0.5  # Neutral for unknown categories
        
        expected_terms = criteria[category]
        prediction_lower = prediction.lower()
        
        # Check for expected terms
        term_matches = sum(1 for term in expected_terms if term in prediction_lower)
        term_score = term_matches / len(expected_terms)
        
        # Check response appropriateness (length, structure)
        structure_score = 1.0
        if len(prediction.split()) < 10:  # Too short
            structure_score -= 0.3
        if not any(char in prediction for char in '.!?'):  # No proper ending
            structure_score -= 0.2
        
        structure_score = max(0.0, structure_score)
        
        return (term_score * 0.7 + structure_score * 0.3)