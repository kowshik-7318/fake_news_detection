#!/usr/bin/env python3
"""
Fake News Detection Using NLP
==============================
A complete machine learning pipeline for detecting fake news articles.
Compatible with Python 3.13 and uses NLP techniques with multiple ML models.

Author: AI Assistant
Version: 1.0.0
Date: 2026-06-28
"""

# Standard library imports
import os
import sys
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import init, Fore, Style, Back
import joblib

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Download required NLTK data (with error handling)
def setup_nltk():
    """Download all required NLTK datasets with proper error handling."""
    nltk_data = [
        'stopwords',
        'punkt',
        'punkt_tab',
        'wordnet',
        'omw-1.4'
    ]
    
    for data in nltk_data:
        try:
            nltk.download(data, quiet=True)
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not download NLTK data '{data}': {e}")
    
    # Verify punkt is available (common issue in newer NLTK versions)
    try:
        word_tokenize("Test sentence")
    except LookupError:
        try:
            nltk.download('punkt', quiet=True, force=True)
            nltk.download('punkt_tab', quiet=True, force=True)
        except:
            pass

# Setup logging
def setup_logging() -> logging.Logger:
    """
    Configure logging to both file and console.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure logging
    log_filename = f"logs/fake_news_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles dataset loading, validation, and initial exploration.
    """
    
    def __init__(self, filepath: str = 'news.csv'):
        """
        Initialize DataLoader.
        
        Args:
            filepath (str): Path to the CSV dataset file
        """
        self.filepath = filepath
        self.data = None
        self.logger = logging.getLogger(__name__)
    
    def load_dataset(self) -> pd.DataFrame:
        """
        Load dataset from CSV file with proper error handling.
        
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If required columns are missing
        """
        try:
            if not os.path.exists(self.filepath):
                raise FileNotFoundError(
                    f"{Fore.RED}Dataset file '{self.filepath}' not found. "
                    f"Please ensure the file exists in the current directory."
                )
            
            self.data = pd.read_csv(self.filepath)
            self.logger.info(f"Dataset loaded successfully: {self.filepath}")
            
            # Validate required columns
            required_columns = ['title', 'text', 'label']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                # Check for alternative column names (common in some datasets)
                column_mapping = {
                    'title': ['title', 'Title', 'TITLE', 'headline', 'Headline'],
                    'text': ['text', 'Text', 'TEXT', 'body', 'Body', 'article', 'Article'],
                    'label': ['label', 'Label', 'LABEL', 'class', 'Class', 'category', 'Category']
                }
                
                for col_name, alternatives in column_mapping.items():
                    if col_name not in self.data.columns:
                        found = False
                        for alt in alternatives:
                            if alt in self.data.columns:
                                self.data.rename(columns={alt: col_name}, inplace=True)
                                found = True
                                self.logger.info(f"Renamed column '{alt}' to '{col_name}'")
                                break
                        if not found:
                            raise ValueError(
                                f"{Fore.RED}Required column '{col_name}' not found in dataset. "
                                f"Available columns: {list(self.data.columns)}"
                            )
            
            # Clean label column (standardize to binary)
            self.data['label'] = self.data['label'].str.upper().str.strip()
            
            # Map various label formats to FAKE/REAL
            label_mapping = {
                'FAKE': 0, 'FALSE': 0, '0': 0, 'UNRELIABLE': 0, 'MISLEADING': 0,
                'REAL': 1, 'TRUE': 1, '1': 1, 'RELIABLE': 1, 'AUTHENTIC': 1
            }
            
            self.data['label_numeric'] = self.data['label'].map(label_mapping)
            
            # Check for unmapped labels
            unmapped = self.data[self.data['label_numeric'].isna()]['label'].unique()
            if len(unmapped) > 0:
                self.logger.warning(f"Found unmapped labels: {unmapped}. These will be dropped.")
                self.data = self.data.dropna(subset=['label_numeric'])
            
            self.data['label_numeric'] = self.data['label_numeric'].astype(int)
            
            # Handle missing text content
            self.data['text'] = self.data['text'].fillna('')
            self.data['title'] = self.data['title'].fillna('')
            
            # Combine title and text for better feature extraction
            self.data['combined_text'] = self.data['title'] + ' ' + self.data['text']
            
            self.logger.info(f"Dataset processed: {len(self.data)} samples")
            return self.data
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset information.
        
        Returns:
            Dict: Dataset statistics and information
        """
        if self.data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        info = {
            'total_samples': len(self.data),
            'fake_count': len(self.data[self.data['label_numeric'] == 0]),
            'real_count': len(self.data[self.data['label_numeric'] == 1]),
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'class_distribution': self.data['label'].value_counts().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        info['fake_percentage'] = (info['fake_count'] / info['total_samples']) * 100
        info['real_percentage'] = (info['real_count'] / info['total_samples']) * 100
        
        return info


class TextPreprocessor:
    """
    Handles text preprocessing including cleaning, tokenization, and vectorization.
    """
    
    def __init__(self):
        """Initialize TextPreprocessor with required NLTK components."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.logger = logging.getLogger(__name__)
        
        # Additional custom stopwords for news context
        self.custom_stopwords = {'said', 'would', 'could', 'also', 'one', 'two', 
                                 'first', 'last', 'new', 'like', 'get', 'make',
                                 'many', 'much', 'still', 'back', 'even'}
        self.stop_words.update(self.custom_stopwords)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Raw text input
            
        Returns:
            str: Cleaned and preprocessed text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        import re
        import string
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation and special characters
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize text and apply lemmatization.
        
        Args:
            text (str): Cleaned text
            
        Returns:
            List[str]: List of lemmatized tokens
        """
        if not text:
            return []
        
        try:
            # Tokenization
            tokens = word_tokenize(text)
            
            # Remove stopwords and short tokens
            tokens = [token for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
            
            # Lemmatization
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            return tokens
        except Exception as e:
            self.logger.error(f"Tokenization error: {e}")
            return text.split()
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline for a single text.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text ready for vectorization
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        return ' '.join(tokens)
    
    def preprocess_corpus(self, texts: pd.Series) -> pd.Series:
        """
        Preprocess an entire corpus of texts.
        
        Args:
            texts (pd.Series): Series of raw texts
            
        Returns:
            pd.Series: Series of preprocessed texts
        """
        self.logger.info("Starting corpus preprocessing...")
        processed = texts.apply(self.preprocess_text)
        self.logger.info(f"Preprocessing complete. Processed {len(processed)} texts.")
        return processed
    
    def vectorize(self, texts: pd.Series, fit: bool = True) -> np.ndarray:
        """
        TF-IDF vectorization of text data.
        
        Args:
            texts (pd.Series): Preprocessed texts
            fit (bool): Whether to fit the vectorizer (True for training data)
            
        Returns:
            np.ndarray: TF-IDF matrix
        """
        if fit:
            self.logger.info("Fitting TF-IDF vectorizer...")
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=5,
                max_df=0.7,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
            vectors = self.vectorizer.fit_transform(texts)
            self.logger.info(f"TF-IDF vectorization complete. Shape: {vectors.shape}")
        else:
            if self.vectorizer is None:
                raise ValueError("Vectorizer not fitted. Please fit on training data first.")
            vectors = self.vectorizer.transform(texts)
        
        return vectors.toarray()


class ModelTrainer:
    """
    Trains and evaluates multiple machine learning models for fake news detection.
    """
    
    def __init__(self):
        """Initialize ModelTrainer with model configurations."""
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                C=1.0, 
                random_state=42,
                n_jobs=-1
            ),
            'Multinomial NB': MultinomialNB(alpha=0.1),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Linear SVM': CalibratedClassifierCV(
                LinearSVC(
                    C=1.0,
                    max_iter=1000,
                    random_state=42,
                    dual=False
                ),
                cv=5
            )
        }
        
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.logger = logging.getLogger(__name__)
    
    def train_and_evaluate(self, X_train: np.ndarray, X_test: np.ndarray, 
                          y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train all models and evaluate their performance.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
            
        Returns:
            Dict: Evaluation results for all models
        """
        self.logger.info("Starting model training and evaluation...")
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Probability predictions (for ROC-AUC)
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                except:
                    # For models without predict_proba, use decision function
                    try:
                        y_prob = model.decision_function(X_test)
                        # Normalize to [0, 1] range
                        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
                    except:
                        y_prob = y_pred
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary')
                recall = recall_score(y_test, y_pred, average='binary')
                f1 = f1_score(y_test, y_pred, average='binary')
                
                # ROC-AUC
                try:
                    roc_auc = roc_auc_score(y_test, y_prob)
                except:
                    roc_auc = None
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'y_pred': y_pred,
                    'y_prob': y_prob,
                    'classification_report': classification_report(
                        y_test, y_pred, 
                        target_names=['FAKE', 'REAL']
                    ),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                self.logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
                continue
        
        # Find best model based on F1 score
        if self.results:
            best_model_name = max(self.results.items(), 
                                 key=lambda x: x[1]['f1_score'])[0]
            self.best_model_name = best_model_name
            self.best_model = self.models[best_model_name]
            self.logger.info(f"Best model: {best_model_name}")
        
        return self.results
    
    def display_results(self) -> None:
        """
        Display formatted comparison table of model results.
        """
        if not self.results:
            print(f"{Fore.RED}No results to display. Train models first.")
            return
        
        # Header
        print(f"\n{Fore.CYAN}{'='*100}")
        print(f"{Fore.YELLOW}{'MODEL PERFORMANCE COMPARISON':^100}")
        print(f"{Fore.CYAN}{'='*100}")
        
        # Table header
        header = f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'ROC-AUC':<12}"
        print(f"{Fore.WHITE}{Style.BRIGHT}{header}")
        print(f"{Fore.CYAN}{'-'*100}")
        
        # Table rows
        for name, metrics in self.results.items():
            roc_auc = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A"
            
            # Highlight best model
            if name == self.best_model_name:
                color = Fore.GREEN
                style = Style.BRIGHT
            else:
                color = Fore.WHITE
                style = Style.NORMAL
            
            row = (f"{style}{name:<25} {metrics['accuracy']:<12.4f} "
                   f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                   f"{metrics['f1_score']:<12.4f} {roc_auc:<12}")
            print(color + row)
        
        print(f"{Fore.CYAN}{'='*100}")
        print(f"{Fore.GREEN}{Style.BRIGHT}Best Model: {self.best_model_name}")
        print(f"{Fore.CYAN}{'='*100}\n")
    
    def plot_confusion_matrices(self, save_path: str = 'confusion_matrix.png') -> None:
        """
        Plot confusion matrices for all models.
        
        Args:
            save_path (str): Path to save the plot
        """
        if not self.results:
            print(f"{Fore.RED}No results to plot. Train models first.")
            return
        
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            if idx >= 4:
                break
            
            cm = metrics['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['FAKE', 'REAL'],
                       yticklabels=['FAKE', 'REAL'],
                       ax=axes[idx])
            
            axes[idx].set_title(f'{name}\nF1 Score: {metrics["f1_score"]:.4f}', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrices saved to {save_path}")
        print(f"{Fore.GREEN}Confusion matrices saved to: {save_path}")
    
    def save_models(self) -> None:
        """
        Save the best model and vectorizer to disk.
        """
        try:
            # Save best model
            joblib.dump(self.best_model, 'best_model.pkl')
            self.logger.info("Best model saved to best_model.pkl")
            print(f"{Fore.GREEN}Best model saved to: best_model.pkl")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            print(f"{Fore.RED}Error saving model: {e}")


class NewsPredictor:
    """
    Handles prediction of new/unseen news articles.
    """
    
    def __init__(self, model, vectorizer, preprocessor):
        """
        Initialize predictor with trained components.
        
        Args:
            model: Trained ML model
            vectorizer: Fitted TF-IDF vectorizer
            preprocessor: TextPreprocessor instance
        """
        self.model = model
        self.vectorizer = vectorizer
        self.preprocessor = preprocessor
        self.logger = logging.getLogger(__name__)
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict whether a news article is fake or real.
        
        Args:
            text (str): News article text
            
        Returns:
            Dict: Prediction results with probabilities
        """
        if not text or not text.strip():
            raise ValueError(f"{Fore.RED}Empty text provided for prediction.")
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess_text(text)
        
        # Vectorize
        vector = self.vectorizer.transform([processed_text]).toarray()
        
        # Predict
        prediction = self.model.predict(vector)[0]
        
        # Get probabilities
        try:
            probabilities = self.model.predict_proba(vector)[0]
            fake_prob = probabilities[0] * 100
            real_prob = probabilities[1] * 100
        except:
            # If model doesn't support predict_proba
            fake_prob = 100 - (prediction * 100)
            real_prob = prediction * 100
        
        # Determine confidence level
        max_prob = max(fake_prob, real_prob)
        if max_prob >= 90:
            confidence = "HIGH"
        elif max_prob >= 70:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return {
            'prediction': 'REAL' if prediction == 1 else 'FAKE',
            'fake_probability': fake_prob,
            'real_probability': real_prob,
            'confidence': confidence,
            'predicted_class': int(prediction)
        }
    
    def display_prediction(self, result: Dict[str, Any], model_name: str) -> None:
        """
        Display formatted prediction results.
        
        Args:
            result (Dict): Prediction results
            model_name (str): Name of the model used
        """
        is_fake = result['prediction'] == 'FAKE'
        
        if is_fake:
            header_color = Back.RED + Fore.WHITE
            title = "FAKE NEWS DETECTED"
        else:
            header_color = Back.GREEN + Fore.WHITE
            title = "REAL NEWS"
        
        print(f"\n{header_color}{Style.BRIGHT}")
        print(f"{'='*40}")
        print(f"{title:^40}")
        print(f"{'='*40}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}Prediction    : {Fore.YELLOW}{Style.BRIGHT}{result['prediction']}")
        
        # Color code probabilities
        fake_color = Fore.RED if result['fake_probability'] > 50 else Fore.GREEN
        real_color = Fore.GREEN if result['real_probability'] > 50 else Fore.RED
        
        print(f"{Fore.CYAN}Fake Probability   : {fake_color}{result['fake_probability']:.2f}%")
        print(f"{Fore.CYAN}Real Probability   : {real_color}{result['real_probability']:.2f}%")
        
        # Confidence with color
        if result['confidence'] == 'HIGH':
            conf_color = Fore.GREEN
        elif result['confidence'] == 'MEDIUM':
            conf_color = Fore.YELLOW
        else:
            conf_color = Fore.RED
        
        print(f"{Fore.CYAN}Confidence     : {conf_color}{Style.BRIGHT}{result['confidence']}")
        print(f"{Fore.CYAN}Model          : {Fore.MAGENTA}{model_name}")
        print(f"\n{Fore.WHITE}{'='*40}\n")


class FakeNewsDetectionApp:
    """
    Main application class for the Fake News Detection system.
    """
    
    def __init__(self):
        """Initialize the application components."""
        self.data_loader = DataLoader()
        self.preprocessor = TextPreprocessor()
        self.trainer = ModelTrainer()
        self.predictor = None
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.logger = logging.getLogger(__name__)
        
        # Setup NLTK on initialization
        setup_nltk()
    
    def display_header(self) -> None:
        """Display the application header."""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}")
        print("=" * 50)
        print("     FAKE NEWS DETECTION USING NLP")
        print("=" * 50)
        print(Style.RESET_ALL)
    
    def display_menu(self) -> None:
        """Display the main menu options."""
        print(f"\n{Fore.YELLOW}{'─' * 50}")
        print(f"{Fore.WHITE}{Style.BRIGHT}  MAIN MENU")
        print(f"{Fore.YELLOW}{'─' * 50}")
        print(f"{Fore.CYAN}  1. {Fore.WHITE}Load & Explore Dataset")
        print(f"{Fore.CYAN}  2. {Fore.WHITE}Preprocess Text")
        print(f"{Fore.CYAN}  3. {Fore.WHITE}Train Models")
        print(f"{Fore.CYAN}  4. {Fore.WHITE}Load Saved Model")
        print(f"{Fore.CYAN}  5. {Fore.WHITE}Predict News")
        print(f"{Fore.CYAN}  6. {Fore.WHITE}Exit")
        print(f"{Fore.YELLOW}{'─' * 50}")
    
    def option1_load_explore(self) -> None:
        """Load and explore the dataset."""
        print(f"\n{Fore.GREEN}{'='*50}")
        print(f"{Fore.YELLOW}  LOAD & EXPLORE DATASET")
        print(f"{Fore.GREEN}{'='*50}\n")
        
        try:
            # Load dataset
            self.data = self.data_loader.load_dataset()
            
            # Get dataset info
            info = self.data_loader.get_dataset_info()
            
            # Display information
            print(f"{Fore.CYAN}📊 Dataset Information:")
            print(f"{Fore.WHITE}  • Total articles: {Fore.YELLOW}{info['total_samples']:,}")
            print(f"{Fore.WHITE}  • Fake news articles: {Fore.RED}{info['fake_count']:,} ({info['fake_percentage']:.1f}%)")
            print(f"{Fore.WHITE}  • Real news articles: {Fore.GREEN}{info['real_count']:,} ({info['real_percentage']:.1f}%)")
            print(f"{Fore.WHITE}  • Columns: {Fore.YELLOW}{', '.join(info['columns'])}")
            print(f"{Fore.WHITE}  • Memory usage: {Fore.YELLOW}{info['memory_usage']:.2f} MB\n")
            
            # Display missing values
            print(f"{Fore.CYAN}🔍 Missing Values:")
            for col, count in info['missing_values'].items():
                if count > 0:
                    print(f"{Fore.WHITE}  • {col}: {Fore.RED}{count}")
                else:
                    print(f"{Fore.WHITE}  • {col}: {Fore.GREEN}0")
            
            # Display dataset info
            print(f"\n{Fore.CYAN}📋 Dataset Info:")
            print(f"{Fore.WHITE}{self.data.info()}")
            
            # Display basic statistics
            print(f"\n{Fore.CYAN}📈 Dataset Statistics:")
            print(f"{Fore.WHITE}{self.data.describe()}")
            
            # Display class distribution
            print(f"\n{Fore.CYAN}📊 Class Distribution:")
            dist = self.data['label'].value_counts()
            for label, count in dist.items():
                pct = (count / len(self.data)) * 100
                bar = '█' * int(pct / 2)
                if label in ['FAKE', 'FALSE', '0']:
                    color = Fore.RED
                else:
                    color = Fore.GREEN
                print(f"{color}  {label}: {count:,} ({pct:.1f}%) {bar}")
            
            # Create and save visualization
            self.create_data_analysis_plots()
            
            self.logger.info("Dataset loaded and explored successfully")
            
        except Exception as e:
            self.logger.error(f"Error in option 1: {e}")
            print(f"{Fore.RED}Error: {e}")
    
    def create_data_analysis_plots(self) -> None:
        """Create and save data analysis visualizations."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Class Distribution
            class_dist = self.data['label'].value_counts()
            colors = ['#FF6B6B', '#4ECDC4']
            axes[0, 0].pie(class_dist.values, labels=class_dist.index, 
                          autopct='%1.1f%%', colors=colors, startangle=90)
            axes[0, 0].set_title('Class Distribution', fontsize=14, fontweight='bold')
            
            # Plot 2: Text Length Distribution
            self.data['text_length'] = self.data['text'].str.len()
            fake_lengths = self.data[self.data['label_numeric'] == 0]['text_length']
            real_lengths = self.data[self.data['label_numeric'] == 1]['text_length']
            
            axes[0, 1].hist(fake_lengths, bins=30, alpha=0.7, label='Fake', color='#FF6B6B')
            axes[0, 1].hist(real_lengths, bins=30, alpha=0.7, label='Real', color='#4ECDC4')
            axes[0, 1].set_xlabel('Text Length (characters)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Text Length Distribution', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
            
            # Plot 3: Title Length Distribution
            self.data['title_length'] = self.data['title'].str.len()
            fake_title_len = self.data[self.data['label_numeric'] == 0]['title_length']
            real_title_len = self.data[self.data['label_numeric'] == 1]['title_length']
            
            axes[1, 0].hist(fake_title_len, bins=30, alpha=0.7, label='Fake', color='#FF6B6B')
            axes[1, 0].hist(real_title_len, bins=30, alpha=0.7, label='Real', color='#4ECDC4')
            axes[1, 0].set_xlabel('Title Length (characters)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Title Length Distribution', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            
            # Plot 4: Missing Values
            missing = self.data.isnull().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                axes[1, 1].bar(missing.index, missing.values, color='#FF6B6B')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].set_title('Missing Values', fontsize=14, fontweight='bold')
                axes[1, 1].tick_params(axis='x', rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, 'No Missing Values', 
                               ha='center', va='center', fontsize=14)
                axes[1, 1].set_title('Missing Values', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"{Fore.GREEN}✅ Data analysis visualization saved to: data_analysis.png")
            
        except Exception as e:
            self.logger.error(f"Error creating plots: {e}")
            print(f"{Fore.YELLOW}Warning: Could not create some plots: {e}")
    
    def option2_preprocess(self) -> None:
        """Preprocess the text data."""
        print(f"\n{Fore.GREEN}{'='*50}")
        print(f"{Fore.YELLOW}  PREPROCESS TEXT")
        print(f"{Fore.GREEN}{'='*50}\n")
        
        if self.data is None:
            print(f"{Fore.RED}Please load the dataset first (Option 1).")
            return
        
        try:
            # Display sample before preprocessing
            print(f"{Fore.CYAN}📝 Sample text BEFORE preprocessing:")
            print(f"{Fore.WHITE}{self.data['combined_text'].iloc[0][:200]}...\n")
            
            # Preprocess the corpus
            self.data['processed_text'] = self.preprocessor.preprocess_corpus(
                self.data['combined_text']
            )
            
            # Display sample after preprocessing
            print(f"{Fore.CYAN}📝 Sample text AFTER preprocessing:")
            print(f"{Fore.WHITE}{self.data['processed_text'].iloc[0][:200]}...\n")
            
            # Split data
            X = self.data['processed_text']
            y = self.data['label_numeric']
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"{Fore.GREEN}✅ Text preprocessing complete!")
            print(f"{Fore.WHITE}  • Training samples: {Fore.YELLOW}{len(self.X_train):,}")
            print(f"{Fore.WHITE}  • Testing samples: {Fore.YELLOW}{len(self.X_test):,}")
            
            self.logger.info("Text preprocessing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            print(f"{Fore.RED}Error during preprocessing: {e}")
    
    def option3_train_models(self) -> None:
        """Train and evaluate ML models."""
        print(f"\n{Fore.GREEN}{'='*50}")
        print(f"{Fore.YELLOW}  TRAIN MODELS")
        print(f"{Fore.GREEN}{'='*50}\n")
        
        if self.X_train is None or self.y_train is None:
            print(f"{Fore.RED}Please preprocess the data first (Option 2).")
            return
        
        try:
            # Vectorize the text data
            print(f"{Fore.CYAN}🔄 Vectorizing text data...")
            X_train_vec = self.preprocessor.vectorize(self.X_train, fit=True)
            X_test_vec = self.preprocessor.vectorize(self.X_test, fit=False)
            
            print(f"{Fore.GREEN}✅ Vectorization complete!")
            print(f"{Fore.WHITE}  • Training set shape: {Fore.YELLOW}{X_train_vec.shape}")
            print(f"{Fore.WHITE}  • Testing set shape: {Fore.YELLOW}{X_test_vec.shape}\n")
            
            # Train and evaluate models
            print(f"{Fore.CYAN}🤖 Training models...")
            self.trainer.train_and_evaluate(X_train_vec, X_test_vec, 
                                           self.y_train, self.y_test)
            
            # Display results
            self.trainer.display_results()
            
            # Plot and save confusion matrices
            self.trainer.plot_confusion_matrices()
            
            # Save best model
            self.trainer.save_models()
            
            # Save vectorizer
            joblib.dump(self.preprocessor.vectorizer, 'tfidf_vectorizer.pkl')
            print(f"{Fore.GREEN}✅ Vectorizer saved to: tfidf_vectorizer.pkl")
            
            # Initialize predictor
            self.predictor = NewsPredictor(
                self.trainer.best_model,
                self.preprocessor.vectorizer,
                self.preprocessor
            )
            
            self.logger.info("Model training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            print(f"{Fore.RED}Error during model training: {e}")
    
    def option4_load_model(self) -> None:
        """Load a previously saved model."""
        print(f"\n{Fore.GREEN}{'='*50}")
        print(f"{Fore.YELLOW}  LOAD SAVED MODEL")
        print(f"{Fore.GREEN}{'='*50}\n")
        
        try:
            # Check if model files exist
            if not os.path.exists('best_model.pkl'):
                print(f"{Fore.RED}Model file 'best_model.pkl' not found!")
                print(f"{Fore.YELLOW}Please train models first (Option 3).")
                return
            
            if not os.path.exists('tfidf_vectorizer.pkl'):
                print(f"{Fore.RED}Vectorizer file 'tfidf_vectorizer.pkl' not found!")
                print(f"{Fore.YELLOW}Please train models first (Option 3).")
                return
            
            # Load model and vectorizer
            model = joblib.load('best_model.pkl')
            vectorizer = joblib.load('tfidf_vectorizer.pkl')
            
            # Update preprocessor's vectorizer
            self.preprocessor.vectorizer = vectorizer
            
            # Create predictor
            self.predictor = NewsPredictor(model, vectorizer, self.preprocessor)
            
            print(f"{Fore.GREEN}✅ Model and vectorizer loaded successfully!")
            print(f"{Fore.WHITE}  • Model type: {Fore.YELLOW}{type(model).__name__}")
            print(f"{Fore.WHITE}  • Vectorizer features: {Fore.YELLOW}{vectorizer.max_features}")
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            print(f"{Fore.RED}Error loading model: {e}")
    
    def option5_predict_news(self) -> None:
        """Predict fake/real news."""
        print(f"\n{Fore.GREEN}{'='*50}")
        print(f"{Fore.YELLOW}  PREDICT NEWS")
        print(f"{Fore.GREEN}{'='*50}\n")
        
        if self.predictor is None:
            print(f"{Fore.RED}No model loaded. Please train or load a model first.")
            return
        
        # Prediction sub-menu
        print(f"{Fore.CYAN}Choose prediction method:")
        print(f"{Fore.WHITE}  1. Enter custom news text")
        print(f"{Fore.WHITE}  2. Use sample from dataset")
        
        try:
            choice = input(f"\n{Fore.YELLOW}Enter your choice (1-2): {Fore.WHITE}").strip()
            
            if choice == '1':
                # Custom text input
                print(f"\n{Fore.CYAN}Enter news text (press Enter twice to finish):")
                lines = []
                while True:
                    line = input()
                    if line == '' and lines and lines[-1] == '':
                        break
                    lines.append(line)
                
                news_text = '\n'.join(lines).strip()
                
                if not news_text:
                    print(f"{Fore.RED}Error: Empty text provided.")
                    return
                
            elif choice == '2':
                # Sample from dataset
                if self.data is None:
                    print(f"{Fore.RED}Dataset not loaded. Please load dataset first (Option 1).")
                    return
                
                print(f"\n{Fore.CYAN}Select sample type:")
                print(f"{Fore.WHITE}  1. Random FAKE news")
                print(f"{Fore.WHITE}  2. Random REAL news")
                
                sample_choice = input(f"\n{Fore.YELLOW}Enter your choice (1-2): {Fore.WHITE}").strip()
                
                if sample_choice == '1':
                    sample = self.data[self.data['label_numeric'] == 0].sample(1)
                    actual_label = "FAKE"
                elif sample_choice == '2':
                    sample = self.data[self.data['label_numeric'] == 1].sample(1)
                    actual_label = "REAL"
                else:
                    print(f"{Fore.RED}Invalid choice.")
                    return
                
                news_text = sample['combined_text'].values[0]
                print(f"\n{Fore.CYAN}Sample text:")
                print(f"{Fore.WHITE}{news_text[:300]}...")
                print(f"\n{Fore.MAGENTA}Actual label: {actual_label}")
                
            else:
                print(f"{Fore.RED}Invalid choice.")
                return
            
            # Make prediction
            if self.trainer.best_model_name:
                model_name = self.trainer.best_model_name
            else:
                model_name = "Saved Model"
            
            result = self.predictor.predict(news_text)
            self.predictor.display_prediction(result, model_name)
            
            self.logger.info(f"Prediction made: {result['prediction']} "
                           f"(Fake: {result['fake_probability']:.2f}%, "
                           f"Real: {result['real_probability']:.2f}%)")
            
        except ValueError as e:
            print(f"{Fore.RED}{e}")
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            print(f"{Fore.RED}Error during prediction: {e}")
    
    def run(self) -> None:
        """
        Main application loop.
        """
        self.display_header()
        
        while True:
            self.display_menu()
            
            try:
                choice = input(f"\n{Fore.YELLOW}Enter your choice (1-6): {Fore.WHITE}").strip()
                
                if choice == '1':
                    self.option1_load_explore()
                elif choice == '2':
                    self.option2_preprocess()
                elif choice == '3':
                    self.option3_train_models()
                elif choice == '4':
                    self.option4_load_model()
                elif choice == '5':
                    self.option5_predict_news()
                elif choice == '6':
                    print(f"\n{Fore.GREEN}Thank you for using Fake News Detection System!")
                    print(f"{Fore.CYAN}Goodbye! 👋\n")
                    sys.exit(0)
                else:
                    print(f"{Fore.RED}Invalid choice! Please enter a number between 1 and 6.")
                
                # Pause before showing menu again
                if choice != '6':
                    input(f"\n{Fore.YELLOW}Press Enter to continue...")
                    
            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}Program interrupted by user.")
                print(f"{Fore.CYAN}Goodbye! 👋\n")
                sys.exit(0)
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                print(f"{Fore.RED}An unexpected error occurred: {e}")
                print(f"{Fore.YELLOW}Please try again.")


def generate_sample_dataset():
    """
    Generate a sample dataset for testing if no dataset is provided.
    This ensures the application can be demonstrated even without real data.
    """
    import random
    
    if os.path.exists('news.csv'):
        return
    
    print(f"{Fore.YELLOW}No dataset found. Generating sample dataset for demonstration...")
    
    # Sample fake and real news headlines and texts
    fake_templates = [
        "SHOCKING: {celebrity} Found {action} with {object}! You Won't Believe What Happened Next!",
        "BREAKING: {politician} Secretly {action} {object} - The Truth Exposed!",
        "Doctors Are Speechless: This Simple {object} Cures {disease} in Just Days!",
        "URGENT: {celebrity}'s Dark Secret About {object} Finally Revealed!",
        "The Government Doesn't Want You to Know About This {object} Discovery!",
        "Scientists Baffled: {object} Found to {action} Completely Naturally!",
        "You've Been Lied To: The REAL Truth About {object} That {celebrity} Hid!",
        "WARNING: This Common {object} Could {action} Your Health Immediately!",
        "EXCLUSIVE: {politician}'s Secret Meeting About {object} LEAKED!",
        "They Don't Want You to See This: {celebrity}'s {object} Scandal EXPOSED!"
    ]
    
    real_templates = [
        "City Council Approves New {object} Development Project in Downtown Area",
        "Study Shows Improved Results for {disease} Treatment Using {object}",
        "Local {celebrity} Announces New Initiative for Community Development",
        "Economic Report Indicates Growth in {object} Sector This Quarter",
        "Scientists Publish Research on Effects of {object} on Climate Change",
        "Government Introduces New Legislation Regarding {object} Safety Standards",
        "International Conference on {object} Held in Major City This Week",
        "Education Board Approves Updated Curriculum Including {object} Studies",
        "New Technology in {object} Promises to Revolutionize Industry Standards",
        "Health Officials Report Decrease in {disease} Cases After New Measures"
    ]
    
    celebrities = ['Elon Musk', 'Taylor Swift', 'Cristiano Ronaldo', 'Beyoncé', 
                   'Jeff Bezos', 'Kim Kardashian', 'Leonardo DiCaprio', 'Oprah Winfrey']
    politicians = ['Joe Biden', 'Donald Trump', 'Barack Obama', 'Nancy Pelosi',
                   'Boris Johnson', 'Emmanuel Macron', 'Justin Trudeau']
    actions = ['destroying', 'creating', 'hiding', 'revealing', 'investigating', 
               'developing', 'testing', 'discovering', 'launching', 'canceling']
    objects = ['vaccine', 'technology', 'energy device', 'water purifier', 'diet pill',
               'financial system', 'social media platform', 'AI robot', 'crypto currency',
               'space craft', 'medical breakthrough', 'electric car']
    diseases = ['cancer', 'diabetes', 'heart disease', 'COVID-19', 'Alzheimer\'s',
                'arthritis', 'depression', 'anxiety', 'obesity', 'asthma']
    
    np.random.seed(42)
    n_samples = 200  # Generate 200 samples (100 fake + 100 real)
    
    data = []
    
    # Generate fake news
    for _ in range(n_samples // 2):
        template = np.random.choice(fake_templates)
        text = template.format(
            celebrity=np.random.choice(celebrities),
            politician=np.random.choice(politicians),
            action=np.random.choice(actions),
            object=np.random.choice(objects),
            disease=np.random.choice(diseases)
        )
        # Add some extra text to simulate real articles
        extra_text = " ".join([np.random.choice(actions) + " " + np.random.choice(objects) 
                               for _ in range(20)])
        full_text = text + " " + extra_text
        data.append({
            'title': text[:100],
            'text': full_text,
            'label': 'FAKE'
        })
    
    # Generate real news
    for _ in range(n_samples // 2):
        template = np.random.choice(real_templates)
        text = template.format(
            celebrity=np.random.choice(celebrities),
            politician=np.random.choice(politicians),
            action=np.random.choice(actions),
            object=np.random.choice(objects),
            disease=np.random.choice(diseases)
        )
        extra_text = " ".join([np.random.choice(actions) + " " + np.random.choice(objects) 
                               for _ in range(20)])
        full_text = text + " " + extra_text
        data.append({
            'title': text[:100],
            'text': full_text,
            'label': 'REAL'
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
    df.to_csv('news.csv', index=False)
    
    print(f"{Fore.GREEN}✅ Sample dataset generated: news.csv ({len(df)} samples)")


def main():
    """
    Main entry point of the application.
    """
    # Setup logging
    global logger
    logger = setup_logging()
    
    try:
        # Display startup message
        print(f"{Fore.CYAN}{Style.BRIGHT}")
        print("╔══════════════════════════════════════════════╗")
        print("║     FAKE NEWS DETECTION USING NLP            ║")
        print("║     Version 1.0 - Python 3.13                ║")
        print("╚══════════════════════════════════════════════╝")
        print(Style.RESET_ALL)
        
        # Check and generate sample dataset if needed
        generate_sample_dataset()
        
        # Create and run the application
        app = FakeNewsDetectionApp()
        app.run()
        
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Program interrupted by user.")
        print(f"{Fore.CYAN}Goodbye! 👋\n")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"{Fore.RED}Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()