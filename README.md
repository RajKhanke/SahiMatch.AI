# SahiMatch.AI  
**An AI Powered Partial Invoice Matcher**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/yourusername/SahiMatch.AI/actions)  
[![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

---

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Share](#share)

---

## Overview
**SahiMatch.AI** is an AI-driven partial invoice matching system that leverages a combinatory approach of Machine Learning (ML), Natural Language Processing (NLP), and fuzzy logic. It is designed to intelligently match invoice numbers between two datasets—even when confronted with challenges such as partial data, formatting differences, or human errors. The system provides detailed reports on classification and similarity between invoices and features a drag-and-drop interface for uploading files, editing predictions, and manual review. Through reinforcement learning, it continuously improves over time as it learns from user corrections.

---

## Problem Statement
**Intelligent Partial Invoice Matching**

Matching invoice numbers between two datasets is inherently challenging due to:
- **Partial Data:** Invoices may be truncated or missing prefixes/suffixes.
- **Discrepancies:** Minor formatting differences, extra spaces, special characters, or human errors.
- **Multiple Potential Matches:** Situations where multiple entries appear similar, leading to ambiguity.

**Objective:**  
Develop an AI/ML-powered solution that:
- **Automatically matches** invoice numbers even when only partial or inconsistent data is available.
- **Recommends potential matches** with a confidence score (on a scale from 1 to 100) for entries with low certainty.
- **Facilitates manual matching** via an intuitive drag-and-drop interface, enabling users to edit, freeze, or confirm predictions.
- **Learns from user corrections** using reinforcement learning to continuously improve matching accuracy over time.

**Expected Outcomes:**
- **High accuracy** in identifying exact, partial, and unmatched invoice entries.
- A **self-learning model** that adapts and refines its predictions through ongoing user feedback.
- A **comprehensive final report** (e.g., an Excel sheet) detailing which invoice entries were matched, along with their confidence scores and underlying reasoning.

---

## Key Features
- **🤖 AI-Powered Matching:**  
  Utilizes advanced ML models such as Siamese Neural Networks, Random Forest, and XGBoost to accurately detect similarities and discrepancies.

- **🔍 Intelligent Partial Matching:**  
  Applies robust NLP techniques including:
  - Inuise NLP, TextBlob, and regex for cleaning and pattern matching.
  - Stopwords removal, stemming, and lemmatization.
  - Keyword extraction tools like PK/YAKE.
  - Word embeddings, cosine similarity, and Sentence-BERT for semantic comparison.

- **🔢 Hybrid Fuzzy Matching:**  
  Integrates multiple fuzzy matching algorithms such as:
  - Jacob Matrix
  - Hamming Distance
  - Jaro-Winkler
  - Levenshtein Distance
  - RapidFuzz and FuzzyWuzzy

- **📊 Confidence Scoring:**  
  Generates match recommendations with confidence scores (1–100), including reasons for each score, to help users gauge match reliability.

- **✍️ Manual Correction Interface:**  
  Features an intuitive drag-and-drop UI with options to:
  - Edit predictions manually.
  - Freeze certain entries.
  - Use checkboxes for quick selection during review.

- **🔄 Self-Learning via Reinforcement Learning:**  
  Continuously improves model performance by incorporating user feedback and manual corrections into subsequent iterations.

- **⚡ Scalable & Optimized:**  
  Designed for large-scale data processing using parallel processing and efficient hashing techniques to minimize time complexity.

- **📈 Real-Time Analytics & Reporting:**  
  Provides dashboards for instant insights into matching accuracy and workflow efficiency, along with detailed reports in Excel format.

---

## Architecture
SahiMatch.AI is built on a modular architecture that consists of:

- **Backend:**
  - **Matching Engine:**  
    Integrates ML models, NLP preprocessing, and fuzzy matching algorithms to process and compare invoice data.
  - **Reinforcement Learning Module:**  
    Continuously retrains and optimizes the matching model based on manual corrections and user feedback.
  - **API Layer:**  
    Offers RESTful endpoints for seamless interaction between the backend and frontend components.

- **Frontend:**
  - **User Interface:**  
    Developed with HTML, CSS, and JavaScript, featuring a drag-and-drop system for file uploads and manual corrections.
  - **Real-Time Reporting:**  
    Displays match results, confidence scores, and analytical dashboards in real time.

---

## Technology Stack
- **Machine Learning:**
  - **Models:** Siamese Neural Networks, Random Forest, XGBoost, etc.
  - **Frameworks:** TensorFlow, PyTorch

- **Natural Language Processing (NLP):**
  - **Libraries & Tools:**  
    Inuise NLP, TextBlob, regex for pattern matching, stopwords removal, stemming, lemmatization, PK/YAKE, word embeddings, cosine similarity, Sentence-BERT

- **Fuzzy Matching Algorithms:**
  - Jacob Matrix, Hamming Distance, Jaro-Winkler, Levenshtein Distance, RapidFuzz, FuzzyWuzzy

- **Reinforcement Learning:**
  - Self-learning algorithms that refine predictions based on user corrections.

- **Optimization Techniques:**
  - Parallel processing and efficient hashing to ensure scalability and reduce processing time for large datasets.

---

## Setup & Installation

### Backend
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/SahiMatch.AI.git
