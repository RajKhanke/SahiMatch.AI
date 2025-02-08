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


# Setup & Installation

## Backend

### Clone the Repository:
~~~bash
git clone https://github.com/yourusername/SahiMatch.AI.git
~~~

### Navigate to the Project Directory:
~~~bash
cd SahiMatch.AI
~~~

### Install Dependencies:
~~~bash
pip install -r requirements.txt
~~~

### Run the Application:
~~~bash
python app.py
~~~

## Frontend

### Open the Frontend Interface:
The main UI file is `index.html` located in the root directory.

### Run on a Live Server:
You can use a live server extension (e.g., in VS Code) or run a simple HTTP server:
~~~bash
python -m http.server 8000
~~~
Open your browser and navigate to `http://localhost:8000`.

## Usage

### Automated Matching:
- Upload your invoice datasets using the drag-and-drop feature.
- The system will automatically perform matching analysis, compute similarity scores, and display results with corresponding confidence scores.

### Manual Review & Corrections:
- Use the user-friendly interface to adjust and fine-tune uncertain matches.
- The system records these manual edits to reinforce and improve future predictions.

### Reporting:
After processing, a detailed report (e.g., an Excel sheet) is generated. This report outlines:
- Matched entries and their confidence scores.
- The rationale behind each match (e.g., reasons for score assignments).
- Recommendations for manual review where necessary.

## Contributing

Contributions to SahiMatch.AI are welcome! To contribute:

### Fork the Repository

### Create a Feature Branch:
~~~bash
git checkout -b feature/YourFeature
~~~

### Commit Your Changes:
~~~bash
git commit -m "Add new feature or fix issue"
~~~

### Push to Your Branch:
~~~bash
git push origin feature/YourFeature
~~~

### Open a Pull Request:
Please refer to our `CONTRIBUTING.md` for detailed guidelines.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

### Open Source Community:
Thanks to the developers of the open-source libraries and frameworks that have made this project possible.

### Inspiration:
Inspired by real-world challenges in invoice reconciliation and the need for intelligent, scalable matching solutions.

### Support:
Special thanks to all early adopters and contributors for their invaluable feedback and support.

## Share

If you find SahiMatch.AI useful, please consider sharing it with your network!

- **GitHub**: Fork and Star the Repository.
- **Social Media**: Share on Twitter, LinkedIn, or your favorite platform with the hashtag `#SahiMatchAI`.
- **Community**: Let us know your feedback or contribute to the project by opening an issue or submitting a pull request.

**Happy Matching! 🚀**
