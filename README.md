<p align="center">
  <img src="https://img.shields.io/badge/University-of%20Tehran-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Course-NLP-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=white" />
</p>

<h1 align="center">📝 Natural Language Processing — Course Projects</h1>

<p align="center">
  <b>A comprehensive collection of NLP projects covering the full spectrum from classical text processing to modern LLM-based applications.</b>
</p>

<p align="center"><i>University of Tehran · Fall 2025</i></p>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Projects at a Glance](#-projects-at-a-glance)
- [CA1 — Tokenization & N-gram Language Modeling](#-ca1--tokenization--n-gram-language-modeling)
- [CA2 — Text Classification with Logistic Regression & Naive Bayes](#-ca2--text-classification-with-logistic-regression--naive-bayes)
- [CA3 — Word2Vec & Neural Text Classification](#-ca3--word2vec--neural-text-classification)
- [CA4 — Transformer Fine-Tuning](#-ca4--transformer-fine-tuning)
- [CA5 — LLM-Powered Applications](#-ca5--llm-powered-applications)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)

---

## 🔭 Overview

This repository contains five progressive assignments from the **Natural Language Processing** course at the University of Tehran. The projects trace a learning path from foundational NLP concepts to cutting-edge LLM applications:

```
Classical NLP ──────────────────────────────────────────▶ Modern LLM Apps

  CA1              CA2              CA3              CA4              CA5
┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐
│ Regex   │   │ Logistic │   │ Word2Vec │   │ BART &   │   │ Travel Agent │
│ Tokens  │──▶│ Regress. │──▶│ CBOW &   │──▶│ GPT-2    │──▶│ Legal RAG    │
│ N-grams │   │ Naive    │   │ Skip-Gram│   │ LoRA     │   │ LangGraph    │
│         │   │ Bayes    │   │ FastText │   │ IFEval   │   │ LanceDB      │
└─────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────────┘
```

---

## 🗂 Projects at a Glance

| # | Project | Topics | Key Techniques |
|---|---------|--------|----------------|
| **CA1** | Tokenization & N-gram LM | Regex, BPE, WordPiece, N-gram, Perplexity | Rule-based tokenizer, BPE/WordPiece from scratch, smoothing methods |
| **CA2** | Text Classification | Spam detection, URL classification, BoW, TF-IDF | Logistic Regression & Naive Bayes (from scratch + sklearn) |
| **CA3** | Word Embeddings & MLP | CBOW, Skip-Gram, FastText, News classification | Word2Vec from scratch, MLP classifier with pretrained embeddings |
| **CA4** | Transformer Fine-Tuning | Text-to-SQL, Instruction following, LoRA, IFEval | BART vs GPT-2 comparison, TinyLlama fine-tuning with LoRA |
| **CA5** | LLM Applications | Travel chatbot, Legal RAG system, ReAct agents | LangGraph agents, LanceDB vector search, multi-tool orchestration |

---

## 📌 CA1 — Tokenization & N-gram Language Modeling

> **Foundations of text processing and statistical language modeling**

### Topics Covered

| Section | Description |
|---------|-------------|
| **Regular Expressions & Edit Distance** | Email validation with regex patterns; auto-correction using Minimum Edit Distance |
| **Tokenization** | Rule-based tokenizer, BPE (Byte Pair Encoding), WordPiece — all implemented from scratch |
| **N-gram Language Modeling** | Unigram/Bigram/Trigram models, Perplexity evaluation, Laplace & Interpolation smoothing, Temperature simulation |

### Highlights
- Built a **BPE tokenizer** and **WordPiece tokenizer** from scratch without libraries
- Implemented **N-gram language models** with multiple smoothing techniques
- Compared tokenization strategies with visual analysis

📂 **Directory:** [`CA1/`](CA1/)

---

## 📌 CA2 — Text Classification with Logistic Regression & Naive Bayes

> **Classical ML approaches to text classification**

### Topics Covered

| Section | Description |
|---------|-------------|
| **Q1: From-Scratch Implementation** | Email spam detection — preprocess text with Bag of Words, implement Logistic Regression & Naive Bayes from scratch, evaluate with Accuracy/Precision/Recall/F1 |
| **Q2: Feature Engineering** | URL phishing detection — extract structural, statistical, and content-based features; classify with sklearn models |

### Highlights
- **Logistic Regression** and **Naive Bayes** classifiers implemented from scratch (no sklearn for Q1)
- Custom evaluation metrics (Accuracy, Precision, Recall, F1) built manually
- Extensive **feature engineering** for URL classification including structural, statistical, and custom features

📂 **Directory:** [`CA2/`](CA2/)

---

## 📌 CA3 — Word2Vec & Neural Text Classification

> **Learning word representations and applying them to downstream tasks**

### Topics Covered

| Section | Description |
|---------|-------------|
| **Q1: Word2Vec from Scratch** | Implement CBOW and Skip-Gram models from scratch on WikiText-2; compare learned embeddings |
| **Q2: News Classification** | Use pretrained FastText embeddings with an MLP neural network for AG News topic classification |

### Highlights
- **CBOW** and **Skip-Gram** architectures implemented from scratch with NumPy/PyTorch
- Trained on **WikiText-2** dataset with custom preprocessing pipeline
- News classifier achieving strong results using **FastText** embeddings + MLP
- Embedding quality analysis with similarity and analogy tasks

📂 **Directory:** [`CA3/`](CA3/)

---

## 📌 CA4 — Transformer Fine-Tuning

> **Exploring modern transformer architectures for generation and instruction following**

### Part 1: Comparing Encoder-Decoder and Decoder-Only Models for Text-to-SQL

A comparative study of **BART** (Encoder-Decoder) vs **GPT-2** (Decoder-Only) on the Text-to-SQL task.

| Aspect | BART | GPT-2 |
|--------|------|-------|
| Architecture | Bidirectional encoder + autoregressive decoder | Causal left-to-right language model |
| Input Processing | Cross-attention over encoded context | Prefix-based sequence continuation |
| Dataset | Gretel Synthetic Text-to-SQL (20K train / 2K test) | Same |

📂 **Directory:** [`CA4/Part1/`](CA4/Part1/) · [Detailed README](CA4/Part1/README.md)

### Part 2: Improving Instruction-Following in LLMs

Fine-tuning **TinyLlama-1.1B** with **LoRA** (Low-Rank Adaptation) on instruction-following data, evaluated with the **IFEval** benchmark.

| Component | Details |
|-----------|---------|
| Base Model | TinyLlama-1.1B-Chat |
| Training Method | SFT + LoRA (Parameter-Efficient Fine-Tuning) |
| Dataset | Tulu 3 SFT Personas Instruction-Following |
| Evaluation | IFEval benchmark via lm-eval-harness |

📂 **Directory:** [`CA4/Part2/`](CA4/Part2/) · [Detailed README](CA4/Part2/README.md)

---

## 📌 CA5 — LLM-Powered Applications

> **Building end-to-end intelligent systems with LLMs, RAG, and agentic architectures**

### Part 1: TravelBot — AI-Powered Travel Planning Assistant

An intelligent multi-tool chatbot built with **LangGraph's ReAct architecture**, combining real-time APIs, semantic search, and LLM-powered planning.

| Tool | Function | Source |
|------|----------|--------|
| ✈️ Flight Search | Search flights with pricing & schedules | Amadeus API |
| 🏨 Hotel Search | Find hotels with ratings & location | Amadeus API + Tavily |
| 🍽️ Restaurant Search | Discover local dining options | Tavily + LLM |
| 🌤️ Weather Forecast | Get weather & clothing advice | Tavily + LLM |
| 💱 Currency Exchange | Real-time exchange rates | Tavily + LLM |
| ❓ FAQ Search | Semantic search over travel FAQs | LanceDB (RAG) |
| 🗺️ Trip Planner | Full day-by-day itinerary generation | RAG + Tavily + LLM |

📂 **Directory:** [`CA5/Part1/`](CA5/Part1/) · [Detailed README](CA5/Part1/README.md)

### Part 2: Intelligent Response System Based on Country Laws

An advanced **RAG system** for answering legal and constitutional questions based on Persian law documents.

| Feature | Description |
|---------|-------------|
| 6-Node LangGraph Pipeline | Query rewriting → Intent classification → Metadata extraction → Retrieval → Reranking → Generation |
| Document Processing | Hybrid PDF extraction with OCR, smart chunking, Persian text normalization |
| Vector Database | LanceDB with multilingual-e5-large embeddings |
| Coverage | 8 major Persian legal domains (Labor, Check, Rent, Tax, etc.) |
| Quality Assurance | RAGAS evaluation metrics, timing analysis, comprehensive logging |

📂 **Directory:** [`CA5/Part2/`](CA5/Part2/) · [Detailed README](CA5/Part2/README.md)

---

## 🧰 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Languages** | Python, JavaScript |
| **Deep Learning** | PyTorch, HuggingFace Transformers, PEFT (LoRA) |
| **NLP Libraries** | NLTK, scikit-learn, FastText, Gensim |
| **LLM & Agents** | LangGraph, LangChain, OpenAI API |
| **Vector Databases** | LanceDB |
| **APIs** | Amadeus (flights/hotels), Tavily (web search) |
| **Evaluation** | lm-eval-harness (IFEval), RAGAS, custom metrics |
| **Data** | Pandas, NumPy, Matplotlib |

---

## 📁 Repository Structure

```
├── CA1/                          # Tokenization & N-gram Language Modeling
│   ├── CA1.ipynb                 # Main notebook
│   ├── vocab.txt                 # Vocabulary file
│   └── images/                   # Visualizations
│
├── CA2/                          # Text Classification (LR & NB)
│   ├── CA2.ipynb                 # Main notebook
│   └── datasets/
│       ├── q1/                   # Email spam detection data
│       └── q2/                   # URL classification data
│
├── CA3/                          # Word2Vec & Neural Classification
│   ├── NLP_CA3_*.ipynb           # Main notebook
│   ├── wikitext-2-train.txt      # Training corpus
│   ├── wikitext-2-valid.txt      # Validation corpus
│   └── wikitext-2-test.txt       # Test corpus
│
├── CA4/                          # Transformer Fine-Tuning
│   ├── Part1/                    # BART vs GPT-2 for Text-to-SQL
│   │   ├── code.ipynb
│   │   └── README.md
│   └── Part2/                    # LLM Instruction Following with LoRA
│       ├── code.ipynb
│       └── README.md
│
├── CA5/                          # LLM-Powered Applications
│   ├── Part1/                    # TravelBot (ReAct Agent)
│   │   ├── TravelBot.ipynb
│   │   ├── FAQ.js
│   │   ├── requirements.txt
│   │   └── README.md
│   └── Part2/                    # Legal RAG System
│       ├── system.ipynb
│       ├── Data/
│       └── README.md
│
└── README.md                     # This file
```

---

## 🚀 Getting Started

Each project is self-contained in its own directory. To run any notebook:

1. **Clone the repository**
   ```bash
   git clone https://github.com/Amink8203/Natural-Language-Processing-Course-Projects.git
   cd Natural-Language-Processing-Course-Projects
   ```

2. **Install dependencies** — each project may require different packages. Check the notebook's first cells or the `requirements.txt` file (if available).

3. **Open the notebook** in Jupyter or VS Code and run the cells sequentially.

> **Note:** Some projects (CA4, CA5) require GPU access and API keys. Refer to the individual README files in each subdirectory for detailed setup instructions.