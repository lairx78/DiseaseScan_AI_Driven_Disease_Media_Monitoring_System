# DiseaseScan: AI-Driven Disease Media Monitoring System

**DiseaseScan** is an AI-powered system designed for the automated monitoring, classification, and analysis of communicable disease information within digital news. Developed at the **University of Hong Kong, School of Computing and Data Science**, this system integrates Natural Language Processing (NLP), Large Language Models (LLMs), and orchestration frameworks like **LangGraph** and **LangChain** to transform unstructured media content into actionable public health intelligence.

## 📖 Overview

In the modern digital environment, public health officials are inundated with vast amounts of online news, making it difficult to rapidly detect disease outbreaks. Conventional monitoring systems are often reactive and struggle with unstructured content.

**DiseaseScan** solves this by automating the workflow. It operates in two primary modes:
1.  **Conversational Agent:** A chatbot UI for real-time analysis of specific URLs, HTML files, or text, capable of answering follow-up health questions.
2.  **Media Monitoring Pipeline:** An end-to-end batch processing system that ingests Excel records and stores structured results in a SQLite database.

## ✨ Key Features

*   **Automated Classification:** Distinguishes between disease-related and non-disease-related news with **98.33% accuracy**.
*   **Multi-Input Support:** Seamlessly processes inputs via **URL**, **Local HTML files**, or **Plain Text**.
*   **Smart Entity Extraction:** Precisely extracts **Disease Names** and **Article Titles**, filtering out web noise (navigation menus, footers, ads).
*   **Intelligent Summarization:** Generates concise summaries capturing crucial outbreak details.
*   **Report Generation:** Automatically creates formatted **Word reports (.docx)** for downstream analysis.
*   **Contextual Q&A:** The agent allows users to ask follow-up questions strictly related to the health content of the analyzed article.
*   **Batch Processing Pipeline:** Ingests news records from Excel and saves structured data (ID, Classification, Title, Disease Name, Summary) to **SQLite**.

## 🏗️ System Architecture

DiseaseScan utilizes a multi-agent workflow orchestrated by **LangGraph**.

### 1. Conversational Agent
The agent uses a state-machine approach to decide the next step based on user input. It utilizes three specialized tools:
*   **Identify Input Type:** Determines if input is a URL, HTML file, or Text using Regex/String analysis.
*   **Fetch and Save HTML:** Retrieves web content (with user-agent spoofing) and saves it locally to prevent redundant requests.
*   **Analyze Communicable Disease Article:** The core analysis engine that classifies content, extracts entities, and generates reports using structured output.

### 2. Media Monitoring Pipeline
A wrapper around the agent that:
1.  Reads an Excel file of news sources.
2.  Creates timestamped working directories to isolate batches.
3.  Generates unique Thread IDs.
4.  Invokes the agent for each record.
5.  Stores results in a centralized SQLite database.

## 🛠️ Tech Stack

*   **Orchestration:** LangChain, LangGraph
*   **LLMs (Google Gemini):**
    *   *Orchestration:* Gemini 2.5 Flash Preview 04-17
    *   *Analysis:* Gemini 2.0 Flash-Lite
*   **Database:** SQLite
*   **Data Processing:** Pandas (Excel/DataFrame manipulation)
*   **Output Generation:** python-docx
*   **Web Scraping:** BeautifulSoup / Requests

## 🚀 Getting Started

### Prerequisites
*   Python 3.9+
*   Google Gemini API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/DiseaseScan.git
    cd DiseaseScan
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Setup:**
    Create a `.env` file in the root directory and add your API key:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```

## 💡 Usage

### Running the Conversational Agent
Start the UI (e.g., via Streamlit or Chainlit depending on implementation):
```bash
python app.py

Workflow:

Input a URL, HTML filename, or text snippet.
The agent identifies the type and processes the content.
Receive a summary, classification, and Word report.
Ask follow-up questions like "What symptoms are mentioned?"
Running the Media Monitoring Pipeline
To process a batch of news articles:

Place your source Excel file in the data/ directory.
Run the pipeline script:
bash
python pipeline.py --input data/news_records.xlsx

Results will be saved to results.db (SQLite) and can be exported to a DataFrame.
📊 Performance Evaluation
The system was evaluated on a testing dataset of 60 news articles (50/50 split of disease/non-disease related).

Classification Accuracy: 98.33% (59/60 correctly classified).
Entity Extraction: High precision in capturing disease names and titles, even in cluttered HTML.
Prompt Engineering: Utilizes robust System Prompts and Few-Shot Prompting to ensure reliability and strict adherence to the health-analysis role.
🔮 Future Work
Local LLM Integration: To reduce API costs and latency.
Real-time Alerting: Integration with notification systems for immediate outbreak signals.
Enhanced Entity Extraction: Improving recall for secondary disease mentions in complex articles.
