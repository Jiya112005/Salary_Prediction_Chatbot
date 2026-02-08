# Salary_Prediction_Chatbot
A comprehensive career intelligence tool that combines Classic Machine Learning for salary forecasting with a Local Generative AI (RAG) for personalized career coaching.

💼 AI Career Coach & Salary Predictor

A comprehensive career intelligence tool that combines Classic Machine Learning for salary forecasting with a Local Generative AI (RAG) for personalized career coaching.

This application runs entirely locally using Ollama, ensuring data privacy and zero API costs.

🌟 Features

1. 💰 Salary Prediction Engine

ML Model: Uses a pre-trained regression model (salary_pipeline.pkl) to estimate salary based on:

Years of Experience

Education Level

Job Title

Age & Gender

Used RandomForestRegressor to assess the salary based on the multiple decision trees affected by various categories of the factors mentioned in features.
Result: Provides an immediate estimated annual compensation package.

2. 🤖 RAG-Based AI Career Coach

Local LLM: Powered by Qwen2.5 (via Ollama) and LangChain.

Context-Aware: The chatbot "remembers" your predicted salary and job profile to give the salary ranges and skills required for next stages of career.

Knowledge Base: Uses a vector database (ChromaDB) built from real-world salary data (cleaned_salary.csv) to answer questions like:

"Am I underpaid compared to the market?"

"What skills do I need to become a Senior Manager?"

"What is the typical career path for a Data Analyst?"

3. 📊 Interactive Data Analysis

Visualizes market trends using Altair.

Salary distribution histograms.

Experience vs. Salary regression charts.

🛠️ Tech Stack

Frontend: Streamlit

Machine Learning: Scikit-Learn, Pandas, NumPy

LLM Framework: LangChain (LCEL Architecture)

Vector Database: ChromaDB

Embeddings: HuggingFace (all-MiniLM-L6-v2)

Local LLM Server: Ollama

📂 Project Structure

  ├── app.py                      # Main Streamlit application

  ├── vector.py                   # Script to create/update Vector Database

  ├── cleaned_salary.csv          # Raw dataset

  ├── final_enhanced_salary_dataset.csv # Processed dataset for RAG

  ├── salary_pipeline.pkl         # Trained ML Model

  ├── chroma_langchain_db/        # Generated Vector Store (Created by vector.py)
  
  └── requirements.txt  # Python dependencies

  |__ Analysis.ipynb # preprocessing and salary.pkl created from this analysis


🚀 Installation & Setup

Follow these steps to run the application locally.

1. Prerequisite: Install Ollama

  This app requires a local LLM server.

  Download Ollama from ollama.com.

  Install and run Ollama.

  Pull the Llama 3 model via terminal:

        ollama pull llama3 (or any other model name here used qwen2.5)


2. Clone the Repository

        git clone [https://github.com/Jiya112005/Salary_Prediction_Chatbot.git](https://github.com/your-username/salary-prediction-ai.git)
        cd salary-prediction-ai


3. Install Dependencies

       pip install -r requirements.txt


4. Build the Vector Database

Before running the app, you must embed the dataset into the vector store.

      python vector.py


This will create the ./chroma_langchain_db folder.

5. Run the Application

       streamlit run app.py

   
