import joblib
import streamlit as st
import pandas as pd
import os
import altair as alt

# --- LangChain Imports ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# ---------- CONFIGURATION ----------
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
VECTOR_DB_PATH = "./chroma_langchain_db"
PLACEHOLDER_IMAGE_URL = "web_img.png"  

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Salary Prediction AI",
    page_icon="💼",
    layout="wide"
)
# ---------- GLOBAL STYLING ----------
st.markdown("""
<style>
/* Global & Sidebar */
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #fbeaff 0%, #ffffff 100%); color: #2c003e; font-family: 'Poppins', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(to bottom right, #5f2c82, #9c27b0, #ff9a9e); color: white; }
[data-testid="stSidebar"] * { color: black !important; font-weight: 500; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] label { color: white !important; font-weight: 700 !important; }

/* Form & Inputs */
.section { background: #ffffff; border-radius: 20px; padding: 28px 30px; box-shadow: 0px 6px 18px rgba(95, 44, 130, 0.15); margin-bottom: 25px; }
label { font-weight: 700 !important; color: #4a0072 !important; }
div.stButton > button { background: linear-gradient(to right, #9c27b0, #ff9a9e); color: white; border: none; border-radius: 10px; padding: 12px 35px; font-weight: 600; }
div.stButton > button:hover { transform: scale(1.05); }

/* Custom Chatbot Button Styling (For redirection) */
.redirect-button button { 
    background: linear-gradient(to right, #5f2c82, #9c27b0) !important;
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 12px 35px;
    font-weight: 600;
    margin-top: 15px;
}
/* Metric & Insights */
.metric-container h2 { 
    color: white !important; 
    background: linear-gradient(45deg, #4a0072, #9c27b0); 
    padding: 20px; 
    border-radius: 15px; 
    text-align: center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
.insight-box { background-color: #f3e5f5; border-left: 5px solid #9c27b0; padding: 15px; border-radius: 10px; margin-top: 15px; }

/* Chat Bubbles (For better response formatting) */
[data-testid="stChatMessage"] {
    background-color: #fff;
    padding: 10px 15px;
    border-radius: 15px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
[data-testid="stChatMessage"]:has(div.stMarkdown > p:first-child:contains("Initial Prediction Analysis")) {
    background-color: #e6e6fa; /* Light purple for initial analysis */
    border-left: 5px solid #4a0072;
}
</style>
""", unsafe_allow_html=True)

# ---------- DATA LOADING ----------
@st.cache_resource
def load_data():
    try:
        model = joblib.load('salary_pipeline.pkl')
        data = pd.read_csv('cleaned_salary.csv')
        return model, data
    except Exception as e:
        st.error(f"Failed to load ML files: {e}")
        return None, None

model_pred, data = load_data()

# ---------- SIMPLE RAG WRAPPER ----------
class SimpleRAG:
    """
    Minimal retriever -> LLM wrapper exposing an `invoke` method.
    """
    def __init__(self, retriever, llm, system_prompt_template: str, max_docs=1):
        self.retriever = retriever
        self.llm = llm
        self.system_prompt_template = system_prompt_template
        self.max_docs = max_docs
        
    def _format_context(self, docs):
        parts = []
        for d in docs[:self.max_docs]:
            txt = getattr(d,"page_content",None)
            if txt is None and isinstance(d,dict):
                txt = d.get("page_content") or d.get("content") or str(d)
            parts.append(str(txt)[:500])
        return "\n\n".join(parts)

    
    def invoke(self, inputs):
        query = inputs.get("input") if isinstance(inputs, dict) else str(inputs)

        # 1) Retrieve top docs
        docs = getattr(self.retriever, "get_relevant_documents", lambda q: [])(query)

        # 2) Build prompt
        context_text = self._format_context(docs)
        system_text = self.system_prompt_template.format(context=context_text)
        messages = [SystemMessage(content=system_text), HumanMessage(content=query)]

        # 3) Call LLM
        try:
            resp = self.llm.invoke(messages)
        except Exception as e:
            raise RuntimeError(f"LLM invocation failed: {e}")

        # 4) Extract response text
        if hasattr(resp, "content"): return resp.content
        if isinstance(resp, dict):
            for k in ("content", "response", "text", "output"):
                if k in resp: return resp[k]
        return str(resp)
    
# ---------- RAG CHAIN SETUP (The Integration Logic) ----------
@st.cache_resource
def get_rag_chain():
    """Constructs and returns a SimpleRAG instance (or None on error)."""
    try:
        # 1. Setup LLM
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

        # 2. Setup Embeddings (Must match what you used in vector.py)
        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

        # 3. Load Vector DB
        if not os.path.exists(VECTOR_DB_PATH):
            st.error(f"⚠️ Database not found at {VECTOR_DB_PATH}. Please run 'python vector.py' first.")
            return None

        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings,
            collection_name="salary_explanations"  # Important: Must match vector.py
        )

        # 4. Create Retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Fetch top 2 results

        # 5. Create a system prompt template that expects `{context}` and `{input}` via HumanMessage
        system_prompt = (
            "You are an expert HR Analyst. Use the retrieved context to answer "
            "the user's question about salary, skills, and career steps. "
            "If the answer is not in the context, say you don't know."
            "\n\n"
            "{context}"
        )

        # 6. Return a simple wrapper providing `.invoke({...})`
        return SimpleRAG(retriever=retriever, llm=llm, system_prompt_template=system_prompt, max_docs=2)

    except Exception as e:
        st.error(f"RAG Setup Error: {e}")
        return None

rag_chain = get_rag_chain()

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI assistant. Get a prediction first, then ask me anything about your career."}]
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'new_prediction_made' not in st.session_state:
    st.session_state.new_prediction_made = False

# ---------- UI FUNCTIONS ----------
def salary_prediction_ui():
    st.markdown("<h1>💼 Salary Prediction</h1>", unsafe_allow_html=True)

    if model_pred is None:
        return

    left, right = st.columns([2, 1])

    with left:
        with st.form("salary_prediction_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("👤 Personal Details")
                age = st.number_input("Age", 18, 80, 25)
                gender = st.selectbox("Gender", sorted(data['Gender'].unique()))
                education = st.selectbox("Education Level", sorted(data["Education Level"].unique()))
            with col_b:
                st.subheader("💼 Job Details")
                job_options = sorted(data["Job Title"].unique().tolist()) + ["Other"]
                job_title = st.selectbox("Job Title", job_options)
                experience_year = st.number_input("Years of Experience", 0.0, 50.0, 2.0)

            submitted = st.form_submit_button("🔮 Predict Salary", type="primary")

    with right:
        st.image(PLACEHOLDER_IMAGE_URL, width=300, caption="")
    if submitted:
        # Prediction Logic
        final_job = job_title if job_title != "Other" else "Software Engineer"
        input_df = pd.DataFrame({
            'Age': [age], 'Gender': [gender], 'Education Level': [education],
            'Job Title': [final_job], 'Years of Experience': [experience_year]
        })

        pred_usd = model_pred.predict(input_df)[0]
        formatted_salary = f"${pred_usd:,.2f}"

        # Save Context
        st.session_state.last_prediction = {
            "Role": final_job,
            "Experience": f"{experience_year} years",
            "Predicted": formatted_salary
        }

        # Display Result
        left.success(f"### 💰 Estimated Salary: {formatted_salary}")

        spinner_placeholder = st.empty()
        # Trigger AI Analysis
        if rag_chain:
            with spinner_placeholder.container():
                with st.spinner("AI Agent is analyzing market data..."):
                    query = f"Analyze this profile: {final_job} with {experience_year} years experience. Predicted salary is {formatted_salary}."
                    try:
                        # rag_chain.invoke accepts a dict like {"input": query}
                        analysis = rag_chain.invoke({"input": query})
                        # analysis = response

                        left.info("### 💡 AI Career Analysis")
                        left.write(analysis)

                        # Save to chat history
                        st.session_state.messages.append({"role": "assistant", "content": f"**Analysis:** {analysis}"})
                        # st.session_state.new_prediction_made = True
                    except Exception as e:
                        st.error(f"RAG Invocation Error: {e}")
        else:
            st.error("AI Agent is not connected. Check logs.")

def chatbot_ui():
    st.markdown("<h1>🤖 AI Career Coach Chatbot</h1>", unsafe_allow_html=True)

    # Display Chat History
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about skills, negotiation, or market trends..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Generate Response
        if rag_chain:
            with st.spinner("Thinking..."):
                # Add context from last prediction if available
                context_str = ""
                if st.session_state.last_prediction:
                    p = st.session_state.last_prediction
                    context_str = f"Context: User is a {p['Role']} earning {p['Predicted']}. "

                try:
                    answer = rag_chain.invoke({"input": f"{context_str}\nUser Question: {prompt}"})
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.chat_message("assistant").write(answer)
                except Exception as e:
                    st.error(f"Ollama Error: {e}")
        else:
            st.error("AI Agent is not connected. Check logs.")

def data_analysis_ui():
    """NEW PAGE: Displays charts and analysis of the training data."""
    st.markdown("<h1>📊 ML Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown("#### **Project Component: ML Engineering (Transparency & Insights)**")
    st.info("These charts visualize the distribution and relationships in the training data used to build the salary prediction model.")

    if data is None:
        st.error("Cannot load analysis data. Please check 'cleaned_salary.csv'.")
        return

    # 1. Salary Distribution (Histogram)
    st.markdown("---")
    st.subheader("1. Distribution of Salaries")
    chart1 = alt.Chart(data).mark_bar(color='#9c27b0', opacity=0.8).encode(
        x=alt.X('Salary', bin=alt.Bin(maxbins=30), title='Annual Salary (USD)'), # Adjusted bins for better distribution view
        y=alt.Y('count()', title='Number of Employees'),
        tooltip=['Salary', 'count()']
    ).properties(
        title='Salary Distribution Across the Dataset'
    ).interactive()
    st.altair_chart(chart1, use_container_width=True)

    # 2. Salary vs. Years of Experience (Scatter Plot/Regression Line)
    st.markdown("---")
    st.subheader("2. Salary vs. Years of Experience")
    
    # Scatter plot
    scatter = alt.Chart(data).mark_circle(size=60, color='#ff9a9e').encode(
        x=alt.X('Years of Experience', title='Years of Experience'),
        y=alt.Y('Salary', title='Salary (USD)'),
        tooltip=['Years of Experience', 'Salary', 'Job Title']
    )
    
    # Regression line
    line = scatter.transform_regression('Years of Experience', 'Salary').mark_line(color='#5f2c82', strokeWidth=3)

    chart2 = (scatter + line).properties(
        title='Experience vs. Salary Trend'
    ).interactive()
    st.altair_chart(chart2, use_container_width=True)

    # 3. Average Salary by Education Level (Bar Chart)
    st.markdown("---")
    st.subheader("3. Average Salary by Education Level")
    
    # Calculate means for accurate sorting and display
    avg_salary_data = data.groupby('Education Level')['Salary'].mean().reset_index(name='Average Salary')
    
    chart3 = alt.Chart(avg_salary_data).mark_bar(color='#5f2c82').encode(
        x=alt.X('Average Salary', title='Average Salary (USD)'),
        y=alt.Y('Education Level', title='Education Level', sort='-x'), # Sort by salary descending
        tooltip=[alt.Tooltip('Education Level'), alt.Tooltip('Average Salary', format='$,.0f')]
    ).properties(
        title='Average Salary by Education Level'
    ).interactive()
    st.altair_chart(chart3, use_container_width=True)
    
# ---------- MAIN ----------
def main():
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Salary Prediction"

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Salary Prediction", "AI Chatbot", "Data Analysis"])

    if page == "Salary Prediction":
        salary_prediction_ui()
    elif page == "AI Chatbot":
        chatbot_ui()
    elif page == "Data Analysis":
        data_analysis_ui()

if __name__ == "__main__":
    main()
