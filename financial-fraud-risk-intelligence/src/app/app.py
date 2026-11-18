
import os
import re
from pathlib import Path
from typing import TypedDict, Optional

import streamlit as st
import pandas as pd
import joblib
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from langgraph.graph import StateGraph, START, END


# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Fraud Intelligence",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Financial Fraud & Risk Intelligence")
st.caption("ML + RAG + Gemini + LangGraph")

PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_DIR / "artifacts/models/random_forest_v1.joblib"
)

VECTOR_DB_PATH = (
    PROJECT_DIR / "artifacts/vector_db"
)

FRAUD_THRESHOLD = 0.90

FEATURE_ORDER = [
    "step",
    "amount",
    "oldbalanceOrg",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]


# ==========================================
# 2. LOAD SAVED COMPONENTS
# ==========================================

@st.cache_resource
def load_resources():

    model = joblib.load(MODEL_PATH)

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    chroma_client = chromadb.PersistentClient(
        path=str(VECTOR_DB_PATH)
    )

    collection = chroma_client.get_or_create_collection(
        name="fraud_investigation_policies"
    )

    if collection.count() == 0:
        policy_path = PROJECT_DIR / "data/knowledge_base/fraud_investigation_policy.txt"
        policy_text = policy_path.read_text(encoding="utf-8")
        chunk_size, overlap = 800, 100
        step = chunk_size - overlap
        chunks = [policy_text[i:i + chunk_size] for i in range(0, len(policy_text), step)]
        embeddings = embedding_model.encode(chunks).tolist()
        collection.add(
            ids=[f"policy_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
        )

    return model, embedding_model, collection


try:
    fraud_model, embedding_model, fraud_collection = (
        load_resources()
    )
except Exception as error:
    st.error(f"Unable to load ML/RAG resources: {error}")
    st.stop()


# ==========================================
# 3. GEMINI CLIENT
# ==========================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error(
        "GEMINI_API_KEY is missing. "
        "Set it before starting Streamlit."
    )
    st.stop()

gemini_client = genai.Client(api_key=api_key)

GEMINI_MODEL = "gemini-3.1-flash-lite"


# ==========================================
# 4. LANGGRAPH SHARED STATE
# ==========================================

class FraudInvestigationState(TypedDict):
    transaction: dict
    fraud_score: Optional[float]
    fraud_prediction: Optional[int]
    retrieved_policy: Optional[str]
    investigation_report: Optional[str]
    validation_status: Optional[str]
    review_required: Optional[bool]


# ==========================================
# 5. ML SCORING NODE
# ==========================================

def ml_scoring_node(state):

    transaction = state["transaction"]

    transaction_df = pd.DataFrame(
        [transaction]
    )[FEATURE_ORDER]

    fraud_score = float(
        fraud_model.predict_proba(
            transaction_df.to_numpy()
        )[0][1]
    )

    fraud_prediction = int(
        fraud_score >= FRAUD_THRESHOLD
    )

    return {
        "fraud_score": fraud_score,
        "fraud_prediction": fraud_prediction
    }


# ==========================================
# 6. RAG RETRIEVAL NODE
# ==========================================

def rag_retrieval_node(state):

    transaction = state["transaction"]

    query = f"""
    Fraud investigation policies for:
    Transaction amount: {transaction['amount']}
    Sender balance: {transaction['oldbalanceOrg']}
    Fraud risk score: {state['fraud_score']:.2%}
    """

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = fraud_collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas"]
    )

    documents = results["documents"][0]

    retrieved_policy = "\n\n".join(documents)

    return {
        "retrieved_policy": retrieved_policy
    }


# ==========================================
# 7. GEMINI REPORT GENERATION NODE
# ==========================================

def llm_generation_node(state):

    transaction = state["transaction"]

    transaction_type = next(
        (
            name.replace("type_", "")
            for name in FEATURE_ORDER
            if name.startswith("type_")
            and transaction.get(name) == 1
        ),
        "UNKNOWN"
    )

    prompt = f"""
    You are a financial fraud investigation assistant.

    TRANSACTION:
    Amount: {transaction['amount']}
    Sender Previous Balance:
    {transaction['oldbalanceOrg']}
    Transaction Type: {transaction_type}

    ML RESULT:
    Fraud Risk Score: {state['fraud_score']:.2%}
    Flagged for Investigation:
    {bool(state['fraud_prediction'])}

    RETRIEVED POLICY:
    {state['retrieved_policy']}

    Generate these sections:

    1. Transaction Summary
    2. ML Fraud Risk Assessment
    3. Relevant Investigation Policies
    4. Recommended Investigation Actions
    5. Final Recommendation

    Rules:
    - Do not confirm fraud.
    - Do not invent evidence.
    - Do not recommend automatic holds,
      blocking or freezing accounts.
    - Do not invent policy requirements.
    - Treat the ML score as decision support.
    - Recommend human investigation where appropriate.
    """

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty investigation report."
        )

    return {
        "investigation_report": response.text
    }


# ==========================================
# 8. REPORT VALIDATION NODE
# ==========================================

def validation_node(state):

    report = state.get(
        "investigation_report"
    ) or ""

    fraud_score = state.get("fraud_score")
    fraud_prediction = state.get("fraud_prediction")

    required_sections = [
        "Transaction Summary",
        "ML Fraud Risk Assessment",
        "Relevant Investigation Policies",
        "Recommended Investigation Actions",
        "Final Recommendation"
    ]

    missing_sections = [
        section
        for section in required_sections
        if section.lower() not in report.lower()
    ]

    score_valid = (
        fraud_score is not None
        and 0 <= fraud_score <= 1
        and fraud_prediction in (0, 1)
    )

    report_valid = len(report.strip()) >= 100

    unsafe_patterns = [
        r"\bshould\s+be\s+placed\s+on\s+(?:a\s+)?(?:temporary\s+)?hold\b",
        r"\bmust\s+be\s+placed\s+on\s+(?:a\s+)?(?:temporary\s+)?hold\b",
        r"\b(?:automatically|immediately)\s+block\b",
        r"\b(?:automatically|immediately)\s+freeze\b",
        r"\bfraud\s+is\s+confirmed\b",
        r"\bthis\s+is\s+(?:a\s+)?confirmed\s+fraud\b"
    ]

    safety_flags = [
        pattern
        for pattern in unsafe_patterns
        if re.search(pattern, report, re.IGNORECASE)
    ]

    validation_status = (
        "PASSED"
        if (
            not missing_sections
            and score_valid
            and report_valid
            and not safety_flags
        )
        else "FAILED"
    )

    review_required = (
        fraud_prediction == 1
        or validation_status == "FAILED"
    )

    return {
        "validation_status": validation_status,
        "review_required": review_required
    }


# ==========================================
# 9. HUMAN REVIEW NODE
# ==========================================

def human_review_node(state):

    # Flags the case for human investigation.
    # Does not perform actual human approval.

    return {
        "review_required": True
    }


# ==========================================
# 10. CONDITIONAL ROUTER
# ==========================================

def review_router(state):

    if state["review_required"]:
        return "human_review"

    return "end"


# ==========================================
# 11. BUILD LANGGRAPH WORKFLOW
# ==========================================

@st.cache_resource
def build_workflow():

    workflow = StateGraph(
        FraudInvestigationState
    )

    workflow.add_node(
        "ml_scoring", ml_scoring_node
    )

    workflow.add_node(
        "rag_retrieval", rag_retrieval_node
    )

    workflow.add_node(
        "llm_generation", llm_generation_node
    )

    workflow.add_node(
        "validation", validation_node
    )

    workflow.add_node(
        "human_review", human_review_node
    )

    workflow.add_edge(START, "ml_scoring")
    workflow.add_edge("ml_scoring", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "llm_generation")
    workflow.add_edge("llm_generation", "validation")

    workflow.add_conditional_edges(
        "validation",
        review_router,
        {
            "human_review": "human_review",
            "end": END
        }
    )

    workflow.add_edge("human_review", END)

    return workflow.compile()


fraud_graph = build_workflow()


# ==========================================
# 12. STREAMLIT USER INTERFACE
# ==========================================

tab1, tab2 = st.tabs([
    "🔍 Fraud Investigation",
    "💬 AI Policy Chatbot"
])


with tab1:

    st.subheader("Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=90000.0,
            step=100.0
        )

        sender_balance = st.number_input(
            "Sender Previous Balance",
            min_value=0.0,
            value=95000.0,
            step=100.0
        )

    with col2:
        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "TRANSFER",
                "CASH_OUT",
                "PAYMENT",
                "CASH_IN",
                "DEBIT"
            ]
        )

        transaction_step = st.number_input(
            "Transaction Step",
            min_value=1,
            value=100
        )

    if st.button(
        "Analyze Transaction",
        type="primary"
    ):

        transaction = {
            "step": transaction_step,
            "amount": amount,
            "oldbalanceOrg": sender_balance,
            "type_CASH_IN": int(
                transaction_type == "CASH_IN"
            ),
            "type_CASH_OUT": int(
                transaction_type == "CASH_OUT"
            ),
            "type_DEBIT": int(
                transaction_type == "DEBIT"
            ),
            "type_PAYMENT": int(
                transaction_type == "PAYMENT"
            ),
            "type_TRANSFER": int(
                transaction_type == "TRANSFER"
            )
        }

        initial_state = {
            "transaction": transaction,
            "fraud_score": None,
            "fraud_prediction": None,
            "retrieved_policy": None,
            "investigation_report": None,
            "validation_status": None,
            "review_required": None
        }

        try:
            with st.spinner(
                "Running ML + RAG + Gemini + LangGraph..."
            ):
                result = fraud_graph.invoke(
                    initial_state
                )

            st.session_state["last_result"] = result

        except Exception as error:
            st.error(
                f"Investigation failed: {error}"
            )

    if "last_result" in st.session_state:

        result = st.session_state["last_result"]

        st.divider()
        st.subheader("Fraud Risk Assessment")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Fraud Risk Score",
            f"{result['fraud_score']:.2%}"
        )

        metric2.metric(
            "Validation",
            result["validation_status"]
        )

        metric3.metric(
            "Human Review",
            "Required"
            if result["review_required"]
            else "Not Flagged"
        )

        if result["validation_status"] == "FAILED":
            st.warning(
                "Report failed basic validation. "
                "Human verification is required."
            )

        st.subheader("AI Investigation Report")

        st.markdown(
            result["investigation_report"]
        )

        with st.expander(
            "Retrieved Investigation Policies"
        ):
            st.write(
                result["retrieved_policy"]
            )


# ==========================================
# 13. AI POLICY CHATBOT
# ==========================================

with tab2:

    st.subheader(
        "Ask Questions About Fraud Investigation Policies"
    )

    question = st.text_input(
        "Enter your question",
        placeholder=(
            "What should investigators check "
            "for high-risk transfers?"
        )
    )

    if st.button("Ask Gemini") and question.strip():

        try:
            with st.spinner("Searching policy documents..."):

                question_embedding = (
                    embedding_model.encode(
                        question
                    ).tolist()
                )

                search_results = fraud_collection.query(
                    query_embeddings=[
                        question_embedding
                    ],
                    n_results=3,
                    include=["documents"]
                )

                context = "\n\n".join(
                    search_results["documents"][0]
                )

                chatbot_prompt = f"""
                Answer using only the policy context.

                POLICY CONTEXT:
                {context}

                QUESTION:
                {question}

                If the answer is not in the context,
                say that the policy does not provide
                enough information.

                Do not invent policy requirements.
                """

                response = (
                    gemini_client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=chatbot_prompt
                    )
                )

                st.markdown(
                    response.text
                    or "No response was generated."
                )

                with st.expander(
                    "Retrieved Policy Context"
                ):
                    st.write(context)

        except Exception as error:
            st.error(
                f"Chatbot request failed: {error}"
            )


st.divider()
st.caption(
    "Educational fraud investigation prototype. "
    "ML predictions and AI reports do not confirm fraud "
    "or authorize financial actions."
)
