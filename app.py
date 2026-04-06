import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from duckduckgo_search import DDGS

# =========================
# CONFIGURATION
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# FUNCTION: WEB SEARCH USING DDGS
# =========================
def web_search_ddgs(query):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(r["body"])  # short description
        return "\n".join(results)
    except Exception as e:
        return f"Web search error: {e}"

# =========================
# FUNCTION: GET LLM RECOMMENDATIONS
# =========================
def get_llm_recommendations(user_query):
    try:
        # Step 1: Get web search data
        search_data = web_search_ddgs(user_query + " best LLM models 2026")

        # Step 2: Send to LLM
        response = client.responses.create(
            model="gpt-4.1",  # safer model name
            input=f"""
            User Query: {user_query}

            Web Search Data:
            {search_data}

            Based on this, recommend 5 best LLMs in table format:

            Name | Description | Parameters | Features | Tool Support
            """
        )

        return response.output_text

    except Exception as e:
        return f"Error: {e}"

# =========================
# FUNCTION: GET LOCAL MODELS (OLLAMA)
# =========================
def get_local_models():
    try:
        res = requests.get("http://localhost:11434/api/tags")
        data = res.json()

        models = []
        for m in data.get("models", []):
            models.append({
                "name": m["name"],
                "size": m.get("size", "Unknown")
            })

        return models

    except:
        return []

# =========================
# MAIN PROGRAM (CLI)
# =========================
def main():
    print("🤖 AgentLens - AI LLM Discovery Assistant (DDGS Version)")
    print("------------------------------------------------------")

    query = input("Enter your AI workflow: ")

    if query.strip() == "":
        print("⚠️ Please enter a valid query!")
        return

    # ===== Cloud Models =====
    print("\n🌐 Recommended LLMs:\n")
    result = get_llm_recommendations(query)
    print(result)

    # ===== Local Models =====
    print("\n💻 Local Models (Ollama):\n")
    local_models = get_local_models()

    if local_models:
        for m in local_models:
            print(f"✔ {m['name']} ({m['size']})")
    else:
        print("❌ No local models found or Ollama not running.")

# =========================
# RUN PROGRAM
# =========================
if __name__ == "__main__":
    main()