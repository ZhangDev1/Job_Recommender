import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.title("Job Recommender App")
query = st.text_area("Enter your query")
top_k = st.number_input("Number of Results", min_value=1, value=5, step=1)

# Need to wrap top_k as an int because number_input will still return a float
# value even though we define the step size as 1, FastAPI expects int
payload = {"query": query, "top_k": int(top_k)}

if st.button("Search"):
    if query:
        try:
            response = requests.post(f"{API_URL}/search", json=payload)
            results = response.json()

            for result in results:
                st.subheader(f'**{result["Job Title"]}**')
                st.write("**Company:**", result["Company"])
                st.write("**Role:**", result["Role"])
                st.write("**Salary Range:**", result["Salary Range"].replace("$", "\\$"))
                st.write("**Skills:**", result["skills"])
                st.caption(f"Rank: {result['rank']}  |  Rerank Score: {result['rerank_score']:.4f}")
                st.divider()
        except requests.exceptions.RequestException:
            # Catches connection errors, timeouts, etc.
            st.error("Could not reach the search service. Please try again later.")
    else:
        st.error("Please enter a valid query!")