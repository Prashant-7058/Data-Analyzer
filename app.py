import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import plotly.express as px
import plotly.graph_objects as go



from dotenv import load_dotenv
import io
import chardet

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if api_key is None:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file.")

os.environ["GOOGLE_API_KEY"] = api_key




# # Set Gemini API key
# os.environ["GOOGLE_API_KEY"] = ""  # Put your Gemini API key here

# ✅ Streamlit UI
st.set_page_config(page_title="Gemini Data Analyzer", layout="wide")
st.title("📊Data Analyzer")
# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.write("Upload a CSV file and ask a question. Use the dropdown to specify query type!")

# ✅ File uploader
# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df=pd.read_csv(uploaded_file)
        st.success("✅ File loaded successfully!")
        st.dataframe(df.head())

    except UnicodeDecodeError:
        raw_bytes = uploaded_file.getvalue()  # Keep original bytes

        # ✅ Cached decoding function using chardet, UTF-8, ISO fallback
        @st.cache_data(show_spinner=False)
        def decode_bytes(bytes_data):
            try:
                enc = chardet.detect(bytes_data)['encoding']
                return bytes_data.decode(enc), enc
            except:
                try:
                    return bytes_data.decode('utf-8'), 'utf-8'
                except:
                    return bytes_data.decode('ISO-8859-1'), 'ISO-8859-1'

        try:
            decoded_text, detected_encoding = decode_bytes(raw_bytes)
            df = pd.read_csv(io.StringIO(decoded_text))
            st.success(f"✅ File uploaded successfully with encoding `{detected_encoding}`")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")
            st.stop()

    # Create Gemini agent
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    agent = create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True)

    # 🚦 Radio selection for query type
    query_type = st.selectbox(
        "Choose your query type:",
        ["Visualization", "Slice DataFrame", "Other"]
    )
    if query_type == "Visualization":
        viz_library = st.selectbox("📚 Choose visualization library:", ["matplotlib", "seaborn", "plotly"])

    # 💬 User query
    user_query = st.text_input("💬 Ask a question :")



    if st.button("Ask Analyzer") and user_query:
        with st.spinner("🤖 wait ,I am thinking..."):
            try:
                # ✅ Auto-prompting based on selection
                if query_type == "Visualization":
                    prompt = f"""You are using a pandas DataFrame called df that has already been loaded.
                                Return only clean Python {viz_library} code to draw {user_query} using this df.
                                Do not create any new DataFrame or use sample data.
                                """

                elif query_type == "Slice DataFrame":
                    prompt = f"Return only clean Python pandas code that slices the DataFrame and assigns the result to a variable named result_df. data frame having Query: {user_query}"
                else:
                    prompt = user_query

                # Gemini agent response
                response = agent.run(prompt)
                print("the code...............")
                print(response)

                # If response contains code block
                if "```python" in response:
                    code = response.split("```python")[1].split("```")[0]
                    # Remove fig.show() or plt.show() to prevent pop-up window
                    code = re.sub(r'fig\.show\s*\(.*?\)', '', code)

                    code = re.sub(r'plt\.show\s*\(\s*\)', '', code)
                    print(code)


                    # ✅ Moderate figure size for Streamlit layout
                    fig_width, fig_height = 6, 4
                    code = f"plt.figure(figsize=({fig_width},{fig_height}), dpi=100)\n" + code

                    if "plt.tight_layout" not in code:
                        code += "\nplt.tight_layout()"

                    import matplotlib
                    scale = 1.0  # Use fixed scale for Streamlit to ensure consistent visuals
                    matplotlib.rcParams.update({
                        'axes.titlesize': 14,
                        'axes.labelsize': 12,
                        'xtick.labelsize': 10,
                        'ytick.labelsize': 10,
                        'legend.fontsize': 11
                    })

                    # ✅ Execution
                    exec_globals = {"df": df, "plt": plt, "pd": pd, "px": px}
                    exec_locals = {}
                    exec(code, exec_globals, exec_locals)




                    from selenium import webdriver
                    from selenium.webdriver.chrome.options import Options
                    from PIL import Image
                    import tempfile
                    import time
                    if query_type == "Visualization":

                        if viz_library== "plotly" :
                            if "fig" in exec_locals:
                                fig = exec_locals["fig"]
                                    # ✅ Make colorful
                                fig.update_layout(
                                    template="plotly",  # colorful
                                    paper_bgcolor="white",
                                    plot_bgcolor="white",
                                    font=dict(color="black")
                                )

                                
                                try:
                                    # Save Plotly figure as HTML first
                                    html_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                                    fig.write_html(html_path.name)

                                    # Setup headless browser
                                    options = Options()
                                    options.add_argument('--headless')
                                    options.add_argument('--no-sandbox')
                                    options.add_argument('--disable-dev-shm-usage')
                                    driver = webdriver.Chrome(options=options)
                                    driver.set_window_size(1000, 800)

                                    # Load the HTML file in the headless browser
                                    driver.get("file://" + html_path.name)
                                    time.sleep(2)  # Let it render completely

                                    # Take screenshot
                                    screenshot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                                    driver.save_screenshot(screenshot_path.name)
                                    driver.quit()

                                    # Convert to JPEG (optional)
                                    img = Image.open(screenshot_path.name)
                                    jpg_path = screenshot_path.name.replace(".png", ".jpg")
                                    img.convert("RGB").save(jpg_path, format="JPEG")

                                    # Display in Streamlit
                                    st.image(jpg_path, caption=f"📊 Plot for: {user_query}", use_container_width=True)

                                    # Store path in chat history
                                    st.session_state.chat_history.append({
                                        "type": "visualization",
                                        "question": user_query,
                                        "plot_path": jpg_path
                                    })

                                    # Clean up
                                    del fig
                                    exec_locals.clear()

                                except Exception as e:
                                    st.error(f"❌ Could not render plot via Selenium: {e}")




                        # ✅ Show plot
                        elif viz_library in ["matplotlib", "seaborn"]:

                        # elif "plt" in code in code:
                            fig = plt.gcf()
                            
                            # Save the figure to a temp file
                            import tempfile
                            temp_plot_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                            fig.savefig(temp_plot_file.name)

                            # Display in Streamlit
                            col1, col2, col3 = st.columns([1, 2, 1])  # Middle column is wider
                            with col2:
                                st.image(temp_plot_file.name, caption=f"📊 Plot for: {user_query}")

                            # st.image(temp_plot_file.name, caption=f"📊 Plot for: {user_query}")
                            # st.pyplot(fig, use_container_width=True)
                            plt.close()  

                            # Store in chat history with path
                            st.session_state.chat_history.append({
                                "type": "visualization",
                                "question": user_query,
                                "plot_path": temp_plot_file.name
                            })





                    # ✅ Show filtered DataFrame if created
                    if "result_df" in exec_locals:
                        result_df = exec_locals["result_df"]
                        if isinstance(result_df, pd.DataFrame):
                            st.success("✅ Filtered DataFrame:")
                            st.dataframe(result_df)

                            generated_code = ""
                            for line in code.split("\n"):
                                if line.strip().startswith("result_df"):
                                    generated_code = line.strip()
                                    break

                            # Store Q&A in chat history
                            st.session_state.chat_history.append({
                                "type": "slice",  # or "visualization"/"slice" accordingly
                                "question": user_query,
                                "answer": result_df,  # or "plot_path", or "data",
                                "code": generated_code
                            })




                else:
                    # If Gemini gave text
                    st.success("✅ Gemini's Response:")
                    st.write(response)

                    # Store Q&A in chat history
                    st.session_state.chat_history.append({
                        "type": "text",  # or "visualization"/"slice" accordingly
                        "question": user_query,
                        "answer": response  # or "plot_path", or "data"
                    })




            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # 📜 Show chat history (Q&A log)



else:
    st.info("👈 Please upload a CSV file to begin.")

st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")


# st.subheader("💬 Chat History")
with st.expander("💬 Chat "):
    if st.session_state.chat_history:
        for i, item in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"### Q{i}: {item['question']}")

            if item["type"] == "visualization":
                if os.path.exists(item["plot_path"]):
                    st.image(item["plot_path"], caption=f"📊 Plot for Q{i}")
                    with open(item["plot_path"], "rb") as f:
                        st.download_button(
                            label="📥 Download Plot Image",
                            data=f,
                            file_name=f"plot_q{i}.png",
                            mime="image/png"
                        )

            elif item["type"] == "slice":
                st.success("🧩 Sliced DataFrame:")
                st.dataframe(item["answer"], use_container_width=True)

            elif item["type"] == "text":
                st.success("💡 Gemini's Response:")
                st.write(item["answer"])


from fpdf import FPDF
import tempfile
import matplotlib.pyplot as plt

# Helper to sanitize text for Latin-1 encoding
def safe_text(text):
    return ''.join(c if ord(c) < 256 else '' for c in text)

# Save chat history (text, images, sliced DataFrame) to PDF
def generate_chat_pdf(chat_history):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for i, item in enumerate(chat_history, 1):
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, safe_text(f"Q{i}: {item['question']}"), ln=True)
        pdf.ln(2)

        if item["type"] == "text":
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 8, safe_text(f"Answer:\n{item['answer']}"))
            pdf.ln(4)


        elif item["type"] == "slice":
            pdf.set_font("Arial", style='B', size=11)
            pdf.multi_cell(0, 8, "Generated code:")
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 8, safe_text(item["code"]))
            pdf.ln(4)


        elif item["type"] == "visualization":
            if "plot_path" in item and os.path.exists(item["plot_path"]):
                pdf.set_font("Arial", style='B', size=11)
                pdf.cell(0, 10, safe_text("Generated Plot:"), ln=True)
                pdf.image(item["plot_path"], w=pdf.w - 40)  # auto-scale
                pdf.ln(4)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        return tmp_file.name




# ✅ Download Confirmation Buttons
if st.button("⬇️ Download chat as PDF"):
    st.warning("⚠️ The sliced DataFrame is not included in the PDF due to formatting limitations.")
    with st.spinner("Generating PDF..."):
        pdf_path = generate_chat_pdf(st.session_state.chat_history)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Click to Download PDF",
                data=f,
                file_name="chat_history.pdf",
                mime="application/pdf"
            )


