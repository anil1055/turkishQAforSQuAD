import streamlit as st

st.set_page_config(page_title="Turkish Question-Answering - via AG", page_icon='📖')
st.header("📖Turkish QA Task for SQuAD - TR")

with st.sidebar:
    hf_key = st.text_input("HuggingFace Access Key", key="hf_key", type="password")

MODEL_QA = {
    "mdeberta": "anilguven/mdeberta_tr_qa_turkish_squad",
    "albert": "anilguven/albert_tr_qa_turkish_squad",  
    "distilbert": "anilguven/distilbert_tr_qa_turkish_squad",
    "bert": "anilguven/bert_tr_qa_turkish_squad",  
}

MODEL_QAS = ["mdeberta","albert","distilbert","bert"]

# Use a pipeline as a high-level helper
from transformers import pipeline
# Create a mapping from formatted model names to their original identifiers
def format_model_name(model_key):
    name_parts = model_key
    formatted_name = ''.join(name_parts)  # Join them into a single string with title case
    return formatted_name

formatted_names_to_identifiers = {
    format_model_name(key): key for key in MODEL_QA.keys()
}

# Debug to ensure names are formatted correctly
#st.write("Formatted Model Names to Identifiers:", formatted_names_to_identifiers

with st.expander("About this app"):
    st.write(f"""
    1-Choose your model for Turkish Question-answering task for SQuAD-tr dataset.\n
    2-Enter your context-question pair.\n
    3-And model predict your response.
    """)

model_name: str = st.selectbox("Model", options=MODEL_QAS)
selected_model = MODEL_QA[model_name]

if not hf_key:
    st.info("Please add your HuggingFace Access Key to continue.")
    st.stop()

access_token = hf_key
pipe = pipeline("question-answering", model=selected_model, token=access_token)

#from transformers import AutoTokenizer, AutoModelForSequenceClassification
#tokenizer = AutoTokenizer.from_pretrained(selected_model)
#pipe = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=selected_model)

# Display the selected model using the formatted name
model_display_name = selected_model  # Already formatted
st.write(f"Model being used: `{model_display_name}`")

with st.form('my_form'):
  context = st.text_area('Enter text:', "Normanlar (Normans: Nourmands; Fransizca: Normands; Latince: Normanni), 10. ve 11. yüzyillarda Fransa'da bir bölge olan Normandiya'ya adini veren insanlardir. Onlar Iskandinav soyundan ('Norman' 'Norseman' geliyor) akincilari ve Danimarka, Izlanda ve Norveç'ten korsanlar, liderleri Rollo altinda, Bati Francia Krali Charles III sadakat yemini etmeyi kabul etti. Nesiller boyunca asimilasyon ve yerli Frenk ve Roman-Galya nüfuslariyla karistirma, torunlari yavas yavas Bati Francia'nin Karolina kökenli kültürleriyle birlesecekti. Normanlarin farkli kültürel ve etnik kimligi baslangiçta 10. yüzyilin ilk yarisinda ortaya çikti ve devam eden yüzyillar boyunca gelismeye devam etti.")
  question = st.text_input("Enter your question for analysis:", "Normandiya hangi ülkede bulunur?")
  submitted = st.form_submit_button('Submit')
  
  if submitted:
    if not hf_key:
        st.info("Please add your HuggingFace Access Key to continue.")
        st.stop()

    else:
        result = pipe(question=question, context=context)
        st.text("Your response: \n " + str(result["answer"]) + "\n" + str(result["score"]*100)[:4] + " with score")
        
        


