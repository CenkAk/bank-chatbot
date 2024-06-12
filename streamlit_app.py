import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import pickle

# Load the model
@st.cache_resource
def load_model():
    try:
        return pickle.load(open("model.pkl", 'rb'))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the dataset
@st.cache_resource
def load_data():
    try:
        return load_dataset("PolyAI/banking77")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def chatbot(input_text, data, model, threshold=0.5):
    if not model or not data:
        return []

    try:
        # Create embedding for user input
        input_embedding = model.encode(input_text)
        # Create embeddings for all texts
        all_embeddings = model.encode(data["test"]["text"])
        # Calculate similarity scores
        similarity_scores = cosine_similarity([input_embedding], all_embeddings)[0]
        # Find indices with similarity scores above the threshold
        similar_indices = [i for i, score in enumerate(similarity_scores) if score >= threshold]
        # List of similar texts and their scores
        similar_texts = [(data["test"]["text"][i], similarity_scores[i]) for i in similar_indices]
        # Sort by similarity score in descending order
        similar_texts.sort(key=lambda x: x[1], reverse=True)
        # Select top three responses
        top_three_responses = similar_texts[:3]
        # Calculate total similarity score for the top responses
        total_similarity_score = sum(score for _, score in top_three_responses)
        # Calculate percentage of similarity score for each response
        response_percentages = [(text, score / total_similarity_score * 100) for text, score in top_three_responses]
        return response_percentages
    except Exception as e:
        st.error(f"Error in processing: {e}")
        return []

def main():
    st.set_page_config(
        page_title="Bank Chatbot",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title('Contact')
    st.sidebar.write('')
    st.sidebar.markdown("[Send E-Mail](mailto:@)")
    st.sidebar.title('About Me')
    st.sidebar.write('Orhan Cenk Akcadoğan bitirme projesi.')
    st.sidebar.write('20703030')

    st.title("Bank Chatbot")

    model = load_model()
    data = load_data()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "conversation_active" not in st.session_state:
        st.session_state.conversation_active = True

    if "selected_response" not in st.session_state:
        st.session_state.selected_response = None

    if st.session_state.conversation_active:
        user_input = st.text_input("Soru:")

        if user_input.lower() == 'exit':
            st.write("Çıkış yapılıyor...")
        elif user_input and st.session_state.selected_response is None:
            response_percentages = chatbot(user_input, data, model)
            if response_percentages:
                response_texts = [f"{i+1}. {text}: %{percentage:.2f}" for i, (text, percentage) in enumerate(response_percentages)]
                st.session_state.chat_history.append({"user": user_input, "bot": response_texts})
                selected_response = st.radio("Select a response:", response_texts, key="response_radio")
                
                if selected_response:
                    st.session_state.selected_response = selected_response

                    if "bankaya gitmelisin" in selected_response.lower():
                        st.write("Bankaya Gitmen Gerek.")
                    elif "telefon numarasına yönlendirsin" in selected_response.lower():
                        st.write("Lutfen bu numara ile musteri hizmetlerine ulasiniz: 123-456-7890")

        if st.session_state.selected_response:
            next_action = st.radio("Ne Yapmak Istiyorsun?", ("Yazismaya devam et", "Yazismayi kapat"), key="next_action_radio")

            if next_action == "Yazismaya devam et":
                st.session_state.conversation_active = True
                st.session_state.selected_response = None
            elif next_action == "Yazismayi kapat":
                st.write("İyi Günler Dilerim!")
                st.session_state.chat_history = []  # Clear chat history
                st.session_state.conversation_active = False
                st.session_state.selected_response = None  # Reset selected response to hide options

    if st.session_state.conversation_active and st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.write(f"**Kullanıcı:** {chat['user']}")
            for response in chat['bot']:
                st.write(f"**Bot:** {response}")

if __name__ == "__main__":
    main()
