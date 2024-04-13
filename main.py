import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer

# Download necessary NLTK resources
nltk.download('punkt')

# Load the model and tokenizer for question generation
qg_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-e2e-qg")
qg_model = pipeline("text2text-generation", model="valhalla/t5-base-e2e-qg")

# Load the tokenizer and model for question answering
qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

# Load Sentence-BERT model
sentence_bert_model = SentenceTransformer('average_word_embeddings_glove.6B.300d')

# Define the function to generate flashcards
def generate_flashcards(text, num_flashcards):      
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Generate questions
    questions = []
    for sentence in sentences:
        question = qg_model(sentence)
        if question:
            questions.append(question[0]['generated_text'])

    # Rank questions based on quality/relevance using embeddings
    if questions:
        question_embeddings = sentence_bert_model.encode(questions)
        ranking_scores = question_embeddings.dot(question_embeddings.T).mean(axis=1)  # Compute cosine similarity scores
        selected_indices = ranking_scores.argsort()[-num_flashcards:][::-1]
        selected_questions = [questions[i] for i in selected_indices]
    else:
        return []

    # Generate answers for selected questions
    answers = []
    for question in selected_questions:
        inputs = qa_tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
        outputs = qa_model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)  # Start of answer
        answer_end = torch.argmax(outputs.end_logits) + 1  # End of answer

        # Extract the answer
        answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))
        answer = answer.replace("Ġ", " ")  # Fix subword tokenization artifacts
        answers.append(answer.strip())

    # Prepare flashcards
    flashcards = [{'question': q, 'answer': a} for q, a in zip(selected_questions, answers)]
    return flashcards
