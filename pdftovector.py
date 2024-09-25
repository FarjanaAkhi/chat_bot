import os
import PyPDF2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import chromadb 
from chromadb.config import Settings

import nltk

# Download required NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')



import PyPDF2

#  extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                print(f"No text extracted from page {page_num + 1}")  
        return text


pdf_path = r"C:\Users\ACER\Downloads\Message-from-the-Chair.pdf"
extracted_text = extract_text_from_pdf(pdf_path)

# Write the extracted text to a file
with open("extracted_text.txt", "w", encoding="utf-8") as output_file:
    output_file.write(extracted_text)

print("Extracted text has been written to 'extracted_text.txt'.")


def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text.lower())

    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

    return filtered_words

# Assuming `extracted_text` contains the PDF text
processed_text = preprocess_text(extracted_text)
#print("Processed text:", processed_text)


# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert the processed words into vectors
vectors = model.encode(processed_text)

print("Generated Vectors:", vectors)



def store_vectors_in_chroma(vectors, processed_text, collection_name="pdf_vectors", db_path=r"C:\Users\ACER\Desktop\vec-db"):
    # Ensure the directory exists
    os.makedirs(db_path, exist_ok=True)

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=db_path)

    # Create or get the collection
    collection = client.get_or_create_collection(name=collection_name)

    # Metadata for each document (can be expanded with more info)
    metadata = {"id": "doc1", "text": " ".join(processed_text)}  # Assign an ID and the preprocessed text

    # Convert the vectors to a list of lists if they are not already
    vectors_list = vectors.tolist() if hasattr(vectors, 'tolist') else vectors

    # Store vectors and associated metadata
    collection.add(
        documents=[metadata['text']],  # Store the preprocessed text
        embeddings=[vectors_list],  # Store the vector representation as a list
        metadatas=[metadata],  # Store metadata
        ids=[metadata['id']]   # Assign a unique ID
    )

    print(f"Vector stored in ChromaDB with ID: {metadata['id']}")

# Store the generated vectors into ChromaDB
store_vectors_in_chroma(vectors, processed_text)








