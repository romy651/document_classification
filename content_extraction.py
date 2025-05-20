from io import BytesIO
from sentence_transformers import SentenceTransformer
import pdfplumber
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import fitz
from transformers.models.clip import CLIPProcessor, CLIPModel
from transformers.pipelines import pipeline


question="Year it was issued"
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
text_model = SentenceTransformer('all-MiniLM-L6-v2')

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


reference_text_embeddings = {
    "1040": text_model.encode(" it is a 1040 tax form".lower()),
    "W2": text_model.encode(" it is a W-2 wage and Tax Statement".lower()),
    "1099": text_model.encode(" it is a 1099 - DIV or 1099 - INT tax form".lower()),
    "1098": text_model.encode(" it is a 1098 tax form".lower())
}

reference_image = ["Contains an ID card", "Contains a handwritten note"]


async def extract_images_from_pdf(contents):
    doc = fitz.open(stream=contents, filetype="pdf")
    images = []
    for page in doc:
        for img_index, img_data in enumerate(page.get_images(full=True)):
            xref = img_index
            xref = img_data[0]
            img_data = doc.extract_image(xref)
            img = Image.open(BytesIO(img_data['image']))
            images.append(img)
    return images

def extract_text_from_pdf(contents):
    try:
        with pdfplumber.open(contents) as pdf:
            text = pdf.pages[0].extract_text().lower()
            return text.strip()
    except Exception as e:
        print("Error: ", e)
        return None


def chunk_embed(text):
    if not text:
        return []
    chunks = [chunk.strip() for chunk in text.split("\n") if chunk.strip()]
    return text_model.encode(chunks, normalize_embeddings=True)



def query_embeddings(embeddings):
    best_category = None
    highest_similarity = 0.7
    for category, ref_emb in reference_text_embeddings.items():
        similarities = max(
            cosine_similarity(np.array([embedding]), np.array([ref_emb]))[0][0]
            for embedding in embeddings
        )
        print("The similarities for category ", category, " are: ", similarities)
        if similarities > highest_similarity:
            highest_similarity = similarities
            best_category = category

    return "OTHER" if (best_category == None or best_category == "1098") else best_category

def check_image_category(image):
    inputs = processor(images=image, text=reference_image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).detach().numpy()[0]
    if max(probs) > 0.7:
        return "ID Card" if reference_image[np.argmax(probs)] == "Contains an ID card" else "Handwritten note"
    else:
        return None

async def classify_pdf(pdf_path: BytesIO):
    contents = pdf_path.read()
    ## We check if it contains text
    text = extract_text_from_pdf(pdf_path)
    if text:
        embeddings = chunk_embed(text)
        best_category = query_embeddings(embeddings)
        answer = qa_pipeline(question=question, context=text)
        print("The answer is: ", answer)
        if answer['score'] > 0.0001:
            return best_category, answer['answer']
        else:
            return best_category, "Not Available"
    else:
        ## If No text found, we check the images
        images = await extract_images_from_pdf(contents=contents)
        if images:
            answer = check_image_category(images[0])
            if answer is not None:
                return answer, "Not Available"
    
    return "OTHER", "Not Available"