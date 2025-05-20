from typing import Optional
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from content_extraction import classify_pdf

app = FastAPI()


@app.post("/classify")
async def schedule_classify_task(file: Optional[UploadFile] = File(None)):
    """Endpoint to classify a document into "w2", "1099int", etc"""
    if file is not None:
        content = await file.read()
        print("the file name is: ", file.filename)
        classification, year = await classify_pdf(BytesIO(content))
        print("the classification is: ", classification)
        return {"document_type": classification, "year": year}
    return {"document_type": "NOT IMPLEMENTED", "year": "NOT IMPLEMENTED"}
