"""
Structured extraction from notice images using Gemini.

This is your existing notebook logic (cells 1 and 5), moved into a module
and pointed at config.GOOGLE_API_KEY instead of a hardcoded key. The
Pydantic schema and extraction prompt are unchanged.
"""
from typing import List, Optional

from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from config import GOOGLE_API_KEY

client = genai.Client(api_key=GOOGLE_API_KEY)
import fitz  # pip install pymupdf
from pathlib import Path
from typing import List, Dict

def extract_text_from_multipage_pdf(pdf_path: str | Path) -> List[Dict]:
    """
    Extracts text from a multi-page PDF, preserving page-level metadata.
    Essential for tracing answers back to specific pages in long notices.
    """
    pdf_path = Path(pdf_path)
    document_data = []
    
    # Open the multi-page PDF
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            
            # Only append pages that contain extractable text
            if text:
                document_data.append({
                    "text": text,
                    "metadata": {
                        "source_document": pdf_path.name,
                        "page_number": page_num + 1,
                        "total_pages": len(doc)
                    }
                })
    return document_data


class SignatoryEntity(BaseModel):
    name_or_designation: str = Field(description="Official designation like 'प्राचार्य'")
    organization: Optional[str] = Field(description="Institution name if mentioned")
    date_signed: Optional[str] = Field(description="Date near signature if visible")


class AcademicNoticeSchema(BaseModel):
    issuing_authority: Optional[str] = None
    reference_number: Optional[str] = None
    date_issued: Optional[str] = None
    subject_line: Optional[str] = None
    target_audience: Optional[List[str]] = None
    main_body_content: Optional[str] = None
    signatories: Optional[List[SignatoryEntity]] = None
    distribution_list: Optional[List[str]] = None
    document_type: Optional[str] = None
    extra_fields: Optional[str] = None


_EXTRACTION_PROMPT = """You are an intelligent document
understanding system specialized in academic and
administrative documents.
Your task is to extract structured information from the given
document image.
---
### STEP 1: Identify Document Type
Classify the document into one of the following:
- notice
- office_order
- exam_schedule
- email
- tabular_data
- unknown

---
### STEP 2: CRITICAL PRIORITY FIELDS (High Importance)

The following fields are essential for retrieval systems and
must be extracted with maximum accuracy if present:
- reference_number
- date_issued
- issuing_authority
Guidelines:
- These are usually found in headers, top corners, or near
official seals
- Labels may include: "Ref No", "पत्रांक", "ज्ञापांक", "Date"
- Even if slightly unclear or noisy, extract the best
possible readable value
- Do NOT ignore these fields if visible
---

### STEP 3: MAIN CONTENT (Very Important)

- main_body_content MUST contain all readable textual content
from the document
- Preserve logical reading order
- If structure is unclear, still extract all readable text
into main_body_content
- Do NOT return null unless the document is completely empty

---
### STEP 4: General Extraction Rules

- Extract ONLY fields that are clearly present
- If a field is missing → return null
- Do NOT hallucinate or guess
- Do NOT force values

---

### STEP 5: Handle Unknown or New Structures

- If document structure does not match known formats:
  - Extract full readable content into main_body_content
  - Store additional structured or semi-structured data inside `extra_fields`

- If new structured elements appear (tables, lists, sections):
  - Capture them inside `extra_fields` as a JSON dictionary

---

### STEP 6: Table Handling

- Preserve table structure exactly
- Use Markdown or JSON format
- Do NOT break rows or columns
- Maintain header-to-row relationships

---

### STEP 7: Language & Formatting

- Preserve Hindi and English exactly as written
- Maintain logical reading order
- Remove noise such as random symbols or scan artifacts
- Do NOT summarize or translate

---

### STEP 8: Signatories Handling

- Signatories are NOT critical
- Extract ONLY if clearly visible
- Prefer structured format (designation, organization, date if visible)
- If partially visible → extract at least designation
- If not visible → return null
- Do NOT return empty objects

---

### STEP 9: extra_fields Handling

- Store additional structured data as a JSON stirng
- Include:
  - tables
  - unknown formats
  - additional metadata not covered in schema

---

### FINAL RULES

- Do NOT return empty objects
- Ensure all fields follow schema strictly
- Return valid JSON only"""


def extract_structured_metadata(image_path: str) -> AcademicNoticeSchema:
    """
    Extracts structured, strongly-typed JSON data directly from a scanned notice image
    utilizing the Gemini API's Structured Output capabilities paired with Pydantic schemas.
    """
    try:
        document_image = Image.open(image_path)
    except FileNotFoundError:
        raise ValueError(f"Document image not found at specified path: {image_path}")

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[document_image, _EXTRACTION_PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AcademicNoticeSchema,
            temperature=0.0,
        ),
    )

    parsed = response.parsed
    if not parsed.main_body_content:
        parsed.main_body_content = ""

    return parsed
