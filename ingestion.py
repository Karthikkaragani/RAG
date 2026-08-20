from pypdf import PdfReader
from docx import Document


# ============================================================
# PDF
# ============================================================

def extract_pdf(uploaded_file):

    reader = PdfReader(
        uploaded_file
    )

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text and text.strip():

            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "source": uploaded_file.name,
                        "file_type": "pdf",
                        "page_number": page_number
                    }
                }
            )

    return documents


# ============================================================
# TXT
# ============================================================

def extract_txt(uploaded_file):

    text = uploaded_file.read().decode(
        "utf-8"
    )

    if not text.strip():

        return []

    return [
        {
            "text": text,
            "metadata": {
                "source": uploaded_file.name,
                "file_type": "txt"
            }
        }
    ]


# ============================================================
# DOCX
# ============================================================

def extract_docx(uploaded_file):

    document_file = Document(
        uploaded_file
    )

    documents = []

    for paragraph_number, paragraph in enumerate(
        document_file.paragraphs,
        start=1
    ):

        text = paragraph.text.strip()

        if text:

            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "source": uploaded_file.name,
                        "file_type": "docx",
                        "paragraph_number": paragraph_number
                    }
                }
            )

    return documents


# ============================================================
# MARKDOWN
# ============================================================

def extract_markdown(uploaded_file):

    text = uploaded_file.read().decode(
        "utf-8"
    )

    if not text.strip():

        return []

    return [
        {
            "text": text,
            "metadata": {
                "source": uploaded_file.name,
                "file_type": "markdown"
            }
        }
    ]


# ============================================================
# DOCUMENT ROUTER
# ============================================================

def extract_document(uploaded_file):

    file_type = uploaded_file.type

    if file_type == "application/pdf":

        return extract_pdf(
            uploaded_file
        )

    elif file_type == "text/plain":

        return extract_txt(
            uploaded_file
        )

    elif (
        file_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):

        return extract_docx(
            uploaded_file
        )

    elif file_type == "text/markdown":

        return extract_markdown(
            uploaded_file
        )

    return []