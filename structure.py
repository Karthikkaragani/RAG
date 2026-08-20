import re


# ============================================================
# HEADING DETECTION
# ============================================================

def is_heading(line):

    line = line.strip()

    if not line:

        return False

    # Markdown headings

    if re.match(
        r"^#{1,6}\s+",
        line
    ):

        return True

    # Numbered headings
    #
    # 1 Introduction
    # 1.1 Architecture
    # 2.3 Retrieval

    if re.match(
        r"^\d+(\.\d+)*\s+[A-Z]",
        line
    ):

        return True

    # ALL CAPS headings

    if (
        len(line.split()) <= 12
        and line.isupper()
        and len(line) > 2
    ):

        return True

    return False


# ============================================================
# CLEAN HEADING
# ============================================================

def clean_heading(line):

    line = line.strip()

    line = re.sub(
        r"^#{1,6}\s+",
        "",
        line
    )

    return line.strip()


# ============================================================
# STRUCTURE TEXT DOCUMENT
# ============================================================

def structure_text_document(document):

    text = document["text"]

    metadata = document["metadata"]

    lines = text.split("\n")

    structured_units = []

    current_paragraph = []

    def add_paragraph():

        if current_paragraph:

            paragraph_text = " ".join(
                current_paragraph
            ).strip()

            if paragraph_text:

                structured_units.append(
                    {
                        "type": "paragraph",
                        "text": paragraph_text,
                        "metadata": metadata.copy()
                    }
                )

            current_paragraph.clear()

    for line in lines:

        line = line.strip()

        # Empty line
        if not line:

            add_paragraph()

            continue

        # Heading
        if is_heading(line):

            add_paragraph()

            heading = clean_heading(
                line
            )

            structured_units.append(
                {
                    "type": "heading",
                    "text": heading,
                    "metadata": metadata.copy()
                }
            )

            continue

        # Bullet list
        if re.match(
            r"^[-*•]\s+",
            line
        ):

            add_paragraph()

            bullet_text = re.sub(
                r"^[-*•]\s+",
                "",
                line
            )

            structured_units.append(
                {
                    "type": "list_item",
                    "text": bullet_text,
                    "metadata": metadata.copy()
                }
            )

            continue

        # Numbered list
        if re.match(
            r"^\d+[.)]\s+",
            line
        ):

            add_paragraph()

            list_text = re.sub(
                r"^\d+[.)]\s+",
                "",
                line
            )

            structured_units.append(
                {
                    "type": "list_item",
                    "text": list_text,
                    "metadata": metadata.copy()
                }
            )

            continue

        # Normal paragraph
        current_paragraph.append(
            line
        )

    # Add final paragraph

    add_paragraph()

    return structured_units


# ============================================================
# DOCX STRUCTURE
# ============================================================

def structure_docx(document):

    text = document["text"]

    metadata = document["metadata"].copy()

    if is_heading(text):

        unit_type = "heading"

    else:

        unit_type = "paragraph"

    return [
        {
            "type": unit_type,
            "text": (
                clean_heading(text)
                if unit_type == "heading"
                else text
            ),
            "metadata": metadata
        }
    ]


# ============================================================
# STRUCTURE ROUTER
# ============================================================

def detect_structure(documents):

    structured_documents = []

    for document in documents:

        file_type = document[
            "metadata"
        ]["file_type"]

        if file_type == "docx":

            units = structure_docx(
                document
            )

        else:

            units = structure_text_document(
                document
            )

        structured_documents.extend(
            units
        )

    return structured_documents