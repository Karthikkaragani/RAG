import re


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    # Normalize line endings

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Fix words broken by PDF line wrapping
    #
    # atten-
    # tion
    #
    # becomes:
    #
    # attention

    text = re.sub(
        r"(\w)-\n(\w)",
        r"\1\2",
        text
    )

    # Replace tabs

    text = text.replace(
        "\t",
        " "
    )

    # Remove trailing spaces

    text = "\n".join(
        line.rstrip()
        for line in text.split("\n")
    )

    # Reduce multiple spaces

    text = re.sub(
        r"[ ]{2,}",
        " ",
        text
    )

    # Reduce excessive blank lines

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# CLEAN ALL DOCUMENTS
# ============================================================

def clean_documents(documents):

    cleaned_documents = []

    for document in documents:

        cleaned_text = clean_text(
            document["text"]
        )

        cleaned_documents.append(
            {
                "text": cleaned_text,
                "metadata": document["metadata"].copy()
            }
        )

    return cleaned_documents