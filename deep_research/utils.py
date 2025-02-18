from markdown_pdf import MarkdownPdf, Section

def get_service() -> str:
    global service
    return service

def set_service(new_service: str) -> None:
    global service
    service = new_service

def get_model() -> str:
    global model
    return model

def set_model(new_model: str) -> None:
    global model
    model = new_model

def convert_to_pdf(markdown_file: str, pdf_file: str) -> None:
    """Convert a Markdown file to a PDF file."""
    try:
        # Read the Markdown content from the file
        with open(markdown_file, "r", encoding="utf-8") as md_file:
            markdown_content = md_file.read()

        # Initialize the PDF generator
        pdf = MarkdownPdf()

        # Add a section with the Markdown content
        pdf.add_section(Section(markdown_content))

        # Set PDF metadata (optional)
        pdf.meta["title"] = "Generated PDF"
        pdf.meta["author"] = "Your Name"

        # Save the generated PDF to the specified file
        pdf.save(pdf_file)

        print(f"PDF successfully saved to: {pdf_file}")

    except Exception as e:
        print(f"Error converting to PDF: {e}")
