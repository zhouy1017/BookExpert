# Document Extractors (PDF, DOCX, DOC)
import os
import tempfile
import logging
import fitz  # PyMuPDF
import docx

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        pass

    def extract_text(self, file_path: str) -> str:
        """
        Extracts text from a document based on its file extension.
        Handles .pdf, .docx, and .doc files.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return self._extract_pdf(file_path)
        elif ext == '.docx':
            return self._extract_docx(file_path)
        elif ext == '.doc':
            return self._extract_doc(file_path)
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _extract_pdf(self, file_path: str) -> str:
        """Extracts text from a PDF file using PyMuPDF (fitz)."""
        logger.info(f"Extracting PDF: {file_path}")
        text = []
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    text.append(page.get_text())
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise
        return "\n\n".join(text)

    def _extract_docx(self, file_path: str) -> str:
        """Extracts text from a DOCX file using python-docx."""
        logger.info(f"Extracting DOCX: {file_path}")
        text = []
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text.append(para.text)
        except Exception as e:
            logger.error(f"Error extracting DOCX: {e}")
            raise
        return "\n\n".join(text)

    def _extract_doc(self, file_path: str) -> str:
        """
        Extracts text from a DOC file by first converting it to DOCX
        using pywin32 (Windows COM) then extracting it.
        """
        import win32com.client as win32
        
        logger.info(f"Converting DOC to DOCX: {file_path}")
        abs_in_path = os.path.abspath(file_path)
        
        # Output docx temp path
        temp_dir = tempfile.gettempdir()
        base_name = os.path.basename(file_path)
        out_name = os.path.splitext(base_name)[0] + ".docx"
        abs_out_path = os.path.join(temp_dir, out_name)

        word = None
        try:
            # 16 represents wdFormatDocumentDefault (docx)
            word = win32.Dispatch('Word.Application')
            word.Visible = False
            
            doc = word.Documents.Open(abs_in_path)
            doc.SaveAs(abs_out_path, FileFormat=16)
            doc.Close()
            
            # Now extract the docx text
            return self._extract_docx(abs_out_path)
            
        except Exception as e:
            logger.error(f"Error converting/extracting DOC: {e}")
            raise
        finally:
            if word is not None:
                word.Quit()
            
            # Cleanup temp .docx file
            if os.path.exists(abs_out_path):
                try:
                    os.remove(abs_out_path)
                except:
                    pass

