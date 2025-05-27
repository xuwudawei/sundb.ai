"""import logging
import docx
import pptx
import openpyxl
from pydantic import BaseModel
from typing import Generator, IO
from pypdf import PdfReader

from app.models import Document, Upload
from app.file_storage import default_file_storage
from app.types import MimeTypes
from .base import BaseDataSource

logger = logging.getLogger(__name__)


class FileConfig(BaseModel):
    file_id: int


class FileDataSource(BaseDataSource):
    def validate_config(self):
        if not isinstance(self.config, list):
            raise ValueError("config must be a list")
        for f_config in self.config:
            FileConfig.model_validate(f_config)

    def load_documents(self) -> Generator[Document, None, None]:
        for f_config in self.config:
            upload_id = f_config["file_id"]
            upload = self.session.get(Upload, upload_id)
            if upload is None:
                logger.error(f"Upload with id {upload_id} not found")
                continue

            with default_file_storage.open(upload.path) as f:
                if upload.mime_type == MimeTypes.PDF:
                    content = extract_text_from_pdf(f)
                    mime_type = MimeTypes.PLAIN_TXT
                elif upload.mime_type == MimeTypes.DOCX:
                    content = extract_text_from_docx(f)
                    mime_type = MimeTypes.PLAIN_TXT
                elif upload.mime_type == MimeTypes.PPTX:
                    content = extract_text_from_pptx(f)
                    mime_type = MimeTypes.PLAIN_TXT
                elif upload.mime_type == MimeTypes.XLSX:
                    content = extract_text_from_xlsx(f)
                    mime_type = MimeTypes.PLAIN_TXT
                else:
                    content = f.read()
                    mime_type = upload.mime_type
            document = Document(
                name=upload.name,
                hash=hash(content),
                content=content,
                mime_type=mime_type,
                data_source_id=self.data_source_id,
                user_id=self.user_id,
                source_uri=upload.path,
                last_modified_at=upload.created_at,
            )
            yield document


def extract_text_from_pdf(file: IO) -> str:
    reader = PdfReader(file)
    print("\n\n".join([page.extract_text() for page in reader.pages]))
    return "\n\n".join([page.extract_text() for page in reader.pages])


def extract_text_from_docx(file: IO) -> str:
    document = docx.Document(file)
    full_text = []
    for paragraph in document.paragraphs:
        full_text.append(paragraph.text)
    return "\n\n".join(full_text)


def extract_text_from_pptx(file: IO) -> str:
    presentation = pptx.Presentation(file)
    full_text = []
    for slide in presentation.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                full_text.append(shape.text)
    return "\n\n".join(full_text)


def extract_text_from_xlsx(file: IO) -> str:
    wb = openpyxl.load_workbook(file)
    full_text = []
    for sheet in wb.worksheets:
        full_text.append(f"Sheet: {sheet.title}")
        sheet_string = "\n".join(
            ",".join(map(str, row))
            for row in sheet.iter_rows(values_only=True)
        )
        full_text.append(sheet_string)
    return "\n\n".join(full_text)
"""
import logging
import re
import docx
import pptx
import openpyxl
from pydantic import BaseModel
from typing import Generator, IO, Optional
from pypdf import PdfReader
from io import BytesIO
from app.models import Document, Upload
from app.file_storage import default_file_storage
from app.types import MimeTypes
from .base import BaseDataSource

logger = logging.getLogger(__name__)

class FileConfig(BaseModel):
    file_id: int


class FileDataSource(BaseDataSource):
    def validate_config(self):
        if not isinstance(self.config, list):
            raise ValueError("config must be a list")
        for f_config in self.config:
            FileConfig.model_validate(f_config)

    def load_documents(self) -> Generator[Document, None, None]:
        for f_config in self.config:
            upload_id = f_config["file_id"]
            upload = self.session.get(Upload, upload_id)
            if upload is None:
                logger.warning(f"Upload ID {upload_id} not found in database.")
                continue

            with default_file_storage.open(upload.path, 'rb') as f:
                raw_content = f.read()

                # 解析章节信息
                part_number, chapter_number, section_number = self.parse_section_numbers(upload.name)
                logger.info(f"Parsed sections: part={part_number}, chapter={chapter_number}, section={section_number} for file {upload.name}")

                # 处理所有文本类型文件
                if upload.mime_type in [
                    MimeTypes.PLAIN_TXT,
                    MimeTypes.MARKDOWN,
                    MimeTypes.PDF,
                    MimeTypes.DOCX,
                    MimeTypes.PPTX,
                    MimeTypes.XLSX
                ]:
                    if upload.mime_type == MimeTypes.PDF:
                        content = extract_text_from_pdf(BytesIO(raw_content))
                    elif upload.mime_type == MimeTypes.DOCX:
                        content = extract_text_from_docx(BytesIO(raw_content))
                    elif upload.mime_type == MimeTypes.PPTX:
                        content = extract_text_from_pptx(BytesIO(raw_content))
                    elif upload.mime_type == MimeTypes.XLSX:
                        content = extract_text_from_xlsx(BytesIO(raw_content))
                    else:
                        content = self._decode_text(raw_content)

                    mime_type = MimeTypes.PLAIN_TXT
                else:
                    content = raw_content
                    mime_type = upload.mime_type

                yield Document(
                    name=upload.name,
                    hash=hash(content),
                    content=content,
                    mime_type=mime_type,
                    data_source_id=self.data_source_id,
                    user_id=self.user_id,
                    source_uri=upload.path,
                    last_modified_at=upload.created_at,
                    part_number=part_number,
                    chapter_number=chapter_number,
                    section_number=section_number
                )

    def _decode_text(self, raw_content: bytes) -> str:
        encodings = ['utf-8', 'gb18030', 'big5', 'shift_jis', 'iso-8859-1']
        for encoding in encodings:
            try:
                return raw_content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw_content.decode('utf-8', errors='replace')

    def parse_section_numbers(self, filename: str):
        """
        解析文件名，提取 part_number、chapter_number、section_number
        文件名格式示例：[1][章]-[2][节]-[3][小节].txt 或 [1.3][小节].txt
        """
        # 优先级 1：匹配完整的三级结构（part-chapter-section）
        full_pattern = re.compile(
            r"\[(\d+)\]\[.*?\]-\[(\d+)\]\[.*?\]-\[(\d+(\.\d+)?)\]\[.*?\]"
        )
        match = full_pattern.search(filename)
        if match:
            part = int(match.group(1))
            chapter = int(match.group(2))
            section_str = match.group(3)
            section = int(section_str.split(".")[-1])  # 提取最后一个数字
            return part, chapter, section

        # 优先级 2：匹配二级结构（part-chapter 或 part-section 或 chapter-section）
        patterns = [
            # part-chapter
            r"\[(\d+)\]\[.*?\]-\[(\d+)\]\[.*?\]",
            # part-section（无 chapter）
            r"\[(\d+)\]\[.*?\]-\[(\d+(\.\d+)?)\]\[.*?\]",
            # chapter-section（无 part）
            r"\[(\d+)\]\[.*?\]-\[(\d+(\.\d+)?)\]\[.*?\]",
            # 单独 section
            r"\[(\d+(\.\d+)?)\]\[.*?\]"
        ]

        part, chapter, section = None, None, None
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                groups = match.groups()
                if len(groups) >= 2 and all(groups[:2]):  # part-chapter 或 part-section
                    first, second = groups[0], groups[1]
                    if second.isdigit():
                        part = int(first)
                        chapter = int(second)
                    else:
                        part = int(first)
                        section_str = second.split(".")[-1]
                        section = int(section_str)
                elif len(groups) >= 1 and groups[0]:  # 单独 section 或 chapter
                    group = groups[0]
                    if "." in group:
                        section = int(group.split(".")[-1])
                    else:
                        chapter = int(group) if "chapter" in pattern else int(group)

        logger.debug(f"解析文件名: {filename}")
        logger.debug(f"匹配结果: part={part}, chapter={chapter}, section={section}")
        return part, chapter, section


def extract_text_from_pdf(file: IO) -> str:
    reader = PdfReader(file)
    return "\n\n".join([page.extract_text() for page in reader.pages if page.extract_text()])


def extract_text_from_docx(file: IO) -> str:
    document = docx.Document(file)
    return "\n\n".join(paragraph.text for paragraph in document.paragraphs)


def extract_text_from_pptx(file: IO) -> str:
    presentation = pptx.Presentation(file)
    return "\n\n".join(shape.text for slide in presentation.slides for shape in slide.shapes if hasattr(shape, "text"))


def extract_text_from_xlsx(file: IO) -> str:
    wb = openpyxl.load_workbook(file)
    return "\n\n".join(
        f"Sheet: {sheet.title}\n" + "\n".join(
            ",".join(map(str, row)) for row in sheet.iter_rows(values_only=True)
        )
        for sheet in wb.worksheets
    )