import copy
import re
from typing import Iterable

from langchain_text_splitters import CharacterTextSplitter

from lfx.custom.custom_component.component import Component
from lfx.io import DropdownInput, HandleInput, IntInput, MessageTextInput, Output
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message
from lfx.utils.util import unescape_string
from lfx.log import logger

from langchain_core.documents import Document


class SplitTextComponent(Component):
    display_name: str = "Split Text"
    description: str = "Split text into chunks based on specified criteria."
    documentation: str = "https://docs.langflow.org/components-processing#split-text"
    icon = "scissors-line-dashed"
    name = "SplitText"

    inputs = [
        HandleInput(
            name="data_inputs",
            display_name="Input",
            info="The data with texts to split in chunks.",
            input_types=["Data", "DataFrame", "Message"],
            required=True,
        ),
        IntInput(
            name="chunk_overlap",
            display_name="Chunk Overlap",
            info="Number of characters to overlap between chunks.",
            value=200,
        ),
        IntInput(
            name="chunk_size",
            display_name="Chunk Size",
            info=(
                "The maximum length of each chunk. Text is first split by separator, "
                "then chunks are merged up to this size. "
                "Individual splits larger than this won't be further divided."
            ),
            value=1000,
        ),
        MessageTextInput(
            name="separator",
            display_name="Separator",
            info=(
                "The character to split on. Use \\n for newline. "
                "Examples: \\n\\n for paragraphs, \\n for lines, . for sentences"
            ),
            value="\n",
        ),
        MessageTextInput(
            name="text_key",
            display_name="Text Key",
            info="The key to use for the text column.",
            value="text",
            advanced=True,
        ),
        DropdownInput(
            name="keep_separator",
            display_name="Keep Separator",
            info="Whether to keep the separator in the output chunks and where to place it.",
            options=["False", "True", "Start", "End"],
            value="False",
            advanced=True,
        ),
        DropdownInput(
            name="splitter_type",
            display_name="Splitter Type",
            info="Which text splitter to use to chunk the documents.",
            options=["CharacterTextSplitter", "TableAwareTextSplitter", "LineBasedTextSplitter"],
            value="CharacterTextSplitter",
            advanced=True,
        ),
        MessageTextInput(
            name="model_id",
            display_name="Model ID",
            info="The name of the model that will be used for computing embeddings.",
            value="ibm-granite/granite-embedding-30m-english",
            advanced=True,
        ),
        DropdownInput(
            name="use_document_title",
            display_name="Use Document Title",
            info="Whether to use the document title as a prefix in each chunk.",
            options=["False", "True"],
            value="False",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Chunks", name="dataframe", method="split_text"),
    ]

    def _docs_to_data(self, docs) -> list[Data]:
        return [Data(text=doc.page_content, data=doc.metadata) for doc in docs]

    def _fix_separator(self, separator: str) -> str:
        """Fix common separator issues and convert to proper format."""
        if separator == "/n":
            return "\n"
        if separator == "/t":
            return "\t"
        return separator

    @staticmethod
    def to_bool(val):
        if isinstance(val, str):
            if val.lower() == "false":
                return False
            elif val.lower() == "true":
                return True
        elif isinstance(val, bool):
            return val
        raise RuntimeError(f"Cannot convert value {val} to a boolean value. Expected 'True' or 'False'.")

    def split_text_base(self):
        separator = self._fix_separator(self.separator)
        separator = unescape_string(separator)

        if isinstance(self.data_inputs, DataFrame):
            if not len(self.data_inputs):
                msg = "DataFrame is empty"
                raise TypeError(msg)

            self.data_inputs.text_key = self.text_key
            try:
                documents = self.data_inputs.to_lc_documents()
            except Exception as e:
                msg = f"Error converting DataFrame to documents: {e}"
                raise TypeError(msg) from e
        elif isinstance(self.data_inputs, Message):
            self.data_inputs = [self.data_inputs.to_data()]
            return self.split_text_base()
        else:
            if not self.data_inputs:
                msg = "No data inputs provided"
                raise TypeError(msg)

            documents = []
            if isinstance(self.data_inputs, Data):
                self.data_inputs.text_key = self.text_key
                documents = [self.data_inputs.to_lc_document()]
            else:
                try:
                    documents = [input_.to_lc_document() for input_ in self.data_inputs if isinstance(input_, Data)]
                    if not documents:
                        msg = f"No valid Data inputs found in {type(self.data_inputs)}"
                        raise TypeError(msg)
                except AttributeError as e:
                    msg = f"Invalid input type in collection: {e}"
                    raise TypeError(msg) from e
        try:
            if self.splitter_type == "CharacterTextSplitter":
                # Convert string 'False'/'True' to boolean
                keep_sep = self.to_bool(self.keep_separator)
                logger.debug("SPLIT: Creating a CharacterTextSplitter..")
                splitter = CharacterTextSplitter(
                    chunk_overlap=self.chunk_overlap,
                    chunk_size=self.chunk_size,
                    separator=separator,
                    keep_separator=keep_sep,
                )
            elif self.splitter_type == "LineBasedTextSplitter":
                use_document_title = self.to_bool(self.use_document_title)
                splitter = LineBasedTextSplitter(
                    chunk_size=self.chunk_size,
                    model_id=self.model_id,
                    use_document_title=use_document_title,
                )
            elif self.splitter_type == "TableAwareTextSplitter":
                logger.debug(f"SPLIT: Creating a TableAwareTextSplitter with chunk_size={self.chunk_size} and model_id '{self.model_id}'.")
                splitter = TableAwareTextSplitter(
                    chunk_size=self.chunk_size,
                    model_id=self.model_id
                )
            else:
                raise RuntimeError(f"Unknown splitter type value '{self.splitter_type}'.")
            return splitter.split_documents(documents)
        except Exception as e:
            msg = f"Error splitting text: {e}"
            raise TypeError(msg) from e

    def split_text(self) -> DataFrame:
        return DataFrame(self._docs_to_data(self.split_text_base()))

class LineBasedTextSplitter:
    def __init__(
        self,
        chunk_size: int,
        model_id: str,
        prefix: str = "",
        use_document_title: bool = False,
    ):
        self._chunk_size = chunk_size
        self.use_tiktoken = False
        if model_id in ["text-embedding-3-small", "text-embedding-3-large"]:
            self.use_tiktoken = True
        logger.debug(f"SPLIT: Initializing LineBasedTextSplitter, chunk_size = {chunk_size}, model_id = '{model_id}', use_tiktoken = {self.use_tiktoken}, use_document_title={use_document_title}.")
        if self.use_tiktoken:
            import tiktoken
            # The tokenizer for text-embedding-3-small, text-embedding-3-large
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_id,
            )
        self._prefix = ""
        self._prefix_len = 0
        self.set_prefix(prefix)
        self.use_document_title = use_document_title

    def set_prefix(self, prefix):
        logger.debug(f"SPLIT: setting prefix to '{prefix}'..")
        prefix_len = len(self.tokenize(prefix))
        if prefix_len >= self._chunk_size:
            raise RuntimeError(
                f"Chunks prefix: {prefix} is too long for chunk size {self._chunk_size}"
            )
        else:
            self._prefix = prefix
            self._prefix_len = prefix_len

    def tokenize(self, text: str) -> list[int]:
        if self.use_tiktoken:
            return self._tokenizer.encode(text)
        else:
            return self._tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, tokens: list[int]):
        return self._tokenizer.decode(tokens)

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """Given Documents, chunk the text to smaller pieces and return them as list of Documents"""

        chunks = []
        for document in documents:
            chunks.extend(self._chunk_document(document))
        return chunks

    def _chunk_document(self, document: Document):
        document_text = document.page_content
        document_metadata = document.metadata
        chunks = []
        chunk_seq_num = 0

        first_character_index = document_metadata.get("start_index", 0)
        if self.use_document_title:
            file_name = document_metadata.get("filename", "unknown-file-name")
            logger.debug(f"SPLIT: Chunking document with file name '{file_name}'..")
            document_title = get_title(file_name)
            logger.debug(f"SPLIT: Found title '{document_title}'..")
            self.set_prefix(document_title)

        current = self._prefix
        current_len = self._prefix_len

        new_line_token_count =  len(self.tokenize("\n"))
        lines = document_text.split("\n")
        for line in lines:
            line_tokens = self.tokenize(line)

            while (
                len(line_tokens) > self._chunk_size - current_len
            ):  # line cannot fit into current
                num_available_tokens_in_chunk = (
                    self._chunk_size - current_len
                    if len(line_tokens) + self._prefix_len > self._chunk_size
                    else 0
                )  # if whole line can fit into a new chunk, do not add anything to current chunk,
                # otherwise, split the line between current and next chunks.

                if num_available_tokens_in_chunk > 0:
                    # split line
                    if current:
                        current += "\n"
                        current_len += new_line_token_count
                    current += self.decode_tokens(
                        line_tokens[:num_available_tokens_in_chunk]
                    )
                    current_len += num_available_tokens_in_chunk

                # add current chunk
                chunks.append(
                    self._new_chunk(
                        current, chunk_seq_num, first_character_index, document_metadata
                    )
                )

                # new current chunk
                first_character_index += len(current)
                chunk_seq_num += 1
                current = self._prefix
                current_len = self._prefix_len
                line_tokens = line_tokens[num_available_tokens_in_chunk:]

            # rest of line fits into current
            if len(line_tokens) > 0:
                if current:
                    current += "\n"
                    current_len += new_line_token_count
                current += self.decode_tokens(line_tokens)
                current_len += len(line_tokens)

        # final chunk
        chunks.append(
            self._new_chunk(
                current, chunk_seq_num, first_character_index, document_metadata
            )
        )

        return chunks

    @staticmethod
    def _new_chunk(
        text: str, seq_no: int, start_index: int, doc_metadata: dict
    ) -> Document:
        chunk_metadata = copy.deepcopy(doc_metadata)
        chunk_metadata["sequence_number"] = seq_no
        chunk_metadata["start_index"] = start_index
        return Document(page_content=text, metadata=chunk_metadata)


class TableAwareTextSplitter:

    def __init__(self, chunk_size: int, model_id: str):
        self.chunk_size = chunk_size
        self.model_id = model_id

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """Given Documents, chunk the text to smaller pieces and return them as list of Documents"""

        chunks = []
        for document in documents:
            chunks.extend(self._chunk_document(document))
        return chunks

    def _chunk_document(self, document: Document) -> list[Document]:
        segments = self._get_segments(document)

        chunks = []
        for segment in segments:
            prefix = self.get_prefix(segment)
            line_splitter = LineBasedTextSplitter(
                chunk_size=self.chunk_size,
                model_id=self.model_id,
                prefix=prefix
            )

            chunks.extend(line_splitter.split_documents([segment]))

        return chunks

    # fix me: does not indicate sub headers
    def _get_segments(self, doc):
        segments = []
        doc_metadata = doc.metadata
        segments_count = 0
        start_index = doc.metadata.get("start_index", 0)
        current_segment = Document(
            page_content="",
            metadata={"type": "text", "seq_no": segments_count, "start_index": start_index}
            | doc_metadata,
        )
        separator_found = False
        lines = doc.page_content.split("\n")
        for line in lines:

            if self._is_table_line(line):
                if current_segment.metadata["type"] != "table":  # first table line
                    segments.append(current_segment)
                    segments_count += 1
                    start_index += len(current_segment.page_content)
                    current_segment = Document(
                        page_content="",
                        metadata={
                            "type": "table",
                            "caption": self.get_caption(current_segment),
                            "header": self.condense_table_row(line),
                            "seq_no": segments_count,
                            "start_index": start_index,
                        }
                        | doc_metadata,
                    )
                    separator_found = False
                elif self._is_table_seperator(line):

                    separator_found = True
                    current_segment.metadata[
                        "header"
                    ] += "\n" + self.condense_separator(line)
                elif not separator_found:

                    current_segment.metadata[
                        "header"
                    ] += "\n" + self.condense_table_row(line)
                else:
                    current_segment.page_content += "\n" + line

            else:  # text line
                if current_segment.metadata["type"] == "table":
                    segments.append(current_segment)
                    segments_count += 1
                    start_index += len(current_segment.page_content)
                    current_segment = Document(
                        page_content="",
                        metadata={
                            "type": "text",
                            "seq_no": segments_count,
                            "start_index": start_index,
                        }
                        | doc_metadata,
                    )
                current_segment.page_content += "\n" + line

        # last segment
        segments.append(current_segment)
        return [c for c in segments if len(c.page_content.strip()) > 0]

    @staticmethod
    def get_prefix(segment: Document) -> str:
        if segment.metadata["type"] == "text":
            return ""
        elif segment.metadata["type"] == "table":
            result = segment.metadata["caption"]
            if result:
                result += "\n"
            result += segment.metadata["header"]
            return result
        else:
            raise RuntimeError(f"Internal error: unknown segment type '{segment['type']}' for segment {segment}.")

    # returns last sentence before table
    @staticmethod
    def get_caption(prev_segment) -> str:
        last_sentence = prev_segment.page_content.strip().split("\n")[-1].split(".")[-1]
        return last_sentence

    # each line starting with | is included in table
    @staticmethod
    def _is_table_line(line: str):
        return line.startswith("|")

    @staticmethod
    def _is_table_seperator(line: str):
        cells = [c.strip() for c in line.strip().split("|")]
        return all(
            re.match(r"[-]+", cell.strip()) for cell in cells if len(cell.strip()) > 0
        )

    @staticmethod
    def condense_separator(line: str):
        numCells = len(line.strip().split("|")) - 2
        return "| --- " * numCells + "|"

    @staticmethod
    def condense_table_row(line: str) -> str:
        if sum([t.isalnum() for t in line]) == 0:
            return ""
        cells = [c.strip() for c in line.strip().split("|")]

        return " | ".join(cells).strip()

def get_title(file_name: str) -> str:
    file_name_to_title = {
        "docling.pdf": "Docling Technical Report"
    }
    file_name_to_title.update(filename_to_output)
    return file_name_to_title.get(file_name, "")


filename_to_output = {
  "Alaska-2017.pdf": """This document is the 2017 annual report (Form 10-K) of Alaska Air Group, Inc., filed with the United States Securities and Exchange Commission (SEC). The report covers the fiscal year ended December 31, 2017. Important entities mentioned include:

* Alaska Air Group, Inc. (the company)
* United States Securities and Exchange Commission (SEC)
* New York Stock Exchange (where the company's common stock is registered)

Important dates mentioned include:

* December 31, 2017 (end of the fiscal year)
* January 31, 2018 (date of share outstanding total)
* June 30, 2017 (date used to calculate aggregate market value of shares held by nonaffiliates)""",
  "Alaska-2018.pdf": """This document is the 2018 annual report (Form 10-K) of Alaska Air Group, Inc., filed with the United States Securities and Exchange Commission (SEC). The report covers the fiscal year ended December 31, 2018. Important entities mentioned include:

* Alaska Air Group, Inc. (the company)
* United States Securities and Exchange Commission (SEC)
* New York Stock Exchange (where the company's common stock is listed)

Important dates mentioned include:

* December 31, 2018 (end of the fiscal year)
* January 31, 2019 (date of share outstanding total)
* June 30, 2018 (date used to calculate aggregate market value of shares held by nonaffiliates)""",
  "AmericanAirlines-2017.pdf": "This document is the 2017 annual report (Form 10-K) of American Airlines Group Inc., filed with the United States Securities and Exchange Commission (SEC).",
  "AmericanAirlines-2018.pdf": "The document \"AmericanAirlines-2018.pdf\" is the 2018 Annual Report on Form 10-K for American Airlines Group Inc.",
  "AmericanAirlines-2019.pdf": "This document is the 2020 Annual Report on Form 10-K for American Airlines Group Inc., filed for the year ending 2019.",
  "Delta-2017.pdf": "This document is the 2017 annual report (Form 10-K) of Delta Air Lines, Inc. for the fiscal year ended December 31, 2017.",
  "Delta-2018.pdf": "This document is the 2018 annual report (Form 10-K) of Delta Air Lines, Inc. for the fiscal year ended December 31, 2018.",
  "Delta-2019.pdf": "This document is the 2019 annual report (Form 10-K) of Delta Air Lines, Inc. for the fiscal year ended December 31, 2019.",
  "Southwest-2017.pdf": "This document is the 2017 Annual Report to Shareholders of Southwest Airlines Co.",
  "Southwest-2018.pdf": "This document is the 2018 Annual Report to Shareholders of Southwest Airlines Co.",
  "Southwest-2019.pdf": "This document is the 2019 Annual Report to Shareholders of Southwest Airlines Co.",
  "United-2017.pdf": "This document is the 2017 annual report (Form 10-K) of United Continental Holdings, Inc. and United Airlines, Inc.",
  "United-2018.pdf": "This document is the 2018 annual report (Form 10-K) of United Continental Holdings, Inc. and United Airlines, Inc.",
  "United-2019.pdf": "This document is the 2019 annual report (Form 10-K) of United Airlines Holdings, Inc. and United Airlines, Inc."
}
