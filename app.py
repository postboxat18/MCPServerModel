from mcp.server.fastmcp import FastMCP
import os
import json
import sys
import faiss
import re
from paddleocr import PaddleOCR
from datetime import datetime
from paddle_coordinates import coordinates_process, log_exception, processLogger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
# from easyocr_with_redis import ocr_process_pdf_with_fitz, ocr_process_pdf

mcp = FastMCP("My Server", json_response=True)

logfile = "mcpLog.txt"


def split_textter(all_text, logfile):
    try:
        total_texts = ""
        chunk_list = []
        splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_overlap=0
        )

        for page_num, context in enumerate(all_text[:]):
            page_txt = f"\n\n\t\tThe Above Text is from the page number {page_num + 1}.\n\n"
            splTxt = splitter.split_text(re.sub(r"\s+", " ", context))
            for txt in splTxt:
                chunk_list.append(txt)
            total_texts += context + page_txt
        return total_texts, chunk_list
    except Exception as e:
        log_exception(e, "split_textter", logfile)

#
# @mcp.tool()
# def RAGColbertMethod(all_text: list, query: str, top_k: int = 4):
#     try:
#         total_texts, chunk_list = split_textter(all_text, logfile)
#         if len(total_texts) < 50000:
#             return total_texts
#         else:
#             # # colbert
#             RAGCol.index(
#                 collection=chunk_list,
#                 document_ids=[str(page + 1) for page, context in enumerate(chunk_list)],
#                 index_name="crux",
#                 overwrite_index=True,
#                 max_document_length=512,
#                 split_documents=True,
#                 use_faiss=True
#             )
#             results = RAGCol.search(query=query, k=top_k)
#             print(results)
#             total_texts = "\n".join([data["content"] for data in results])
#         return total_texts
#     except Exception as e:
#         log_exception(e, "RAGColMethod", logfile)


@mcp.tool()
def RAGFAISSMethod(all_text: list, query: str, top_k: int = 4):
    try:
        total_texts, chunk_list = split_textter(all_text, logfile)
        embeddings = RAGFaiss.encode(chunk_list, convert_to_numpy=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        q_emb = RAGFaiss.encode([query], convert_to_numpy=True)
        D, I = index.search(q_emb, k=top_k)
        total_texts = "\n".join([chunk_list[i] for i in I[0]])
        return total_texts
    except Exception as e:
        log_exception(e, "RAGFAISSMethod", logfile)
        return ""


@mcp.tool()
def PaddleOCRProcess(pdf_path):
    try:
        print("file exist",os.path.isfile(pdf_path))
        ocr_result = coordinates_process(pipeline, pdf_path, logfile)
        return ocr_result
    except Exception as e:
        log_exception(e, "ocr process", logfile)
        return "failed to PaddleOCRProcess process"


@mcp.tool()
def EasyOCRProcess(pdf_path):
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=True)
        print("reader")
        # m = 1
        # if m:
        #     result = ocr_process_pdf_with_fitz(pdf_path, logfile, reader)
        # else:
        #     result = ocr_process_pdf(pdf_path, logfile, reader)
        # return result
    except Exception as e:
        log_exception(e, "ocr process", logfile)
        return "failed to PaddleOCRProcess process"


@mcp.tool()
def SVMMethod(all_text, tag_name_list, chunk_size):
    try:
        return "work in progress"
    except Exception as e:
        log_exception(e, "SVMMethod", logfile)
        return "failed to process SVM Model"


@mcp.tool()
def NERMethod(all_text, tag_name_list, chunk_size):
    try:
        # CHUNKING
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import Tokenizer
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
        chunk_list = []
        for context in all_text:
            ls = splitter.split_text(context)
            chunk_list.extend(ls)
        # CREATE A TAG LIST
        dataset_list = []
        for context in chunk_list:
            dp = {}
            dp["context"] = context
            tag_list = []
            for word in context.split():
                if word.lower() in tag_name_list:
                    if tag_list:
                        if tag_list[::-1] == "B-TAG":
                            tag_list.append("I-TAG")
                        else:
                            tag_list.append("B-TAG")
                else:
                    tag_list.append("O")
            if "B-TAG" in tag_list:
                dp["tag"] = tag_list
                dataset_list.append(dp)

        # ASSIGN
        x_res = [z["context"] for z in dataset_list]
        y_res = [z["tag"] for z in dataset_list]
        unique_id_ = sorted(list(set([i for data in y_res for i in data])), key=lambda x: x)
        tag_list = ["O", unique_id_[len(unique_id_) - 2]]
        tag_id = {}
        for i, data in enumerate(tag_list):
            tag_id[data] = i
        import string
        chars = string.punctuation
        tokenizer = Tokenizer(num_words=6000, filters=chars)
        tokenizer.fit_on_texts(x_res)
        text2seq = tokenizer.texts_to_sequences(x_res)
        word2id = tokenizer.word_index
        id2word = [{m: v} for v, m in word2id.items()]
        x_pre = text2seq
        y_pre = []
        for tag_list_ in y_res:
            tag_pad = []
            for tag in tag_list_:
                tag_pad.append(tag_id[tag])
            y_pre.append(tag_pad)

        x_pad = pad_sequences(x_pre, maxlen=chunk_size, padding='post', truncating='post', value=0)
        y_pad = pad_sequences(y_pre, maxlen=chunk_size, padding='post', truncating='post', value=0)
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x_pad, y_pad, random_state=4, shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42,
                                                          shuffle=True)
        import tensorflow as tf
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
        test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(100,)),
            tf.keras.layers.Embedding(input_dim=6000, output_dim=128, input_length=chunk_size),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(tag_list), activation='softmax'))
        ])
        model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.summary()
        model.fit(train_data, validation_data=val_data, epochs=5,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
        model.evaluate(val_data)
        model.save("model/dataset_with_pagetext_model.keras")
        return "model/dataset_with_pagetext_model.keras"
    except Exception as e:
        log_exception(e, "NERMethod", logfile)
        return "failed to NER process"

if __name__ == '__main__':
    pipeline = PaddleOCR(lang="en")
    RAGFaiss = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # RAGCol = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    mcp.run(transport='streamable-http')
