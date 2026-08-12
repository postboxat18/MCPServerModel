import os
import re

import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragatouille import RAGPretrainedModel

os.environ["COLBERT_DISABLE_EXTENSIONS"] = "1"
os.environ["COLBERT_LOAD_TORCH_EXTENSION_VERBOSE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.device('cuda')

import sys
from datetime import datetime

from flask import Flask, request

assert torch.cuda.is_available(), "CUDA not available"

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0", n_gpu=1)

logfile = "demolog.txt"
app = Flask(__name__)


def log_exception(e, func_name, logfile):
    exc_type, exc_obj, tb = sys.exc_info()
    lineno = tb.tb_lineno
    error_message = f"\n{datetime.now()}In {func_name} LINE.NO-{lineno} : {exc_obj}"
    print(error_message)
    with open(logfile, 'a', encoding='utf-8') as fp:
        fp.writelines(error_message)


def processLogger(process, logfile):
    print(process)
    with open(logfile, 'a', encoding='utf-8') as fp:
        fp.writelines(f'\n{datetime.now()} {process}')


@app.route('/', methods=['GET', 'POST'])
def relevant_query():
    try:
        if request.method == 'POST':
            query = request.json.get('query', "")
            all_text = request.json.get('all_text', "")
            top_k = request.json.get('top_k', "")
            total_texts = ""
            chunk_list = []
            splitter = RecursiveCharacterTextSplitter(
                separators="\n",
                chunk_overlap=0
            )

            for page_num, context in enumerate(all_text):
                page_txt = f"\n\n\t\tThe Above Text is from the page number {page_num + 1}.\n\n"
                splTxt = splitter.split_text(re.sub(r"\s+", " ", context))
                for txt in splTxt:
                    # txt = txt + page_txt
                    chunk_list.append(txt)
                total_texts += context + page_txt
            if not top_k:
                top_k = int(len(all_text) * 0.2)
            RAG.index(
                collection=chunk_list,
                document_ids=[str(page + 1) for page, context in enumerate(chunk_list)],
                index_name="crux",
                overwrite_index=True,
                max_document_length=512,
                split_documents=True,
                use_faiss=True
            )
            results = RAG.search(query=query, k=top_k)
            total_texts = "\n".join([data["content"] for data in results])
            return total_texts

    except Exception as e:
        log_exception(e, "upload_doc", logfile)
        return ""


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9001)
