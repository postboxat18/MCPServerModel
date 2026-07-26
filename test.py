import json
import re
import sys
from datetime import datetime
# import faiss
# import numpy as np
# import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
os.environ["COLBERT_DISABLE_EXTENSIONS"] = "1"
os.environ["COLBERT_LOAD_TORCH_EXTENSION_VERBOSE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from sentence_transformers import SentenceTransformer
from ragatouille import RAGPretrainedModel
all_text = json.loads(open("context_list.json", "r+").read())
query = "How many leave days do employees get?"
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
            # txt = txt + page_txt
            chunk_list.append(txt)
        total_texts += context + page_txt
    if False:
    # if len(total_texts) < 50000:
        p=0
        # return total_texts
    else:
        # RAG = getRag(logfile)
        # RAG = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        top_k = int(len(all_text) * 0.2)
        # # sentence transformation
        # embeddings = RAG.encode(all_text, convert_to_numpy=True)
        # index = faiss.IndexFlatL2(embeddings.shape[1])
        # index.add(embeddings)
        # q_emb = RAG.encode([query], convert_to_numpy=True)
        # D, I = index.search(q_emb, k=top_k)
        # total_texts = "\n".join([all_text[i] for i in I[0]])

        # # colbert
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
        print(results)
        total_texts = "\n".join([data["content"] for data in results])

except Exception as e:
    print(e)