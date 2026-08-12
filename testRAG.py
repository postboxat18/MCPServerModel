from ragatouille import RAGPretrainedModel
import os
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import json
torch.device('cuda')
assert torch.cuda.is_available(), "CUDA not available"

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0",n_gpu=1)

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
    top_k = int(len(all_text) * 0.2)
    print(f"Total number of chunks: {len(chunk_list)}")
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
    # print(results)
    total_texts = "\n".join([data["content"] for data in results])
    print(total_texts)
except Exception as e:
    print(e)