# Eval Chain

```{attention}
This feature is not implemented yet ! Will update this shortly 😊
```

get started with eval chains

```python
from ragrank.llm import openai
from ragrank.embedding import openai_embedding
from ragrank.vectorstores import faiss
from ragrank.chains import evalchain

llm = openai()
embedding = openai_embedding()
retriever = faiss.as_retriever()

chain = evalchain(
    llm=llm,
    embedding=embedding,
    retriever=retriever,
)

response = chain.invoke("Who is the president of America ?")
eval_result = response.evaluation()
eval_result
```