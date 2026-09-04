# Databricks RAG Pipeline

from pyspark.sql import Row
from pyspark.sql.functions import concat_ws, col

# 1. Load TXT documents
base_path = "file:/Workspace/Users/jatin.seelam@gmail.com/"
files = dbutils.fs.ls(base_path)

documents = []
for f in files:
    if f.path.endswith(".txt"):
        content = dbutils.fs.head(f.path, 100000)
        documents.append(Row(file_name=f.name, content=content))

docs_df = spark.createDataFrame(documents)
display(docs_df)

# 2. Chunk documents
chunk_size = 500
chunk_overlap = 50
chunks = []

for row in docs_df.collect():
    text = row.content
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append(
            Row(file_name=row.file_name, chunk_id=chunk_id, content=chunk_text)
        )

        chunk_id += 1
        start += chunk_size - chunk_overlap

chunks_df = spark.createDataFrame(chunks)
display(chunks_df)

# 3. Test embedding model
test_text = "Employees receive 24 days of annual leave."

result = spark.sql(f"""
SELECT ai_query(
    'databricks-gte-large-en',
    '{test_text}'
) AS embedding
""")
display(result)

# 4. Generate embeddings
embedded_df = chunks_df.selectExpr(
    "file_name",
    "chunk_id",
    "content",
    "ai_query('databricks-gte-large-en', content) AS embedding"
)
display(embedded_df)

# 5. Create unique ID
embedded_df = embedded_df.withColumn(
    "id",
    concat_ws("_", col("file_name"), col("chunk_id"))
)
display(embedded_df)

# 6. Save to Delta table
spark.sql("DROP TABLE IF EXISTS workspace.default.embeddings")

embedded_df.write     .format("delta")     .mode("overwrite")     .saveAsTable("workspace.default.embeddings")

# 7. Enable Change Data Feed
spark.sql("""
ALTER TABLE workspace.default.embeddings
SET TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
)
""")

# 8. AI Search index
# Created/configured in Databricks UI:
# Endpoint: rag-search-endpoint
# Index: workspace.default.rag_embeddings_index
# Source: workspace.default.embeddings
# Type: Delta Sync
# Search mode: Hybrid
# Primary key: id
# Text column: content
# Embedding column: embedding
# Dimension: 1024
# Update mode: Triggered

# 9. Retrieve relevant chunks
question = "How many annual leave days do employees receive?"

retrieved_df = spark.sql(f"""
SELECT *
FROM vector_search(
    index => 'workspace.default.rag_embeddings_index',
    query_text => '{question}',
    query_type => 'HYBRID',
    num_results => 3
)
""")
display(retrieved_df)

# 10. Build context
context = "\n\n".join(
    row["content"]
    for row in retrieved_df.select("content").collect()
)
print(context)

# 11. Build grounded prompt
prompt = f"""
You are a helpful company assistant.

Answer the question using ONLY the context provided below.
If the context does not contain the answer, say:
"I don't have enough information to answer that."

Context:
{context}

Question:
{question}
"""
print(prompt)

# 12. Send context + question to LLM
prompt_df = spark.createDataFrame([(prompt,)], ["prompt"])

answer_df = prompt_df.selectExpr(
    "ai_query('databricks-meta-llama-3-3-70b-instruct', prompt) AS answer"
)

display(answer_df)
