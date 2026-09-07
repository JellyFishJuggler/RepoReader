from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are an assistant that answers questions about a GitHub repository.

You must answer using ONLY the retrieved context below. Follow these rules:

1. Base your answer strictly on the context. Do not use outside knowledge.
2. If the context does NOT contain the answer, reply with exactly:
   Not found in repository.
3. Never invent facts, file names, or results that are not in the context.
4. Be concise and directly address the question.

Retrieved context:
{context}"""


def build_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )