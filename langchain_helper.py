from dotenv import load_dotenv
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import JSONLoader
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()


def create_vector_db_from_proverbs() -> FAISS:
    loader = JSONLoader(
        file_path="./proverbs.json",
        jq_schema=".proverbs[].content",
    )

    data = loader.load()

    embeddings = OpenAIEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)

    db = FAISS.from_documents(docs, embeddings)
    return db


print(create_vector_db_from_proverbs())


def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([doc.page_content for doc in docs])

    llm = OpenAI(model="text-davinci-003")

    prompt_template = PromptTemplate(
        input_variables=["problem", "docs"],
        template="""
        You are an expirienced Bible scholar particularly interested in the book of Proverbs.
        You try to help people with situations in theire lives by giving them advice solely from the Proverbs.

        How should I deal with the following problem: "{problem}"
        By searching "{docs}" you try to find the advice best fit for their situation.

        You only use the Proverbs as a source of wisdom.

        If you feel like you don't have enough information, answer with "Maybe another book can help you".

        Explain why you choose the advice you choose as concise as possible and address me with "You".
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)

    response = chain.run(problem=query, docs=docs_page_content)
    response = response.replace("\n", " ")
    return response
