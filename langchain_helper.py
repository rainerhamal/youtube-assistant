import chunk
import langchain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from openai import embeddings

load_dotenv()

embeddings = OpenAIEmbeddings()
# openai_api_key

# check if the video function is working
video_url = "https://youtu.be/lG7Uxts9SXs?si=NBf7WkD7BdOAZBDV"

def create_vector_db_from_youtubeUrl(video_url: str) -> FAISS:
    # load the youtube video
    loader = YoutubeLoader.from_youtube_url(video_url)
    # create a transcript of the video loaded
    transcript = loader.load()

    # split the transcript into smaller chunks so OpenAI can digest them accordingly base on the model you will be using in this case text-davince-003
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # save splitted etxt into a docs variable
    docs = text_splitter.split_documents(transcript)

    # initiate the FAISS
    db = FAISS.from_documents(docs, embeddings)

    return db

# check if the video function is working
# print(create_vector_db_from_youtubeUrl(video_url))

def get_response_from_query(db, query, k=4):
    # text-davince-003 4097 tokens
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model="text-davince-003")
    # llm = OpenAI(model="text-davince-003", openai_api_key=the_value)
    prompt = PromptTemplate(
        input_variables = ["question", "docs"],
        template = """
        You are a helpful Youtube assistant that can answer questions about the videos base on the video's transcript.

        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know".

        Your answers should be detailed.
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response