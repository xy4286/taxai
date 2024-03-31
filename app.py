

import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from uuid import uuid4
from langchain_core.tracers.context import tracing_v2_enabled

import os
from dotenv import load_dotenv
import functools, operator, requests, os, json
from bs4 import BeautifulSoup
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.vectorstores import FAISS

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from langchain.schema.output_parser import StrOutputParser

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from typing import Dict, TypedDict
from langchain_core.messages import BaseMessage



import json
import operator
from typing import Annotated, Sequence, TypedDict

from langchain import hub
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pprint

from langgraph.graph import END, StateGraph

# ===================================================================



if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

if "TAVILY_API_KEY" in st.secrets:
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]


if "LANGCHAIN_API_KEY" in st.secrets:
    LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]

# load_dotenv()


# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph Research Agents"




print('--- 33333--')


# def _set_if_undefined(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = input(f"Please provide your {var}")


# load_dotenv()        


# _set_if_undefined("OPENAI_API_KEY")
# # _set_if_undefined("LANGCHAIN_API_KEY")
# _set_if_undefined("TAVILY_API_KEY")
# _set_if_undefined("LANGCHAIN_API_KEY")





unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"self-rag - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY



# urls = [
#     'https://zh.wikipedia.org/wiki/%E5%88%98%E7%A6%B9%E9%94%A1',
#  ]



# docs = [WebBaseLoader(url).load() for url in urls]
# # print(docs)
# # print('here')
# docs_list = [item for sublist in docs for item in sublist]

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=250, chunk_overlap=0
# )
# doc_splits = text_splitter.split_documents(docs_list)

# # Add to vectorDB
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=OpenAIEmbeddings(),
# )
# retriever = vectorstore.as_retriever()

retriever = Any

def create_retriever(urls):
    docs = [WebBaseLoader(url).load() for url in urls]
    # print(docs)
    # print('here')
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    # vectorstore = Chroma.from_documents(
    #     documents=doc_splits,
    #     collection_name="rag-chroma",
    #     embedding=OpenAIEmbeddings(),
    # )
    # retriever = vectorstore.as_retriever()

    vectorstore = FAISS.from_documents(
        doc_splits, OpenAIEmbeddings()
    )

    retriever = vectorstore.as_retriever()

    return retriever

# retriever = create_retriever(urls)


st.set_page_config(page_title='ask questions', page_icon="🤖")
st.title("Ask Questions")

if 'urls' not in st.session_state:    
        st.session_state.urls = []

def submit():
    print('=== inside submit() ==')
    website_url = st.session_state.widget
    st.session_state.urls.append(website_url)   
    st.session_state.widget = '' 
    
print(f'=== before urls loop, show me urls: {st.session_state.urls}')
for url in st.session_state.urls:
    with st.sidebar:
        st.sidebar.write(url)


with st.sidebar:
    st.header("Kindly input your website URL below and press 'Enter'")
    st.text_input('', key = 'widget', on_change=submit)  

print('=========== 2 ============')


# def create_retriever(): 
#     print(f'===== call create_retroiever with urls: {st.session_state.urls}')
#     docs = [WebBaseLoader(url).load() for url in urls]
#     # print(docs)
#     # print('here')
#     docs_list = [item for sublist in docs for item in sublist]

#     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=250, chunk_overlap=0
#     )
#     doc_splits = text_splitter.split_documents(docs_list)

#     # Add to vectorDB
#     vectorstore = Chroma.from_documents(
#         documents=doc_splits,
#         collection_name="rag-chroma",
#         embedding=OpenAIEmbeddings(),
#     )
#     retriever = vectorstore.as_retriever()
#     return retriever           
        

 
# def get_response(user_query):
#         return "I don't know"  



class AgentState(TypedDict):
    question: str
    answer:str
    outcome: str
    documents: List[str]
    chat_history: Annotated[Sequence[BaseMessage], operator.add]
    retriever: Any



if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content='Hello, i am a bot, how can i help you') 
        ]

print('--- 444444 --')


# this is the first node
def retrieve_docs(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("===in retrieve docs based on question ===")
    print('\n')
    # state_dict = state["keys"]
    # question = state_dict["question"]
    question = state['question']
    print('\n')
    print(f"===question: {question}")
    state['chat_history'].append(HumanMessage(content=question))

    # docs = [WebBaseLoader(url).load() for url in urls]
    # # print(docs)
    # # print('here')
    # docs_list = [item for sublist in docs for item in sublist]

    # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=250, chunk_overlap=0
    # )
    # doc_splits = text_splitter.split_documents(docs_list)

    # # Add to vectorDB
    # vectorstore = Chroma.from_documents(
    #     documents=doc_splits,
    #     collection_name="rag-chroma",
    #     embedding=OpenAIEmbeddings(),
    # )
    # retriever = vectorstore.as_retriever()


    # retriever = state['retriever']
    # print(f'====in node one: retriever is {retriever}')
    documents = retriever.get_relevant_documents(question)
    # documents = retriever.get_relevant_documents(question)
    print('\n')
    print(f"===documents: {documents}")
    length = len(documents)
    print(f"===document length: {length}")
    print('\n')
    # return {"keys": {"documents": documents, "question": question}}
    # return {"documents":  documents, "question": question}
    return {
        'documents': documents,
        'question': question
    }





def evaluate_docs_from_question(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with relevant documents
    """

    print("---CHECK RELEVANCE---")
    # state_dict = state["keys"]
    # question = state_dict["question"]
    question = state['question']
    documents = state["documents"]

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")
 # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    # Tool
    grade_tool_oai = convert_to_openai_tool(grade)

    # LLM with tool and enforce invocation
    llm_with_tool = model.bind(
        tools=[grade_tool_oai],
      
        tool_choice={"type": "function", "function": {"name": "grade"}},
    )
    
    # Parser
    parser_tool = PydanticToolsParser(tools=[grade])

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    # Chain
    chain = prompt | llm_with_tool | parser_tool

    # Score
    filtered_docs = []
    for d in documents:
        score = chain.invoke({"question": question, "context": d.page_content})
        grade = score[0].binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    return {"documents": filtered_docs, "question": question}
    # return {"keys": {"documents":  filtered_docs, "question": question}}





def generate_answer_from_question_and_docs(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("=== enter generate answer based on docs ===")
    # state_dict = state["keys"]
    question = state["question"]
    documents = state["documents"]

    print('\n')
    print(f"===pass in question: {question}")
    print('\n')

    print('\n')
    length = len(documents)
    print(f"===pass in docs length: {length}")
    print('\n')


    # Prompt
    prompt = hub.pull("rlm/rag-prompt")
    print('--rag-prompt')
    # print(prompt)

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    # rag_chain = prompt | print_and_return_string |llm | StrOutputParser()
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generated_answer = rag_chain.invoke({"context": documents, "question": question})

    print('\n')
    print(f"===  answer is: {generated_answer}")
    print('\n')

    return {"documents": documents, "question": question, "answer": generated_answer}
    # return {"keys": {"documents":  documents, "question": question,"answer": generated_answer}}
    





# improve the question, generate a new question to ask
def generate_better_question_to_ask(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("=== improve question from old question ===")
    # state_dict = state["keys"]
    question = state["question"]
    documents = state["documents"]

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval. \n 
        Look at the input and try to reason about the underlying sematic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
        input_variables=["question"],
    )

    # Grader
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

    # Prompt
    chain = prompt | model | StrOutputParser()
    better_question = chain.invoke({"question": question})
    print('\n')
    print(f"=== old question: {question} ")
    print('\n')
    print(f"=== improved question: {better_question} ")
    print('\n')

    return {"documents": documents, "question": better_question}




# this do nothing, just pack data
# don't understand why need this step
def pack_data_for_last_step(state):
    """
    Passthrough state for final grade.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): The current graph state
    """

    print("=== do nothing, just pack and pass ===")
    # state_dict = state["keys"]
    question = state["question"]
    documents = state["documents"]
    answer = state["answer"]
    print('\n')
    print(f"=== documents: {documents} ")
    print('\n')
    print(f"=== question: {question} ")
    print('\n')
    print(f"=== answer: {answer} ")
    print('\n')


    return {"documents": documents, "question": question, "answer": answer}




# evaluate docs to question, this will process docs
def evaluate_docs_from_question(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with relevant documents
    """

    print("===grade all documents based on question and collect all good one ===")
    # state_dict = state["keys"]
    question = state["question"]
    documents = state["documents"]

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        print('\n')
        print("=== grade tool called ==")
        print('\n')

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")


    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

    # Tool
    grade_tool_oai = convert_to_openai_tool(grade)

    # LLM with tool and enforce invocation
    llm_with_tool = model.bind(
        tools=[convert_to_openai_tool(grade_tool_oai)],
        tool_choice={"type": "function", "function": {"name": "grade"}},
    )

    # Parser
    parser_tool = PydanticToolsParser(tools=[grade])

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )


    # Chain
    chain = prompt | llm_with_tool | parser_tool

    # Score
    filtered_docs = []
    for d in documents:
        score = chain.invoke({"question": question, "context": d.page_content})
        # print('\n')
        # print(f"===inside loop, question is: {question}")
        print('\n')
        print(f"===inside loop, current page_content is: {d.page_content} ")
        print('\n')
        # print(f"===score is: {score} ")
        print(f"===inside loop, current grade is: {score[0].binary_score} ")
        # print(f"===score[0].binary_score is: {score[0].binary_score} ")
        print('\n')

        grade = score[0].binary_score
        if grade == "yes":
            print("--- collect this---")
            # this single doc is good for question, collect it
            filtered_docs.append(d)
        else:
            print("---not collect this---")
            # this single doc is not good for the question, remove it
            continue


            print('\n')
    temp = len(filtered_docs)
    print(f"===filtered_docs: {temp} ")
    print('\n')
    print(f"=== total selected docs: {filtered_docs} ")
    print('\n')

    return {"documents": filtered_docs, "question": question}




# this will check if the processed docs is empty or not
# if docs is empty, we need to go back to repeat my work again
# if it is not empty, it works go to next step
def check_if_docs_empty(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """

    print("---logic here: improve question OR generate answer? ---")
    # state_dict = state["keys"]
    question = state["question"]
    filtered_documents = state["documents"]
    doc_length = len(filtered_documents)

    print('\n')
    print(f"===  pass in filtered_documents size is: {doc_length} ")
    print('\n')


    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        # print("---DECISION: TRANSFORM QUERY---")
        print('\n')
        l1 = len(filtered_documents)
        print(f"=== current filtered_documents size: {l1} we need to improve question ")
        print('\n')
        print('\n')
        print("=== need to improve question")
        print('\n')
        # docs is empty, need to improve question and ask again'
        return "generate_better_question_to_ask"
    else:
        # We have relevant documents, so generate answer
        # print("---DECISION: GENERATE---")
        print('\n')
        l2 = len(filtered_documents)
        print(f"=== current filtered_documents size is: {l2}, good enough, we can answer now ")
        print('\n')
        print('\n')
        print("=== go answer_node ")
        print('\n')
        # docs is not empty, good enough, go to next step to prepare answer
        return "generate_answer_from_question_and_docs"




# this logic will evaluate generated answer to docs
def evaluate_answer_to_docs(state):
    """
    Determines whether the generation is grounded in the document.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Binary decision
    """

    print("---GRADE GENERATION vs DOCUMENTS---")
    # state_dict = state["keys"]
    question = state["question"]
    documents = state["documents"]
    answer = state["answer"]

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Supported score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)




  # Tool
    grade_tool_oai = convert_to_openai_tool(grade)

    # LLM with tool and enforce invocation
    llm_with_tool = model.bind(
        tools=[convert_to_openai_tool(grade_tool_oai)],
        tool_choice={"type": "function", "function": {"name": "grade"}},
    )

    # Parser
    parser_tool = PydanticToolsParser(tools=[grade])



     # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {answer}
        Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in / supported by a set of facts.""",
        input_variables=["generation", "documents"],
    )


    # Chain
    chain = prompt | llm_with_tool | parser_tool

    print('---some problem in grade answer and documents')
    print('\n')
    print(f"===  generation: {answer} ")
    print('\n')
    print('\n')
    print(f"===  documents: {documents} ")
    print('\n')
    

    score = chain.invoke({"answer": answer, "documents": documents})
    grade = score[0].binary_score

    print('\n')
    print(f"===  we get the score: {score} ")
    print('\n')
    print('\n')
    print(f"===  grade: {grade} ")
    print('\n')



    if grade == "yes":
        print("--- grade answer based on ducuments, get 'yes', go to 'prepare_for_final_grade' fork---")
        # return "supported"
        # when generated answer is good enought based on docs, go to next step
        return "pack_data_for_last_step"
    else:
        print("--- grade answer based on ducuments, get 'no', go to 'answer_node' fork---")
        
        print("---DECISION: NOT SUPPORTED, answer_node AGAIN---")
        # return "not supported"
        # answer is not good, need to go back
        return "generate_answer_from_question_and_docs"



# evaluate generated answer to original question
def evaluate_answer_to_question(state):
    """
    Determines whether the generation addresses the question.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Binary decision
    """

    print("---GRADE GENERATION vs QUESTION---")
    # state_dict = state["keys"]
    question = state["question"]
    documents = state["documents"]
    answer = state["answer"]

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Useful score 'yes' or 'no'")



    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

    # Tool
    grade_tool_oai = convert_to_openai_tool(grade)

    # LLM with tool and enforce invocation
    llm_with_tool = model.bind(
        tools=[convert_to_openai_tool(grade_tool_oai)],
        tool_choice={"type": "function", "function": {"name": "grade"}},
    )

    # Parser
    parser_tool = PydanticToolsParser(tools=[grade])


    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
        Here is the answer:
        \n ------- \n
        {answer} 
        \n ------- \n
        Here is the question: {question}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question.""",
        input_variables=["answer", "question"],
    )




     # Prompt
    chain = prompt | llm_with_tool | parser_tool

    score = chain.invoke({"answer": answer, "question": question})
    grade = score[0].binary_score



    if grade == "yes":
        print("----- grade is 'yes', go to 'END'")
        # print("---DECISION: USEFUL---")
        # return "useful"
        # answer and question match well, can and processing
        answer = state['answer']
        state['chat_history'].append(AIMessage(content=answer))
        return "END"
        
    else:
        # print("---DECISION: NOT USEFUL---")
        # return "not useful"
        print("----- grade is 'no'go to 'transform_query' ")
        # answer to question is not good, need to go back 
        # looks like need to improve question, generate a new question
        return "generate_better_question_to_ask"



# workflow = StateGraph(GraphState)
workflow = StateGraph(AgentState)




# print('------ step 1-----')
# key and def
workflow.add_node("retrieve_docs", retrieve_docs)  # retrieve


# process docs based on asked question
workflow.add_node("evaluate_docs_from_question", evaluate_docs_from_question)  # grade documents




# print('------ step 2 -----')




# imporve question
workflow.add_node("generate_better_question_to_ask", generate_better_question_to_ask)  # transform_query


# fork 1, success, go to next step to answer question
workflow.add_node("generate_answer_from_question_and_docs", generate_answer_from_question_and_docs)  # generatae



workflow.add_node("pack_data_for_last_step", pack_data_for_last_step)  # passthrough

# ----------------------------------------------

# entry point
workflow.set_entry_point("retrieve_docs")

# wire 2 nodes
workflow.add_edge("retrieve_docs", "evaluate_docs_from_question")
# wire 2 node
workflow.add_edge("generate_better_question_to_ask", "retrieve_docs")



# logic here, check if docs is empty or not
workflow.add_conditional_edges(
    "evaluate_docs_from_question",
    check_if_docs_empty,
    {
        # docs is empty, need to improve question
        "generate_better_question_to_ask": "generate_better_question_to_ask",
        # docs is not empty, can go ahead for next processing
        "generate_answer_from_question_and_docs": "generate_answer_from_question_and_docs",
    },
)

# logic here, check if answer is good enough to docs
workflow.add_conditional_edges(
    "generate_answer_from_question_and_docs",
    evaluate_answer_to_docs,
    {
        # good enough, to ahead for further procesing
        "pack_data_for_last_step": "pack_data_for_last_step",
        # no good, repeat
        "generate_answer_from_question_and_docs": "generate_answer_from_question_and_docs",
    },
)




workflow.add_conditional_edges(
    "pack_data_for_last_step",
    evaluate_answer_to_question,
    {
        # if good enough, complete everything,
        "END":END,
        # not good enough, go to improve question
        "generate_better_question_to_ask": "generate_better_question_to_ask",
    },
)




# Compile
app = workflow.compile()




# Run
# inputs = {"keys": {"question": "紫陌红尘拂面来，无人不道看花回, 出自什么诗名"}}
# inputs = {"question": "紫陌红尘拂面来，无人不道看花回, 出自什么诗名"}
# app.invoke(inputs)

# for output in app.stream(inputs):
    # for key, value in output.items():
        # print('')
        # Node
        # pprint.pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    # pprint.pprint("\n---\n")
    # pprint.pprint("----------")   

# Final generation
# pprint.pprint(value['answer'])







# print('--- 66666--')
   



def get_response(user_query):
    print(f' *******in get_response scope new show me session_state: {st.session_state}')
    # inputs = {"question": "紫陌红尘拂面来，无人不道看花回, 出自什么诗名"}
    # app.invoke(inputs)
    # {"keys": {"question": "紫陌红尘拂面来，无人不道看花回, 出自什么诗名"}}
    # retriever = create_retriever()
    inputs = {
        'question': user_query,
        'chat_history': st.session_state.chat_history,
        # 'retriever': retriever
        }
    app.invoke(inputs)
      
    





if not st.session_state.urls:
     print('=========== 4 ============')
     print('=========== urls is empty ============')
     st.info('please input your question')
else:     
    print('=========== 5 ============')
    print('=========== urls is not empty ============')
    user_query = st.chat_input('Please input your question ...')    

    

    if user_query is not None and user_query != '' :
        print('=========== 6 ============')
        print('== get query ==')

        
        # st.session_state.chat_history.append(
        #     HumanMessage(content=user_query)
        # )

        # response = get_response(user_query)
        # st.session_state.chat_history.append(
        #     AIMessage(content=response)
        # )

        # do it this way
        # if('retriever' not in st.session_state):
        #     st.session_state.retriever = create_retriever()
        #     print(f'===== finally we can get retriever == {st.session_state.retriever}====')


        
        # inputs = {
        #     'question': user_query,
        #     'chat_history': st.session_state.chat_history
        # }

        # langgraph code will be packed inside here
        retriever = create_retriever(st.session_state.urls)
        response = get_response(user_query)
    


if 'chat_history' not in st.session_state:
     print('=========== 3 ============')
     st.session_state.chat_history = [
          AIMessage(content='Hello, i am a bot, how can i help you')
     ]



            
for message in st.session_state.chat_history:
        print('=========== 7 ============')
        if isinstance(message, AIMessage):
            with st.chat_message('AI'):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('Human'):
                st.write(message.content)         







# if st.session_state.website_url is None or st.session_state.website_url == '':
#      st.info('please enter a website URL to make it works')
#      print('--- 777777 --')

# else:     
#     print('--- 88888 --')
#     print('***********************in else scoipe before init chat_history ************')

#     # with st.sidebar:
#         # st.write(website_url)
#         # st.text_input.
#         # website_url = st.text_input('website url')


#     # if 'chat_history' not in st.session_state:
#     #     st.session_state.chat_history = [AIMessage(content='Hello, i am a bot, how can i help you') ]
#     print(f'*********in else scope after init chat_history: {st.session_state} *******')

#     user_query = st.chat_input('type your question here ...')    

    # if 'chat_history' not in st.session_state:
    #     st.session_state.chat_history = [ AIMessage(content='Hello, i am a bot, how can i help you')]
        # st.session_state.chat_history = [
        #     AIMessage(content='Hello, i am a bot, how can i help you')
        # ]

    # def get_response(user_query):
    #     return "I don't know"  


    # if user_query is not None and user_query != '' :

    #     print('--- 999999 --')
    #     print(f'== in else-if scope show me current session_state: {st.session_state}')


        
        # inputs = {
        #     'question': user_query,
        #     'chat_history': st.session_state.chat_history
        # }

        # langgraph code will be packed inside here
        # response = get_response(user_query)

        # st.session_state.chat_history.append(
        #     HumanMessage(content=user_query)
        # )

        # st.session_state.chat_history.append(
        #     AIMessage(content=response)
        # )

    # do it here
    # st.session_state.chat_history = chat_history
    # for message in st.session_state.chat_history:
    #     print('--- messages loop here --')

    #     if isinstance(message, AIMessage):
    #         with st.chat_message('AI'):
    #             st.write(message.content)
    #     elif isinstance(message, HumanMessage):
    #         with st.chat_message('Human'):
    #             st.write(message.content)                  
                




# ==========================================

# print(os.getenv("OPENAI_API_KEY"))
# print(os.getenv("TAVILY_API_KEY"))
print('go through')

# print('\n')
# print(os.getenv("OPENAI_API_KEY"))
# print(os.getenv("LANGCHAIN_API_KEY"))
# print(os.getenv("TAVILY_API_KEY"))
# print('\n')
# print(os.getenv("LANGCHAIN_TRACING_V2"))
# print(os.getenv("LANGCHAIN_PROJECT"))
# print(os.getenv("LANGCHAIN_ENDPOINT"))


    



    

