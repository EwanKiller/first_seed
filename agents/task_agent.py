from typing import List, Union
from fastapi import FastAPI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
from tools.extract_md_content import extract_md_content
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool


user_input = "一位年轻的勇者偶遇一位曾经的冒险家老人,绞尽脑汁想要从老人手中获得他的神秘地图。"

world_view = extract_md_content("/story/world_view.md")

vector_store = FAISS.from_texts(
    [world_view],
    embedding=OpenAIEmbeddings()
)

retriever = vector_store.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "get_information_about_world_view",
    "You must refer these information before response user."
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
# Profile

## Role
- 你是一个角色扮演小说中的故事主线设计者。

## Rule
- 请确保根据{brief_task}来发散书写一个
- 请尽量控制在100个字符以内。

## World View
- 如果需要世界观设定相关信息，请确保使用工具`get_information_about_world_view`来获取，不要自行脑补。
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm = ChatOpenAI(model="gpt-4-turbo-preview")
    
tools = [retriever_tool]

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

app = FastAPI(
    title="Chat module",
    version="1.0",
    description="Gen chat",
)

class Input(BaseModel):
    brief_task: str
    input: str
    chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "input", "output": "output"}},
    )

class Output(BaseModel):
    output: str

add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/system/task"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)

#http://localhost:8001/system/task/playground/