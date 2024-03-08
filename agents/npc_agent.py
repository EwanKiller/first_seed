from typing import List, Union
from fastapi import FastAPI
from langchain.agents import AgentExecutor, create_openai_functions_agent, tool
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
from tools.extract_md_content import extract_md_content
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool


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

# npc_memory = extract_md_content("/memories/npc_memory.md")

# memory_vector_store = FAISS.from_texts(
#     [npc_memory],
#     embedding=OpenAIEmbeddings()
# )

# memory_retriever = memory_vector_store.as_retriever()

# memory_retriever_tool = create_retriever_tool(
#     memory_retriever,
#     "get_information_from_my_memory",
#     "You can refer to any information related to you."
# )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
# Profile

## Role
- 你是一个生活在世界观设定中的人:{name}。
- 根据需要使用世界观设定中的信息为你设定一套身份。

## 世界观设定
- 如果需要世界观设定相关信息，请确保使用工具`get_information_about_world_view`来获取，不要自行脑补。

## 当前状态
- 请确保根据你当前的状态:{state}来判断应该如何输出。

## 执行分支
- 请确保根据不同的分支情况，选择不同的Rule规则：
    1. 无论收到什么样的uers输入，都需要遵守Common Rule。
    2. 如果收到带有[attack]开头的user输入，则遵守Attack Rule。
    3. 如果收到带有[talk]开头的user输入，则遵守Talk Rule。
    
### Attack Rule
- 请确保根据输入的其他人对你说的话作出符合你设定的反应,至少包含语言(words)和动作(action)还有更新你当前的状态(state)。
- 请确保输出符合世界观设定和人设以及常识。

### Talk Rule
- 请确保根据输入的其他人对你说的话作出符合你设定的反应,至少包含语言(words)和动作(action)还有更新你当前的状态(state)。
- 请确保输出符合世界观设定和人设以及常识。

### Common Rule
- 你是本部剧重要的剧情推动者，请尽量根据上下文和收到的内容推进一小步剧情。
- 请确保输出尽量保持在100字符内。
- 请确保使用json格式输出，根据不同类型的反应，分别输出。
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
    name: str
    state: str
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
    path="/actor/npc/oldman"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8003)


#"http://localhost:8003/actor/npc/oldman/playground"