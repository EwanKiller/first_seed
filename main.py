from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langserve import RemoteRunnable
import sys, json

class ActorJsonFormatOutput:
    def __init__(self, words: str, action: str, state: str):
        self.words = words
        self.action = action
        self.state = state

class InteractType:
    talk = "[talk]"
    attack = "[attack]"


def parse_actor_json_format_output(content: str) -> ActorJsonFormatOutput:
    """解析LLM输出的json格式字符串"""
    if "json" in content:
        content = content.replace('```json', '').replace('```', '')
    else:
        end_index = content.find('}')
        content = content[:end_index + 1]
    data = json.loads(content)

    words = data.get("words", "")
    action = data.get("action", "")
    state = data.get("state", "")

    return ActorJsonFormatOutput(words=words, action=action, state=state)

class BaseAgent():
    history: list

    def __init__(self, name:str):
        self.name = name
        self.history = []

    def connect(self, url):
        self.agent = RemoteRunnable(url)

class TaskAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)

class StoryAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)

class ActorAgent(BaseAgent):
    interacting_agents: list[BaseAgent]
    current_state: str

    def __init__(self, name: str):
        super().__init__(name)
        self.interacting_agents = []
        self.current_state = ""

    def call(self, starter:str, state: str, type:str ,input: str) -> ActorJsonFormatOutput:
        response = self.agent.invoke({"name": self.name, "state": state, "input": type + input, "chat_history": self.history})
        self.history.extend([HumanMessage(content=type + input),AIMessage(response['output'])])
        output = parse_actor_json_format_output(response['output'])
        self.current_state = output.state
        print(f"!<---{starter}对{self.name}进行了{type}类型操作,内容是:{input}--->")
        print(f"{self.name}回应{starter}说{output.words}\n({output.action})\n!<---{self.name}当前状态:{output.state}--->\n")
        return output

class NpcAgent(ActorAgent):

    def __init__(self, name):
        super().__init__(name)

class PlayerAgent(ActorAgent):

    def __init__(self, name):
        super().__init__(name)
        

def parse_actor_json_format_output(content: str) -> ActorJsonFormatOutput:
    """解析LLM输出的json格式字符串"""
    if "json" in content:
        content = content.replace('```json', '').replace('```', '')
    else:
        end_index = content.find('}')
        content = content[:end_index + 1]
    data = json.loads(content)

    words = data.get("words", "")
    action = data.get("action", "")
    state = data.get("state", "")

    return ActorJsonFormatOutput(words=words, action=action, state=state)
    

def main():
    # 创建代理对象
    task = TaskAgent("Task")
    story = StoryAgent("Story")
    npc = NpcAgent("帕斯卡")
    player = PlayerAgent("Ewan")
    # 绑定AgentServe
    task.connect("http://localhost:8001/system/task/")
    story.connect("http://localhost:8002/system/story/")
    npc.connect("http://localhost:8003/actor/npc/oldman")
    player.connect("http://localhost:8004/actor/player/")
    # 设置通信对象
    player.interacting_agents.append(npc)
    npc.interacting_agents.append(player)
    # 设置状态
    player.current_state = "健康"
    npc.current_state = "健康"

    while True:
        input_type = input("input type【/attack or /talk】:")
        if "attack" in input_type:
            input_type = InteractType.attack
        elif "talk" in input_type:
            input_type = InteractType.talk

        user_set_input = input(f"{player.name}:")
        # Player round
        for agent in player.interacting_agents:
            if isinstance(agent, ActorAgent):
                npc_response = agent.call(player.name, agent.current_state, input_type, user_set_input)
    

    # brief_task = input("请输入任务简述:")
    # if brief_task == "":
    #     brief_task = "Ewan和他的伙伴帕斯卡开启了一段神奇而有趣的冒险之旅。"
    # print(brief_task)

    # task_response = task.agent.invoke({"brief_task":brief_task 
    #                                    , "input": "输出一个任务。", "chat_history":[]})
    # print("[任务系统]" + task_response['output'] + "\n<!------------------------------------------>")
    # # NPC收到任务作出反应
    # npc_response = npc.agent.invoke({"input": task_response['output']
    #                                  , "chat_history": npc_history})
    # npc_response_json = parse_json(npc_response['output'])
    # npc_history.extend([HumanMessage(content=task_response['output']),AIMessage(content=npc_response['output'])])
    # print(f"[帕斯卡]{npc_response_json.words}\n({npc_response_json.action})" + "\n<!------------------------------------------>")    
    # # player_history.append(HumanMessage(content=npc_response['output']))

    # story_input = f"[任务系统]{task_response['output']}\n[帕斯卡]{npc_response_json.words}\n({npc_response_json.action})"
    # story_response = story.agent.invoke({"input": story_input, "chat_history": story_history})
    # story_history.extend([HumanMessage(content=story_input),AIMessage(content=story_response['output'])])
    # print(f"[故事]{story_response['output']}" + "\n<!------------------------------------------>")

    
    # while True:
    #     print("<!---------------------")
    #     user_input = input(f"[Player]:")
    #     print("--------------------->")

    #     player_response = player.agent.invoke({"input": user_input, "chat_history": player_history})
    #     player_response_json = parse_json(player_response['output'])

    #     print(f"[Ewan]{player_response_json.words}\n({player_response_json.action})" + "\n<!------------------------------------------>")

    #     npc_response = npc.agent.invoke({"input":f"[Ewan]{player_response_json.words}\n({player_response_json.action})"
    #                                     , "chat_history": npc_history})
    #     npc_response_json = parse_json(npc_response['output'])
    #     npc_history.extend([HumanMessage(content=f"[Ewan]{player_response_json.words}\n({player_response_json.action})")
    #                         , AIMessage(content=npc_response['output'])])

    #     player_history.extend([AIMessage(content=player_response['output']),HumanMessage(content=npc_response['output'])])

    #     print(f"[帕斯卡]{npc_response_json.words}\n({npc_response_json.action})" + "\n<!------------------------------------------>")

    #     story_input = f"[Ewan]{player_response_json.words}\n({player_response_json.action})\n[帕斯卡]{npc_response_json.words}\n({npc_response_json.action})"
    #     story_response = story.agent.invoke({"input": story_input, "chat_history": story_history})
    #     story_history.extend([HumanMessage(content=story_input),AIMessage(content=story_response['output'])])

    #     print(f"[故事]{story_response['output']}" + "\n<!------------------------------------------>")

if __name__ == "__main__":
    main()