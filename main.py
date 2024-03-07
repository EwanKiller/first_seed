from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langserve import RemoteRunnable
import sys, json

class BaseAgent():
    def __init__(self, name):
        self.name = name

    def connect(self, url):
        self.agent = RemoteRunnable(url)

class TaskAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)

class StoryAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)

class NpcAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)

class PlayerAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)

class ActorOutput:
    def __init__(self, words: str, action: str):
        self.words = words
        self.action = action
        

def parse_json(content: str) -> ActorOutput:
    if "json" in content:
        content = content.replace('```json', '').replace('```', '')
    else:
        end_index = content.find('}')
        content = content[:end_index + 1]
    data = json.loads(content)

    words = data.get("words", "")
    action = data.get("action", "")

    return ActorOutput(words=words, action=action)
    

def main():

    task = TaskAgent("Task")
    story = StoryAgent("Story")
    npc = NpcAgent("NPC")
    player = PlayerAgent("Player")

    task.connect("http://localhost:8001/system/task/")
    story.connect("http://localhost:8002/system/story/")
    npc.connect("http://localhost:8003/actor/npc/oldman")
    player.connect("http://localhost:8004/actor/player/")

    story_history = []
    npc_history = []
    player_history = []
    story_input = ""
    npc_input = ""
    player_input = ""

    brief_task = input("请输入任务简述:")
    if brief_task == "":
        brief_task = "Ewan和他的伙伴帕斯卡开启了一段神奇而有趣的冒险之旅。"
    print(brief_task)

    task_response = task.agent.invoke({"brief_task":brief_task 
                                       , "input": "输出一个任务。", "chat_history":[]})
    print("[任务系统]" + task_response['output'] + "\n==============================================")
    # NPC收到任务作出反应
    npc_response = npc.agent.invoke({"input": task_response['output']
                                     , "chat_history": npc_history})
    npc_response_json = parse_json(npc_response['output'])
    npc_history.extend([HumanMessage(content=task_response['output']),AIMessage(content=npc_response['output'])])
    print(f"[帕斯卡]{npc_response_json.words}\n({npc_response_json.action})" + "\n==============================================")    
    # player_history.append(HumanMessage(content=npc_response['output']))

    story_input = f"[任务系统]{task_response['output']}\n[帕斯卡]{npc_response_json.words}\n({npc_response_json.action})"
    story_response = story.agent.invoke({"input": story_input, "chat_history": story_history})
    story_history.extend([HumanMessage(content=story_input),AIMessage(content=story_response['output'])])
    print(f"[故事]{story_response['output']}" + "\n==============================================")

    
    while True:
        user_input = input(f"[Player]:")
        print("==============================================")

        player_response = player.agent.invoke({"input": "[Ewan]帕斯卡," + user_input, "chat_history": player_history})
        player_response_json = parse_json(player_response['output'])

        print(f"[Ewan]{player_response_json.words}\n({player_response_json.action})" + "\n==============================================")

        npc_response = npc.agent.invoke({"input":f"[Ewan]{player_response_json.words}\n({player_response_json.action})"
                                        , "chat_history": npc_history})
        npc_response_json = parse_json(npc_response['output'])
        npc_history.extend([HumanMessage(content=f"[Ewan]{player_response_json.words}\n({player_response_json.action})")
                            , AIMessage(content=npc_response['output'])])

        player_history.extend([AIMessage(content=player_response['output']),HumanMessage(content=npc_response['output'])])

        print(f"[帕斯卡]{npc_response_json.words}\n({npc_response_json.action})" + "\n==============================================")

        story_input = f"[Ewan]{player_response_json.words}\n({player_response_json.action})\n[帕斯卡]{npc_response_json.words}\n({npc_response_json.action})"
        story_response = story.agent.invoke({"input": story_input, "chat_history": story_history})
        story_history.extend([HumanMessage(content=story_input),AIMessage(content=story_response['output'])])

        print(f"[故事]{story_response['output']}" + "\n==============================================")

if __name__ == "__main__":
    print("==============================================")
    main()