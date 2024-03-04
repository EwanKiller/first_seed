from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langserve import RemoteRunnable
import sys

#希望这个方法仅表达talk的行为
def talk_to_agent(input_val, npc_agent, chat_history):
    response = npc_agent.invoke({"input": input_val, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=input_val), AIMessage(content=response['output'])])
    return response['output']

#希望这个方法仅表达talk的行为
def parse_talk(input_val):
    if "/talk" in input_val:
        return input_val.split("/talk")[1].strip()
    return input_val

def main():

    #
    npc_agent = RemoteRunnable("http://localhost:8001/actor/npc/elder/")
    npc_achat_history = []

    #
    scene_agent = RemoteRunnable("http://localhost:8002/actor/npc/house/")
    scene_achat_history = []

    # 数值计算系统
    numerical_system_agent = RemoteRunnable("http://localhost:8007/system/evaluate")

    #
    scene_state = talk_to_agent(
            f"""
            # 状态
            - 冬天的晚上，我（长者）坐在你的壁炉旁
            # 事件
            - 我在沉思，可能是回忆过往，并向壁炉中的火投入了一根木柴
            # 需求
            - 在上述“状态”下发生了“事件”（“事件”可以改变”状态“），请根据“对话规则”输出文本
            """, 
            scene_agent, scene_achat_history)
    #
    print("[scene]:", scene_state)

    #
    event = "我(勇者)用力推开了屋子的门，闯入屋子而且面色凝重，外面的寒风吹进了屋子"
    print("[event]:", event)

    #
    print("[npc]:", talk_to_agent(
            f"""
            # 状态
            -{scene_state}
            # 事件
            -{event}
            # 需求
            - 在上述“状态”下发生了“事件”（“事件”可以改变”状态“），请根据“对话规则”输出文本
            """, 
            npc_agent, npc_achat_history))

    while True:
        usr_input = input("[user input]: ")
        print("==============================================")
        if "/quit" in usr_input:
            sys.exit()

        elif "/talk" in usr_input:
            real_input = parse_talk(usr_input)
            print("[you]:", real_input)
            print(
                '[npc]:', talk_to_agent(real_input, npc_agent, npc_achat_history)
            )

        else:
            real_input = parse_talk(usr_input)
            print("[default]:", real_input)
            print(
                '[npc]:', talk_to_agent(real_input, npc_agent, npc_achat_history)
            )


if __name__ == "__main__":
    print("==============================================")
    main()