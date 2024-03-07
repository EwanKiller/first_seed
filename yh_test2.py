from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langserve import RemoteRunnable
import sys
import re

class World:
    def __init__(self, name):
        self.name = name
        self.stages = []

    def connect(self, url):
        self.agent = RemoteRunnable(url)
        self.chat_history = []

    def add_stage(self, stage):
        self.stages.append(stage)

#
class Stage:

    def __init__(self, name):
        self.name = name
        self.actors = []

    def connect(self, url):
        self.agent = RemoteRunnable(url)
        self.chat_history = []

    def add_actor(self, actor):
        self.actors.append(actor)

#
class Actor:
    def __init__(self, name):
        self.name = name
        #An assessment of the combat effectiveness of an army
        self.health = 10
        self.power = 10
     

    def connect(self, url):
        self.agent = RemoteRunnable(url)
        self.chat_history = []

#
class Player(Actor):
    def __init__(self, name):
        super().__init__(name)
        self.items = []

#
class NPC(Actor): 
    def __init__(self, name):
        super().__init__(name)

#
class Item(Actor): 
    def __init__(self, name):
        super().__init__(name)

#
def call_agent(target, prompt):
    if not hasattr(target, 'agent') or not hasattr(target, 'chat_history'):
        return None
    response = target.agent.invoke({"input": prompt, "chat_history": target.chat_history})
    target.chat_history.extend([HumanMessage(content=prompt), AIMessage(content=response['output'])])
    return response['output']

#
def parse_input(input_val, split_str):
    if split_str in input_val:
        return input_val.split(split_str)[1].strip()
    return input_val


######################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
#
init_archivist = f"""
# 游戏世界存档
- 大陆纪元2000年1月1日，冬夜.
- “卡斯帕·艾伦德”坐在他的“老猎人隐居的小木屋”中的壁炉旁，在沉思和回忆过往，并向壁炉中的火投入了一根木柴。
- 他的小狗（名叫"断剑"）在屋子里的一角睡觉。
"""

load_prompt = f"""
# 你需要读取存档
## 步骤:
- 第1步，读取{init_archivist}.
- 第2步：理解这些信息(尤其是和你有关的信息).
- 第3步：根据信息更新你的最新状态与逻辑.
- 第4部：如果是你角色输则出你在做什么，如果你是场景则输出场景中正在发生的一切.
## 输出规则：
- 保留关键信息(时间，地点，人物，事件)，不要推断，增加与润色。输出在保证语意完整基础上字符尽量少。
"""

#场景需要根据状态做出计划
def stage_plan_prompt(stage):
    prompt = f"""
    # 你需要做出计划
    - 如果你的场景设定中，允许你做出计划，那么你需要做出计划。否则，仅更新你的状态即可。   

    ## 步骤
    - 第1步：理解你自身当前状态。
    - 第2步：理解你的场景内所有角色的当前状态。
    - 第3步：根据以2步，输出你需要做出计划。

   ## 注意！输出的关键字，只能在如下中选择：
    - [fight]，代表着你计划攻击某个目标（角色）。
    - [stay]，代表着你保持现状，不做任何事。

    ## 输出规则与示例：
    - 如果你想攻击某个目标，那么你的输出格式为：“[fight][目标的名字]:...“，...代表着你本次攻击要说的话与心里活动。
    - 如果你想保持现状，那么你的输出格式为：“[stay]：...“，...代表着你本次保持现状要说的话与心里活动。
    - 如果你想说话，那么你的输出格式为：“[stay]:...“，...代表着你本次要说的话与心里活动。
    - 如果不在以上3种情况，就输出"[stay]:...", ...仅代表着你的心里活动。

    ## 输出规则：
    - 不要推断，增加与润色。
    - 输出在保证语意完整基础上字符尽量少。
    """
    return prompt

#
class FightEvent:
    def __init__(self, stage, src_actor_name, dest_actor_name, say_content):
        self.stage = stage
        self.src_actor_name = src_actor_name
        self.dest_actor_name = dest_actor_name
        self.say_content = say_content
        self.init()

    def init(self):
        for actor in self.stage.actors:
            if actor.name == self.src_actor_name:
                self.src_actor = actor
            elif actor.name == self.dest_actor_name:
                self.dest_actor = actor

    def __str__(self):
        return f"{self.src_actor_name}=>{self.dest_actor_name}:{self.say_content}"
    
    def make_plan(self):
        self.src_actor.health -= 3
        self.dest_actor.health = -3
        if self.src_actor.health <= 0:
            return f"""{self.src_actor_name}准备向{self.dest_actor_name}发起攻击,
            他（她/它）说到（或者内心的想法）：{self.say_content}。
            结果：{self.src_actor_name}将会死亡。"""
        elif self.dest_actor.health <= 0:
            return f"""{self.src_actor_name}准备向{self.dest_actor_name}发起攻击,
            他（她/它）说到（或者内心的想法）：{self.say_content}。
            结果：{self.dest_actor_name}将会死亡。"""
        return f"""{self.src_actor_name}准备向{self.dest_actor_name}发起攻击,他（她/它）说到（或者内心的想法）：{self.say_content}"""
    
class StayEvent:
    def __init__(self, stage, actor_name, say_content):
        self.stage = stage
        self.actor_name = actor_name
        self.say_content = say_content
        self.init()

    def init(self):
        for actor in self.stage.actors:
            if actor.name == self.actor_name:
                self.actor = actor

    def __str__(self):
        return f"{self.actor_name}:{self.say_content}"
    
    def make_plan(self):
        return f"""{self.actor_name}准备保持现状,他（她/它）说到（或者内心的想法）：{self.say_content}"""

##
def director_prompt(stage, plans_group):
    return f"""
    # 你需要根据所有角色（可能包括你自己）的计划，做出最终的决定，推演与执行。

    ## 所有角色的计划如下
    - {plans_group}

    ## 步骤（不需要输出）
    - 第1步：理解所有角色的计划，不要漏掉任何相关角色。
    - 第2步：做出推演与判断（决定每个角色的计划能否能成功）。
    - 第3步：执行所有计划。
    - 第4步：根据执行结果更新场景的状态以及所有角色的状态。
   
    ## 输出规则
    - 最终输出的结果，需要包括每个角色的结果(包括你自己)。
    """

#场景需要根据状态做出计划
def actor_plan_prompt(actor):
    prompt = f"""
    # 你需要做出计划（即你想要做的事，但是还没有做，或者是心里想的事情）    
    ## 步骤
    - 第1步：理解你自身当前状态。
    - 第2步：理解你的场景内所有角色的当前状态。
    - 第3步：根据以2步，输出你需要做出计划。请关注“计划的输出规则”
    
    ## 注意！输出的关键字，只能在如下中选择：
    - [fight]，代表着你计划攻击某个目标（角色）。
    - [leave]，代表着你计划要离开你当前场景。
    - [stay]，代表着你保持现状，不做任何事。
    
    ## 输出规则与示例：
    - 如果你想攻击某个目标，那么你的输出格式为：“[fight][目标的名字]:...“，...代表着你本次攻击要说的话与心里活动。
    - 如果你想离开，那么你的输出格式为：“[leave]:...“，...代表着你本次要说的话与心里活动。
    - 如果不在以上种情况，就输出"[stay]:..."，...代表着你本次要说的话与心里活动。

    ## 输出规则：
    - 输出在保证语意完整基础上字符尽量少。
    """
    return prompt

#
def actor_confirm_prompt(actor, stage_state):
    prompt = f"""
    #这是你所在场景的推演结果与执行结果，你需要接受这个事实，并且强制更新你的状态。
    ## 步骤(不要输出)
    - 第1步：回顾你的计划。
    - 第2步：确认并理解场景{stage_state}的推演结果。如果出现了你的名字（就是你）。
    - 第3步：对比你的计划在推演结果中的表现，是否得到执行。
    - 第4步：你需要更新你的状态。
    - 第5步：输出你的状态
    """
    return call_agent(actor, prompt)

## 输出格式
   
#
def main():
    #
    system_administrator = "系统管理员"

    #load!!!!
    world_watcher = World("世界观察者")
    world_watcher.connect("http://localhost:8004/world/")
    log = call_agent(world_watcher,  load_prompt)
    print(f"[{world_watcher.name}]:", log)
    print("==============================================")

    #
    old_hunter = NPC("卡斯帕·艾伦德")
    old_hunter.connect("http://localhost:8021/actor/npc/old_hunter/")
    log = call_agent(old_hunter, load_prompt)
    print(f"[{old_hunter.name}]:", log)
    print("==============================================")

    #
    old_hunters_dog = NPC("小狗'短剑'")
    old_hunters_dog.connect("http://localhost:8023/actor/npc/old_hunters_dog/")
    log = call_agent(old_hunters_dog, load_prompt)
    print(f"[{old_hunters_dog.name}]:", log)
    print("==============================================")

    #
    old_hunters_cabin = Stage("老猎人隐居的小木屋")
    old_hunters_cabin.connect("http://localhost:8022/stage/old_hunters_cabin/")
    log = call_agent(old_hunters_cabin, load_prompt)
    print(f"[{old_hunters_cabin.name}]:", log)
    print("==============================================")

    #
    world_watcher.add_stage(old_hunters_cabin)
    old_hunters_cabin.add_actor(old_hunter)
    old_hunters_cabin.add_actor(old_hunters_dog)

    #
    player = Player("yang_hang")
    log = call_agent(world_watcher,  f"""你知道了如下事件：{player.name}加入了这个世界""")
    print(f"[{world_watcher.name}]:", log)
    player.health = 10000000
    player.power = 10000000


    print("//////////////////////////////////////////////////////////////////////////////////////")
    print("//////////////////////////////////////////////////////////////////////////////////////")
    print("//////////////////////////////////////////////////////////////////////////////////////")

    #
    while True:
        usr_input = input("[user input]: ")
        if "/quit" in usr_input:
            sys.exit()
        if "/cancel" in usr_input:
            continue
        
        elif "/0" in usr_input:
            content = parse_input(usr_input, "/0")
            print("==============================================")


        elif "/1" in usr_input:
            content = parse_input(usr_input, "/1")
            print(f"[{system_administrator}]:", content)
            print(f"[{world_watcher.name}]:", call_agent(world_watcher, content))
            print("==============================================")

        elif "/2" in usr_input:
            content = parse_input(usr_input, "/2")
            print(f"[{system_administrator}]:", content)
            print(f"[{old_hunter.name}]:",  call_agent(old_hunter, content))
            print("==============================================")

        elif "3" in usr_input:
            content = parse_input(usr_input, "/3")
            print(f"[{system_administrator}]:", content)
            print(f"[{old_hunters_dog.name}]:",  call_agent(old_hunters_dog, content))
            print("==============================================")
        
        elif "/4" in usr_input:
            
            # 所有人都知道了这件事
            content = parse_input(usr_input, "/4")
            print(f"[{system_administrator}]:", content)

            old_hunters_cabin.chat_history.append(HumanMessage(content=content))
            print(f"[{old_hunters_cabin.name}]:", call_agent(old_hunters_cabin, "更新你的状态"))

            for actor in old_hunters_cabin.actors:
                actor.chat_history.append(HumanMessage(content=content))
                print(f"[{actor.name}]:", call_agent(actor, "更新你的状态"))

            #print(f"[{old_hunters_cabin.name}]:", call_agent(old_hunters_cabin, content))
            #for actors old_hunters_cabin.actors
            #last_chat = old_hunters_cabin.chat_history[-1]
            #print(f"[{old_hunters_cabin.name}]%", last_chat.content)
           
            print("==============================================")

        elif "/rr" in usr_input:
            content = parse_input(usr_input, "/rr")
            current_stage = old_hunters_cabin
            all_actors = current_stage.actors
            #最后状态
            last_chat = current_stage.chat_history[-1]
            print(f"[{current_stage.name}]%", last_chat.content)

            ##
            plans = []
            #
            log = call_agent(current_stage, stage_plan_prompt(current_stage))
            #print(f"<{current_stage.name}>:", log)
            str = f"[{current_stage.name}]{log}"
            plans.append(str)
            #print("==============================================")
            
            #
            for actor in all_actors:
                log = call_agent(actor, actor_plan_prompt(actor))
                #print(f"<{actor.name}>:", log)
                str = f"[{actor.name}]{log}"
                plans.append(str)
                #print("==============================================")

            print("==============================================")
            fight_events:FightEvent = []
            stay_events:StayEvent = []
            total_plans = []
            #print('\n'.join(plans))
            for plan in plans:
                #print(plan)
                # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                a, b = plan.split(':')
                extracted_elements = []
                pattern = r"\[(.*?)\]"
                matches = re.findall(pattern, a)
                for match in matches:
                    extracted_elements.append(f"[{match}]")
                    #print(match)

                #print(extracted_elements)

                name = extracted_elements[0]
                actions = extracted_elements[1:]
                if actions[0] == "[fight]":
                    target = actions[1]
                    #print(f"{name}=>{target}:{b}")
                    fight_events.append(FightEvent(current_stage, name, target, b))
                    #print(fight_events[-1])
                    total_plans.append(fight_events[-1].make_plan())

                elif actions[0] == "[leave]":
                    print("想跑？？？？？？？？？？？？？？？？？？！！！！！！？")

                elif actions[0] == "[stay]":
                    #print(f"{name}:{b}")
                    stay_events.append(StayEvent(current_stage, name, b))
                    #print(stay_events[-1])
                    total_plans.append(stay_events[-1].make_plan())
                #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            ##
            #print(total_plans)
            plan_group_str = '\n'.join(total_plans)
            print(plan_group_str)

            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            director_prompt_str = director_prompt(current_stage, plan_group_str)
            director_res = call_agent(current_stage, director_prompt_str)
            print(f"[{current_stage.name}]:", director_res)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

              ##确认行动
            for actor in current_stage.actors:
                actor_comfirm_prompt_str = actor_confirm_prompt(actor, director_res)
                actor_res = call_agent(actor, actor_comfirm_prompt_str)
                print(f"[{actor.name}]=>", actor_res)

            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("==============================================")

        
if __name__ == "__main__":
    print("==============================================")
    main()