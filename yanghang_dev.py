from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langserve import RemoteRunnable
import sys
import json
from actor import Actor
from world import World
from stage import Stage
from stage import NPC
from player import Player
from action import Action, FIGHT, STAY, LEAVE
from make_plan import stage_plan_prompt, stage_plan, npc_plan
from console import Console


#
def parse_input(input_val: str, split_str: str)-> str:
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
## 输出规则：
- 保留关键信息(时间，地点，人物，事件)，不要推断，增加与润色。输出在保证语意完整基础上字符尽量少。
"""
######################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
def director_prompt(stage, movie_script):
    return f"""
    # 你按着我的给你的脚本来演绎过程，并适当润色让过程更加生动。
    ## 剧本如下
    - {movie_script}
    ## 步骤
    - 第1步：理解我的剧本
    - 第2步：根据剧本，完善你的故事讲述。要保证和脚本的结果一致。
    - 第3步：更新你的记忆
    ## 输出规则
    - 输出在保证语意完整基础上字符尽量少。
    """
######################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
#
def actor_confirm_prompt(actor, stage_state):
    stage = actor.stage
    actors = stage.actors
    actor_names = [actor.name for actor in actors]
    all_names = ' '.join(actor_names)
    return f"""
    # 这些事已经发生的事精，你更新了记忆
    - {stage_state}
    - 你知道 {all_names} 都还存在。
    """
######################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################
##################################################################################################################################################################################################################

#
def main():
    #
    system_administrator = "系统管理员"

    #load!!!!
    world_watcher = World("世界观察者")
    world_watcher.connect("http://localhost:8004/world/")
    log = world_watcher.call_agent(load_prompt)
    print(f"[{world_watcher.name}]:", log)
    print("==============================================")

    #
    old_hunter = NPC("卡斯帕·艾伦德")
    old_hunter.connect("http://localhost:8021/actor/npc/old_hunter/")
    log = old_hunter.call_agent(load_prompt)
    print(f"[{old_hunter.name}]:", log)
    print("==============================================")

    #
    old_hunters_dog = NPC("小狗'断剑'")
    old_hunters_dog.connect("http://localhost:8023/actor/npc/old_hunters_dog/")
    log = old_hunters_dog.call_agent(load_prompt)
    print(f"[{old_hunters_dog.name}]:", log)
    print("==============================================")

    #
    old_hunters_cabin = Stage("老猎人隐居的小木屋")
    old_hunters_cabin.connect("http://localhost:8022/stage/old_hunters_cabin/")
    log = old_hunters_cabin.call_agent(load_prompt)
    print(f"[{old_hunters_cabin.name}]:", log)
    print("==============================================")

    #
    world_watcher.add_stage(old_hunters_cabin)
    old_hunters_cabin.world = world_watcher
    #
    old_hunters_cabin.add_actor(old_hunter)
    old_hunter.stage = old_hunters_cabin

    old_hunters_cabin.add_actor(old_hunters_dog)
    old_hunters_dog.stage = old_hunters_cabin

    #
    player = None
    # Player("yang_hang")
    # player.max_hp = 1000000
    # player.hp = 1000000
    # player.damage = 100000
    # #player.connect("http://localhost:8023/12345/")
    # log = world_watcher.call_agent(f"""你知道了如下事件：{player.name}加入了这个世界""")
    # print(f"[{world_watcher.name}]:", log)

    print("//////////////////////////////////////////////////////////////////////////////////////")
    print("//////////////////////////////////////////////////////////////////////////////////////")
    print("//////////////////////////////////////////////////////////////////////////////////////")


    ###
    console = Console("系统管理员")

    #
    while True:
        usr_input = input("[user input]: ")
        if "/quit" in usr_input:
            sys.exit()
 
        elif "/call" in usr_input:
            command = "/call"
            input_content = console.parse_command(usr_input, command)
            print(f"</call>:", input_content)
            tp = console.parse_speak(input_content)
            target = tp[0]
            content = tp[1]
            if target == None:
                print("/call 没有指定@目标") 
                continue

            speak2actors = []
            if target == "all":
                speak2actors = world_watcher.all_actors()
            else:
                find_actor = world_watcher.get_actor(target)
                if find_actor != None:
                    speak2actors.append(find_actor)

            for actor in speak2actors:
                print(f"[{actor.name}] /call:", actor.call_agent(content))
            print("==============================================")


        elif "/run" in usr_input:
            command = "/run"
            stage_name = console.parse_command(usr_input, command)
            if stage_name == None:
                print("/run error1 = ", stage_name, "没有找到这个场景") 
                continue
            stage = world_watcher.get_stage(stage_name)
            if stage == None:
                print("/run error2 = ", stage_name, "没有找到这个场景") 
                continue
            print(f"[{stage.name}] /run:")
            state_run(stage, [])    
            print("==============================================")


        ##某人进入场景的事件
        elif "/player" in usr_input:
            player_settings = parse_input(usr_input, "/player")
            print(f"/player:", player_settings)
            if player == None:
                player = Player("yang_hang")

                if player_settings != None or player_settings != "":
                    player.description = player_settings
                    
                player.max_hp = 1000000
                player.hp = 1000000
                player.damage = 100000
                notify_world = world_watcher.call_agent(f"""你知道了如下事件：{player.name}加入了这个世界""")
                print(f"[{world_watcher.name}]:", notify_world)
                

            # if player.stage == None:
            #     old_hunters_cabin.add_actor(player)
            #     player.stage = old_hunters_cabin
                #print(f"[{old_hunters_cabin.name}]:", old_hunters_cabin.call_agent("更新你的状态"))


            # if player.stage != None:
            #     continue

            # Player("yang_hang")
            # player.max_hp = 1000000
            # player.hp = 1000000
            # player.damage = 100000
            # #player.connect("http://localhost:8023/12345/")
            # log = world_watcher.call_agent(f"""你知道了如下事件：{player.name}加入了这个世界""")
            # print(f"[{world_watcher.name}]:", log)



            # ## 必须加上！！！！！！！
            # old_hunters_cabin.add_actor(player)
            # player.stage = old_hunters_cabin
            # ###
            # event_prompt = f"""{player.name}{content}"""
            # print(f"[{player.name}]=>", event_prompt)

            # old_hunters_cabin.chat_history.append(HumanMessage(content=event_prompt))
            # print(f"[{old_hunters_cabin.name}]:", old_hunters_cabin.call_agent("更新你的状态"))

            # for actor in old_hunters_cabin.actors:
            #     if (actor == player):
            #         continue
            #     actor.chat_history.append(HumanMessage(content=event_prompt))
            #     print(f"[{actor.name}]:", actor.call_agent("更新你的状态"))
            print("==============================================")

               

        # elif "/talk" in usr_input:
        #     flag = "/talk"
        #     content = parse_input(usr_input, flag)
        #     print(f"[{player.name}]:", content)
        #     if player.stage == None:
        #         continue
        #     ###
        #     event_prompt = f"""{player.name}, {content}"""
        #     print(f"[{player.name}]=>", event_prompt)

        #     old_hunters_cabin.chat_history.append(HumanMessage(content=event_prompt))
        #     print(f"[{old_hunters_cabin.name}]:", old_hunters_cabin.call_agent("更新你的状态"))

        #     for actor in old_hunters_cabin.actors:
        #         if (actor == player):
        #             continue
        #         actor.chat_history.append(HumanMessage(content=event_prompt))
        #         print(f"[{actor.name}]:", actor.call_agent("更新你的状态"))
        #     print("==============================================")

        # elif "/attack" in usr_input:
        #     flag = "/attack"
        #     target_name = parse_input(usr_input, flag)
        #     print(f"[{player.name}] xxx:", target_name)
        #     if player.stage == None:
        #         continue
        #     target = player.stage.get_actor(target_name)
        #     if target == None:  
        #         continue

        #     players_action = Action(player, [FIGHT], [target.name], ["看招，雷霆大潮袭！！！！！"], [""])
        #     state_run(old_hunters_cabin, [players_action])          
        #     print("==============================================")


######################################################################
def state_run(current_stage: Stage, players_action: list[Action]) -> None:
        
    actions_collector: list[Action] = []

    ###场景的
    sp = stage_plan(current_stage)
    actions_collector.append(sp)

    ###npc
    for npc in current_stage.npcs:
        np = npc_plan(npc)
        actions_collector.append(np)

    ###角色的
    if len(players_action) > 0:
        actions_collector.extend(players_action)

    if len(actions_collector) == 0:
        print("没有行动")
        return
    print("-------------------------------------------------------------")


    ##记录用
    movie_script: list[str] = []

    # 先说话
    for action in actions_collector:
        if (len(action.say) > 0):
            movie_script.append(f"{action.planer.name}说：{action.say[0]}")
            print(f"{action.planer.name}说（或者是心里活动）：{action.say[0]}")
    print("-------------------------------------------------------------")

    fight_actions = []
    for action in actions_collector:
        if action.action[0] == FIGHT:
            fight_actions.append(action)

    leave_actions = []
    for action in actions_collector:
        if action.action[0] == LEAVE:
            leave_actions.append(action)

    stay_actions = []
    for action in actions_collector:
        if action.action[0] == STAY:
            stay_actions.append(action)

    ###先收集出来
    stay_actors: list[Actor] = []
    for action in stay_actions:
        print(f"stay action: {action}")
        if (action.planer == current_stage):
            continue
        stay_actors.append(action.planer)
        #movie_script.append(f"{action.planer.name}在{current_stage.name}里")
    
    ###先收集出来
    leave_actors: list[Actor] = []
    for action in leave_actions:
        print(f"leave action: {action}")
        leave_actors.append(action.planer)
        #movie_script.append(f"{action.planer.name}想要离开，并去往{action.targets}")

    print("-------------------------------------------------------------")

    #
    if len(fight_actions) > 0:
        movie_script.append("发生了战斗！")
    #
    dead_actors: list[Actor] = []
    for action in fight_actions:
        print(f"fight action: {action}")
        #进入战斗的不能跑
        if action.planer in leave_actors:
            leave_actors.remove(action.planer)

        #print(f"fight action: {action}")
        movie_script.append(f"{action.planer.name}对{action.targets}发动了攻击")

        for target_name in action.targets:
            target = current_stage.get_actor(target_name)
            if target == None or target == action.planer:
                continue

            #被攻击不能走
            if target in leave_actors:
                leave_actors.remove(target) 
                #movie_script.append(f"{target.name}被攻击，不能离开")

            #最简单战斗计算
            print("这是一个测试的战斗计算，待补充")
            target.hp -= action.planer.damage    
            movie_script.append(f"{action.planer.name}对{action.targets}产生了{action.planer.damage}点伤害")
            if target.hp <= 0:
                dead_actors.append(target)
                print(f"{target.name}已经死亡")
                movie_script.append(f"{target.name}已经死亡")
            else:
                print(f"{target.name}剩余{target.hp/target.max_hp*100}%血量")

    ##被打死的不能走
    for actor in dead_actors:
        if actor in leave_actors:
            leave_actors.remove(actor)
            #movie_script.append(f"{actor.name}已经死亡，不能离开")
        if  actor in stay_actors:
            stay_actors.remove(actor)
            #movie_script.append(f"{actor.name}已经死亡，不能留下")

    ##处理离开的
    for actor in leave_actors:
        actor.stage.remove_actor(actor)
        actor.stage = None
        movie_script.append(f"{actor.name}离开了{current_stage.name}")
        raise NotImplementedError("Code to handle leaving actors is not implemented yet.")
    
    for actor in dead_actors:
        actor.stage.remove_actor(actor)
        actor.stage = None
        movie_script.append(f"{actor.name}死了，离开了这个世界")

    ##处理留下的
    for actor in stay_actors:
        pass
        #print(f"{actor.name}留在{current_stage.name}")
        #movie_script.append(f"{actor.name}留在{current_stage.name}")

    movie_script_str = '\n'.join(movie_script)
    if len(movie_script_str) == 0:
        movie_script_str = "无事发生"
        #return

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    director_prompt_str = director_prompt(current_stage, movie_script_str)
    director_res = current_stage.call_agent(director_prompt_str)
    print(f"[{current_stage.name}]:", director_res)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    ##确认行动
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    for actor in current_stage.npcs:
        last_memory = f"""
        # 你知道发生了下面的事，并更新了你的记忆：
        {movie_script_str}
        """
        actor.chat_history.append(AIMessage(content=last_memory))
        actor_comfirm_prompt_str = actor_confirm_prompt(actor, director_res)
        actor_res = actor.call_agent(actor_comfirm_prompt_str)
        print(f"[{actor.name}]=>" + actor_res)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")




if __name__ == "__main__":
    main()