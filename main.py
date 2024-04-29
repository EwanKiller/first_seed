import os
from typing import Optional
from loguru import logger
import datetime
from auxiliary.dialogue_rule import parse_target_and_message
from auxiliary.builders import WorldDataBuilder
from rpg_game import RPGGame 
from auxiliary.player_proxy import create_player_proxy, get_player_proxy, TEST_PLAYER_NAME
from auxiliary.gm_input_command import GMCommandSimulateRequest, GMCommandSimulateRequestThenRemoveConversation, GMCommandPlayerCtrlAnyNPC
from auxiliary.player_input_command import (PlayerCommandLogin)
from auxiliary.extended_context import ExtendedContext
from auxiliary.file_system import FileSystem
from auxiliary.memory_system import MemorySystem
from typing import Optional
from auxiliary.agent_connect_system import AgentConnectSystem
from auxiliary.code_name_component_system import CodeNameComponentSystem
from auxiliary.chaos_engineering_system import EmptyChaosEngineeringSystem, IChaosEngineering
### 专门的混沌工程系统
from budding_world.chaos_budding_world import ChaosBuddingWorld


def user_input_pre_command(input_val: str, split_str: str)-> str:
    if split_str in input_val:
        return input_val.split(split_str)[1].strip()
    return input_val


### 临时的，写死创建budding_world
def read_world_data(worldname: str) -> Optional[WorldDataBuilder]:
    #先写死！！！！
    version = 'ewan'
    runtimedir = f"./budding_world/gen_runtimes/"
    worlddata: str = f"{runtimedir}{worldname}.json"
    if not os.path.exists(worlddata):
        logger.error("未找到存档，请检查存档是否存在。")
        return None

    worldbuilder: Optional[WorldDataBuilder] = WorldDataBuilder(worldname, version, runtimedir)
    if worldbuilder is None:
        logger.error("WorldDataBuilder初始化失败。")
        return None
    
    if not worldbuilder.check_version_valid(worlddata):
        logger.error("World.json版本不匹配，请检查版本号。")
        return None
    
    worldbuilder.build()
    return worldbuilder

##
def create_rpg_game(worldname: str, chaosengineering: Optional[IChaosEngineering]) -> RPGGame:

    # 依赖注入的特殊系统
    file_system = FileSystem("file_system， Because it involves IO operations, an independent system is more convenient.")
    memory_system = MemorySystem("memorey_system， Because it involves IO operations, an independent system is more convenient.")
    agent_connect_system = AgentConnectSystem("agent_connect_system， Because it involves net operations, an independent system is more convenient.")
    code_name_component_system = CodeNameComponentSystem("Build components by codename for special purposes")
    
    ### 混沌工程系统
    if chaosengineering is not None:
        chaos_engineering_system = chaosengineering
    else:
        chaos_engineering_system = EmptyChaosEngineeringSystem("empty_chaos")

    # 创建上下文
    context = ExtendedContext(file_system, 
                              memory_system, 
                              agent_connect_system, 
                              code_name_component_system, 
                              chaos_engineering_system, 
                              False,
                              10000000)

    # 创建游戏
    rpggame = RPGGame(worldname, context)
    return rpggame

###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
############################################################################################################################################### 
def main() -> None:

    log_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.add(f"logs/{log_start_time}.log", level="DEBUG")

    # 读取世界资源文件
    worldname = "World2"#input("请输入要进入的世界名称(必须与自动化创建的名字一致):")
    worlddata = read_world_data(worldname)
    if worlddata is None:
        logger.error("create_world_data_builder 失败。")
        return
    
    # 创建游戏 + 专门的混沌工程系统
    chaos_engineering_system = ChaosBuddingWorld("ChaosBuddingWorld")
    #chaos_engineering_system = None
    rpggame = create_rpg_game(worldname, chaos_engineering_system)
    if rpggame is None:
        logger.error("create_rpg_game 失败。")
        return
    
    # 创建世界
    rpggame.createworld(worlddata)
    # 先直接执行一次
    rpggame.execute()

    while True:
        systeminput = input("[system input]:")
        if "/quit" in systeminput:
            break
        
        elif "/login" in systeminput:
            # 测试的代码，上来就控制一个NPC目标，先写死
            create_player_proxy(TEST_PLAYER_NAME)
            playerproxy = get_player_proxy(TEST_PLAYER_NAME)
            assert playerproxy is not None
            playerstartcmd = PlayerCommandLogin("/player-login", rpggame, playerproxy, "无名的复活者")
            playerstartcmd.execute()
            rpggame.execute() # 测试 直接跑一次

        elif "/run" in systeminput:
            rpggame.execute()

        elif "/push" in systeminput:
            command = "/push"
            input_content = user_input_pre_command(systeminput, command) 
            push_command_parse_res: tuple[Optional[str], Optional[str]] = parse_target_and_message(input_content)
            if push_command_parse_res[0] is None or push_command_parse_res[1] is None:
                continue

            logger.debug(f"</force push command to {push_command_parse_res[0]}>:", input_content)
            ###
            gmcommandpush = GMCommandSimulateRequest("/push", rpggame, push_command_parse_res[0], push_command_parse_res[1])
            gmcommandpush.execute()
            ###
            logger.debug(f"{'=' * 50}")

        elif "/ask" in systeminput:
            if not rpggame.started:
                logger.warning("请先/run")
                continue
            command = "/ask"
            input_content = user_input_pre_command(systeminput, command)
            ask_command_parse_res: tuple[Optional[str], Optional[str]] = parse_target_and_message(input_content)
            if ask_command_parse_res[0] is None or ask_command_parse_res[1] is None:
                continue
            logger.debug(f"</ask command to {ask_command_parse_res[0]}>:", input_content)
            ###
            gmcommandask = GMCommandSimulateRequestThenRemoveConversation("/ask", rpggame, ask_command_parse_res[0], ask_command_parse_res[1])
            gmcommandask.execute()
            ###
            logger.debug(f"{'=' * 50}")

        elif "/who" in systeminput:
            if not rpggame.started:
                logger.warning("请先/run")
                continue
            command = "/who"
            bewho = user_input_pre_command(systeminput, command)
            ###
            playerproxy = get_player_proxy(TEST_PLAYER_NAME)
            assert playerproxy is not None
            playercommandbewho = GMCommandPlayerCtrlAnyNPC(command, rpggame, playerproxy, bewho)
            playercommandbewho.execute()
            ###            
            logger.debug(f"{'=' * 50}")
           
    rpggame.exit()


if __name__ == "__main__":
    main()