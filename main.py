from typing import List, Optional, Union
from entitas import Processors #type: ignore
from loguru import logger
import datetime
import json
from auxiliary.builder import WorldBuilder
from auxiliary.components import (
    BroadcastActionComponent, 
    SpeakActionComponent, 
    WorldComponent,
    StageComponent, 
    NPCComponent, 
    FightActionComponent, 
    PlayerComponent, 
    SimpleRPGRoleComponent, 
    LeaveForActionComponent, 
    HumanInterferenceComponent,
    UniquePropComponent,
    BackpackComponent,
    StageEntryConditionComponent,
    StageExitConditionComponent,
    WhisperActionComponent,
    SearchActionComponent)
from auxiliary.actor_action import ActorAction
from auxiliary.actor_agent import ActorAgent
from auxiliary.extended_context import ExtendedContext
from auxiliary.dialogue_rule import parse_command, parse_target_and_message_by_symbol
from entitas.entity import Entity
from systems.init_system import InitSystem
from systems.stage_plan_system import StagePlanSystem
from systems.npc_plan_system import NPCPlanSystem
from systems.speak_action_system import SpeakActionSystem
from systems.fight_action_system import FightActionSystem
from systems.leave_for_action_system import LeaveForActionSystem
from systems.director_system import DirectorSystem
from systems.dead_action_system import DeadActionSystem
from systems.destroy_system import DestroySystem
from systems.tag_action_system import TagActionSystem
from systems.data_save_system import DataSaveSystem
from systems.broadcast_action_system import BroadcastActionSystem  
from systems.whisper_action_system import WhisperActionSystem 
from systems.search_props_system import SearchPropsSystem
from systems.mind_voice_action_system import MindVoiceActionSystem

from langchain_core.messages import (
    HumanMessage,
    AIMessage)


###############################################################################################################################################
def create_entities(context: ExtendedContext, worldbuilder: WorldBuilder) -> None:
        if worldbuilder.data is None:
            return
        ##创建world
        worldagent = ActorAgent(worldbuilder.data['name'], worldbuilder.data['url'], worldbuilder.data['memory'])
        world_entity = context.create_entity() 
        world_entity.add(WorldComponent, worldagent.name, worldagent)

        for stage_builder in worldbuilder.stage_builders:    
            if stage_builder.data is None:
                logger.error("没有StageBuilder数据，请检查game_settings.json配置。")
                continue 
            #创建stage       
            stage_agent = ActorAgent(stage_builder.data['name'], stage_builder.data['url'], stage_builder.data['memory'])
            stage_entity = context.create_entity()
            stage_entity.add(StageComponent, stage_agent.name, stage_agent, [])
            stage_entity.add(SimpleRPGRoleComponent, stage_agent.name, 100, 100, 1, "")

            for npc_builder in stage_builder.npc_builders:
                if npc_builder.data is None:
                    continue
                #创建npc
                npc_agent = ActorAgent(npc_builder.data['name'], npc_builder.data['url'], npc_builder.data['memory'])
                npc_entity = context.create_entity()
                npc_entity.add(NPCComponent, npc_agent.name, npc_agent, stage_agent.name)
                npc_entity.add(SimpleRPGRoleComponent, npc_agent.name, 100, 100, 20, "")
                npc_entity.add(BackpackComponent, npc_agent.name)

                context.file_system.init_backpack_component(npc_entity.get(BackpackComponent))
            
            for unique_prop_builder in stage_builder.unique_prop_builders:
                if unique_prop_builder.data is None:
                    continue
                #创建道具
                prop_entity = context.create_entity()
                prop_entity.add(UniquePropComponent, unique_prop_builder.data['name'])
            
            enter_condition_set = set()
            for enter_condition_builder in stage_builder.entry_condition_builders:
                if enter_condition_builder.data is not None:
                    enter_condition_set.add(enter_condition_builder.data['name'])
                    
            if len(enter_condition_set) > 0:
                stage_entity.add(StageEntryConditionComponent, enter_condition_set)
                logger.debug(f"{stage_agent.name}的入口条件为：{enter_condition_set}")

            exit_condition_set = set()
            for exit_condition_builder in stage_builder.exit_condition_builders:
                if exit_condition_builder.data is not None:
                    exit_condition_set.add(exit_condition_builder.data['name'])

            if len(exit_condition_set) > 0:
                stage_entity.add(StageExitConditionComponent, exit_condition_set)
                logger.debug(f"{stage_agent.name}的出口条件为：{exit_condition_set}")

            if stage_builder.player_builders is None:
                logger.error("没有PlayerBuilders，请检查game_settings.json配置。")
                return None               
            
            for player_builder in stage_builder.player_builders:
                if player_builder.data is None:
                    logger.error("没有PlayerBuilder数据，请检查game_settings.json配置。")
                    continue
                #创建player
                player_agent = ActorAgent(player_builder.data['name'], player_builder.data['url'], player_builder.data['memory'])
                player_entity = context.create_entity()
                player_entity.add(NPCComponent, player_agent.name, player_agent, stage_agent.name)
                player_entity.add(SimpleRPGRoleComponent, player_agent.name, 10000, 10000, 10, "")
                player_entity.add(BackpackComponent, player_agent.name)
                player_entity.add(PlayerComponent, player_agent.name)

                context.file_system.init_backpack_component(player_entity.get(BackpackComponent))
    
def set_default_player(context: ExtendedContext, player_name: str = "player") -> None:
    player_entity: Optional[Entity] = context.getplayer()
    if player_entity is not None:
        player_entity.replace(PlayerComponent, player_name)
        if player_entity.has(BackpackComponent):
            player_backpack_comp: BackpackComponent = player_entity.get(BackpackComponent)
            context.file_system.add_content_into_backpack(player_backpack_comp, "生锈的铁剑")
            context.file_system.add_content_into_backpack(player_backpack_comp, "破旧的盔甲")

def init_NPCs_settings(context: ExtendedContext) -> None:
    npc_entity: Optional[Entity] = context.get_entity_by_name("坏运气先生")
    if npc_entity is not None:
        context.file_system.add_content_into_backpack(npc_entity.get(BackpackComponent), "老鼠洞的位置")

###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
############################################################################################################################################### 
def main() -> None:

    log_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.add(f"logs/{log_start_time}.log", level="DEBUG")

    context = ExtendedContext()
    processors = Processors()
    path: str = "./game_settings.json"
    playername = "yanghang"

    try:
        with open(path, "r") as file:
            json_data = json.load(file)

            #构建数据
            world_builder = WorldBuilder()
            world_builder.build(json_data)

            #创建所有entities
            create_entities(context, world_builder)
            #设置默认player
            set_default_player(context)
            #初始化npc的背包
            init_NPCs_settings(context)


    except Exception as e:
        logger.exception(e)
        return        

    #初始化系统########################
    processors.add(InitSystem(context))
    #规划逻辑########################
    processors.add(StagePlanSystem(context))
    processors.add(NPCPlanSystem(context))
    #行动逻辑########################
    processors.add(TagActionSystem(context))
    processors.add(MindVoiceActionSystem(context))
    processors.add(WhisperActionSystem(context))
    processors.add(BroadcastActionSystem(context))
    processors.add(SpeakActionSystem(context))
    #死亡必须是战斗之后，因为如果死了就不能离开###############
    processors.add(FightActionSystem(context))
    processors.add(DeadActionSystem(context)) 
    #########################################
    # 处理搜寻道具行为
    processors.add(SearchPropsSystem(context))
    # 处理离开并去往的行为
    processors.add(LeaveForActionSystem(context))
    #行动结束后导演
    processors.add(DirectorSystem(context))
    #########################################
    ###必须最后
    processors.add(DestroySystem(context))
    processors.add(DataSaveSystem(context))

    ####
    inited:bool = False
    started:bool = False

    while True:
        usr_input = input("[user input]: ")
        if "/quit" in usr_input:
            break

        elif "/run" in usr_input:
            #顺序不要动！！！！！！！！！
            if not inited:
                inited = True
                processors.activate_reactive_processors()
                processors.initialize()
            processors.execute()
            processors.cleanup()
            started = True
            logger.debug("==============================================")

        elif "/push" in usr_input:
            # if not started:
            #     logger.warning("请先/run")
            #     continue
            command = "/push"
            input_content = parse_command(usr_input, command) 
            push_command_parse_res: tuple[str, str] = parse_target_and_message_by_symbol(input_content)
            logger.debug(f"</force push command to {push_command_parse_res[0]}>:", input_content)
            debug_push(context, push_command_parse_res[0], push_command_parse_res[1])
            logger.debug(f"{'=' * 50}")

        elif "/ask" in usr_input:
            if not started:
                logger.warning("请先/run")
                continue
            command = "/ask"
            input_content = parse_command(usr_input, command)
            ask_command_parse_res: tuple[str, str] = parse_target_and_message_by_symbol(input_content)
            logger.debug(f"</ask command to {ask_command_parse_res[0]}>:", input_content)
            debug_ask(context, ask_command_parse_res[0], ask_command_parse_res[1])
            logger.debug(f"{'=' * 50}")

        elif "/showstages" in usr_input:
            if not started:
                logger.warning("请先/run")
                continue
            command = "/showstages"
            who = parse_command(usr_input, command)
            log = context.show_stages_log()
            logger.debug(f"/showstages: \n{log}")
            logger.debug(f"{'=' * 50}")

        elif "/who" in usr_input:
            if not started:
                logger.warning("请先/run")
                continue
            command = "/who"
            who = parse_command(usr_input, command)
            debug_be_who(context, who, playername)
            logger.debug(f"{'=' * 50}")
           
        elif "/attack" in usr_input:
            if not started:
                logger.warning("请先/run")
                continue
            command = "/attack"
            target_name = parse_command(usr_input, command)    
            debug_attack(context, target_name)
            logger.debug(f"{'=' * 50}")
        
        elif "/mem" in usr_input:
            if not started:
                logger.warning("请先/run")
                continue
            command = "/mem"
            target_name = parse_command(usr_input, command)
            debug_chat_history(context, target_name)
            logger.debug(f"{'=' * 50}")
        
        elif "/leave" in usr_input:
            if not started:
                logger.warning("请先/run")
                continue
            command = "/leave"
            target_name = parse_command(usr_input, command)
            debug_leave(context, target_name)
            logger.debug(f"{'=' * 50}")
        
        elif "/broadcast" in usr_input:
            if not started:
                logger.warning("请先/run")
                continue
            command = "/broadcast"
            content = parse_command(usr_input, command)
            debug_broadcast(context, content)
            logger.debug(f"{'=' * 50}")
            
        elif "/speak" in usr_input:
            if not started:
                logger.warning("请先/run")
                continue
            command = "/speak"
            content = parse_command(usr_input, command)
            debug_speak(context, content)
            logger.debug(f"{'=' * 50}")

        elif "/whisper" in usr_input:
            if not started:
                logger.warning("请先/run")
                continue
            command = "/whisper"
            content = parse_command(usr_input, command)
            debug_whisper(context, content)
            logger.debug(f"{'=' * 50}")
        
        elif "/search" in usr_input:
            if not started:
                logger.warning("请先/run")
                continue
            command = "/search"
            content = parse_command(usr_input, command)
            debug_search(context, content)
            logger.debug(f"{'=' * 50}")

    processors.clear_reactive_processors()
    processors.tear_down()
    logger.info("Game Over")

###############################################################################################################################################
def debug_push(context: ExtendedContext, name: str, content: str) -> Union[None, NPCComponent, StageComponent, WorldComponent]:

    npc_entity: Optional[Entity] = context.getnpc(name)
    if npc_entity is not None:
        npc_comp: NPCComponent = npc_entity.get(NPCComponent)
        npc_request: Optional[str] = npc_comp.agent.request(content)
        if npc_request is not None:
            npc_comp.agent.chat_history.pop()
        return npc_comp
    
    stage_entity: Optional[Entity] = context.getstage(name)
    if stage_entity is not None:
        stage_comp: StageComponent = stage_entity.get(StageComponent)
        stage_request: Optional[str] = stage_comp.agent.request(content)
        if stage_request is not None:
            stage_comp.agent.chat_history.pop()
        return stage_comp
    
    world_entity: Optional[Entity] = context.getworld()
    if world_entity is not None:
        world_comp: WorldComponent = world_entity.get(WorldComponent)
        request: Optional[str] = world_comp.agent.request(content)
        if request is not None:
            world_comp.agent.chat_history.pop()
        return world_comp

    return None        
    
def debug_ask(context: ExtendedContext, name: str, content: str) -> None:
    pushed_comp = debug_push(context, name, content)
    if pushed_comp is None:
        logger.warning(f"debug_ask: {name} not found.")
        return
    pushed_agent: ActorAgent = pushed_comp.agent
    pushed_agent.chat_history.pop()

###############################################################################################################################################
def debug_be_who(context: ExtendedContext, name: str, playname: str) -> None:

    playerentity = context.getplayer()
    if playerentity is not None:
        playercomp = playerentity.get(PlayerComponent)
        logger.debug(f"debug_be_who current player is : {playercomp.name}")
        playerentity.remove(PlayerComponent)

    entity = context.getnpc(name)
    if entity is not None:
        npccomp = entity.get(NPCComponent)
        logger.debug(f"debug_be_who => : {npccomp.name} is {playname}")
        entity.add(PlayerComponent, playname)
        return
    
    entity = context.getstage(name)
    if entity is not None:
        stagecomp = entity.get(StageComponent)
        logger.debug(f"debug_be_who => : {stagecomp.name} is {playname}")
        entity.add(PlayerComponent, playname)
        return
###############################################################################################################################################
def debug_attack(context: ExtendedContext, dest: str) -> None:
    
    playerentity = context.getplayer()
    if playerentity is None:
        logger.warning("debug_attack: player is None")
        return
       
    if playerentity.has(NPCComponent):
        npc_comp: NPCComponent = playerentity.get(NPCComponent)
        action = ActorAction(npc_comp.name, "FightActionComponent", [dest])
        playerentity.add(FightActionComponent, action)
        if not playerentity.has(HumanInterferenceComponent):
            playerentity.add(HumanInterferenceComponent, f'{npc_comp.name}攻击{dest}')
        logger.debug(f"debug_attack: {npc_comp.name} add {action}")
        return
    
    elif playerentity.has(StageComponent):
        stage_comp: StageComponent = playerentity.get(StageComponent)
        action = ActorAction(stage_comp.name, "FightActionComponent", [dest])
        if not playerentity.has(HumanInterferenceComponent):
            playerentity.add(HumanInterferenceComponent, f'{stage_comp.name}攻击{dest}')
        playerentity.add(FightActionComponent, action)
        logger.debug(f"debug_attack: {stage_comp.name} add {action}")
        return

###############################################################################################################################################
    
def debug_chat_history(context: ExtendedContext, name: str) -> None:
    entity = context.getnpc(name)
    if entity is not None:
        npc_comp: NPCComponent = entity.get(NPCComponent)
        npc_agent: ActorAgent = npc_comp.agent
        logger.info(f"{'=' * 50}\ndebug_chat_history for {npc_comp.name} => :")
        for history in npc_agent.chat_history:
            if isinstance(history, HumanMessage):
                logger.info(f"{'=' * 50}\nHuman:{history.content}")
            elif isinstance(history, AIMessage):
                logger.info(f"{'=' * 50}\nAI:{history.content}")
        logger.info(f"{'=' * 50}")
        return
    
    entity = context.getstage(name)
    if entity is not None:
        stage_comp: StageComponent = entity.get(StageComponent)
        stage_agent: ActorAgent = stage_comp.agent
        logger.info(f"{'=' * 50}\ndebug_chat_history for {stage_comp.name} => :\n")
        for history in stage_agent.chat_history:
            if isinstance(history, HumanMessage):
                logger.info(f"Human:{history.content}")
            elif isinstance(history, AIMessage):
                logger.info(f"AI:{history.content}")
        logger.info(f"{'=' * 50}")
        return


###############################################################################################################################################

def debug_leave(context: ExtendedContext, stagename: str) -> None:
    playerentity = context.getplayer()
    if playerentity is None:
        logger.warning("debug_leave: player is None")
        return
    
    npc_comp: NPCComponent = playerentity.get(NPCComponent)
    action = ActorAction(npc_comp.name, "LeaveForActionComponent", [stagename])
    playerentity.add(LeaveForActionComponent, action)
    if not playerentity.has(HumanInterferenceComponent):
        playerentity.add(HumanInterferenceComponent, f'{npc_comp.name}离开了{stagename}')

    newmemory = f"""{{
        "LeaveForActionComponent": ["{stagename}"]
    }}"""
    context.add_agent_memory(playerentity, newmemory)
    logger.debug(f"debug_leave: {npc_comp.name} add {action}")
    
###############################################################################################################################################
def debug_broadcast(context: ExtendedContext, content: str) -> None:
    playerentity = context.getplayer()
    if playerentity is None:
        logger.warning("debug_broadcast: player is None")
        return
    
    npc_comp: NPCComponent = playerentity.get(NPCComponent)
    action = ActorAction(npc_comp.name, "BroadcastActionComponent", [content])
    playerentity.add(BroadcastActionComponent, action)
    playerentity.add(HumanInterferenceComponent, f'{npc_comp.name}大声说道：{content}')

    newmemory = f"""{{
        "BroadcastActionComponent": ["{content}"]
    }}"""
    context.add_agent_memory(playerentity, newmemory)
    logger.debug(f"debug_broadcast: {npc_comp.name} add {action}")

###############################################################################################################################################
def debug_speak(context: ExtendedContext, content: str) -> None:
    playerentity = context.getplayer()
    if playerentity is None:
        logger.warning("debug_speak: player is None")
        return
    
    npc_comp: NPCComponent = playerentity.get(NPCComponent)
    action = ActorAction(npc_comp.name, "SpeakActionComponent", [content])
    playerentity.add(SpeakActionComponent, action)
    if not playerentity.has(HumanInterferenceComponent):
        playerentity.add(HumanInterferenceComponent, f'{npc_comp.name}说道：{content}')

    newmemory = f"""{{
        "SpeakActionComponent": ["{content}"]
    }}"""
    context.add_agent_memory(playerentity, newmemory)
    logger.debug(f"debug_speak: {npc_comp.name} add {action}")

###############################################################################################################################################
def debug_whisper(context: ExtendedContext, content: str) -> None:
    playerentity = context.getplayer()
    if playerentity is None:
        logger.warning("debug_whisper: player is None")
        return
    
    npc_comp: NPCComponent = playerentity.get(NPCComponent)
    action = ActorAction(npc_comp.name, "WhisperActionComponent", [content])
    playerentity.add(WhisperActionComponent, action)
    if not playerentity.has(HumanInterferenceComponent):
        playerentity.add(HumanInterferenceComponent, f'{npc_comp.name}低语道：{content}')

    newmemory = f"""{{
        "WhisperActionComponent": ["{content}"]
    }}"""
    context.add_agent_memory(playerentity, newmemory)
    logger.debug(f"debug_whisper: {npc_comp.name} add {action}")

###############################################################################################################################################

def debug_search(context: ExtendedContext, content: str) -> None:
    playerentity = context.getplayer()
    if playerentity is None:
        logger.warning("debug_search: player is None")
        return
    
    npc_comp: NPCComponent = playerentity.get(NPCComponent)
    action = ActorAction(npc_comp.name, "SearchActionComponent", [content])
    playerentity.add(SearchActionComponent, action)
    if not playerentity.has(HumanInterferenceComponent):
        playerentity.add(HumanInterferenceComponent, f'{npc_comp.name}搜索{content}')

    newmemory = f"""{{
        "SearchActionComponent": ["{content}"]
    }}"""
    context.add_agent_memory(playerentity, newmemory)
    logger.debug(f"debug_search: {npc_comp.name} add {action}")


if __name__ == "__main__":
    main()