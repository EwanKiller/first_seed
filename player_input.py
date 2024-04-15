from rpg_game import RPGGame
from loguru import logger
from auxiliary.components import (
    BroadcastActionComponent, 
    SpeakActionComponent, 
    StageComponent, 
    NPCComponent, 
    FightActionComponent, 
    PlayerComponent, 
    LeaveForActionComponent, 
    WhisperActionComponent,
    SearchActionComponent,
    PrisonBreakActionComponent)
from auxiliary.actor_action import ActorAction
from player_proxy import PlayerProxy

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
class PlayerInput:
    def __init__(self, inputname: str, game: RPGGame, playerproxy: PlayerProxy) -> None:
        self.inputname: str = inputname
        self.game: RPGGame = game
        self.playerproxy: PlayerProxy = playerproxy
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
class PlayerCommandLogin(PlayerInput):

    def __init__(self, name: str, game: RPGGame, playerproxy: PlayerProxy, targetname: str) -> None:
        super().__init__(name, game, playerproxy)
        self.targetname = targetname

    def execute(self) -> None:
        context = self.game.extendedcontext
        name = self.targetname
        playername = self.playerproxy.name
        logger.debug(f"{self.inputname}, player name: {playername}, target name: {name}")

        npcentity = context.getnpc(name)
        if npcentity is None:
            # 扮演的角色，本身就不存在于这个世界
            logger.warning(f"{self.inputname}, npc is None, login failed")
            return

        playerentity = context.getplayer(playername)
        if playerentity is not None:
            # 已经登陆完成
            logger.debug(f"{self.inputname}, already login")
            return
        
        playercomp: PlayerComponent = npcentity.get(PlayerComponent)
        if playercomp is None:
            # 扮演的角色不是设定的玩家可控制NPC
            logger.warning(f"{self.inputname}, npc is not player ctrl npc, login failed")
            return
        
        if playercomp.name != "" and playercomp.name != playername:
            # 已经有人控制了，但不是你
            logger.warning(f"{self.inputname}, player already ctrl by some player, login failed")
            return
        
        npccomp: NPCComponent = npcentity.get(NPCComponent)
        npcentity.replace(PlayerComponent, playername)
        logger.debug(f"{self.inputname}, [{npccomp.name}] is now controlled by the player [{playername}]")
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
class PlayerCommandCtrlNPC(PlayerInput):

    def __init__(self, name: str, game: RPGGame, playerproxy: PlayerProxy, targetname: str) -> None:
        super().__init__(name, game, playerproxy)
        self.targetname = targetname

    def execute(self) -> None:
        context = self.game.extendedcontext
        name = self.targetname
        playername = self.playerproxy.name
        logger.debug(f"{self.inputname}, player name: {playername}, target name: {name}")

        playerentity = context.getplayer(playername)
        if playerentity is None:
            logger.warning(f"{self.inputname}, player is None")
            return
        
        playercomp: PlayerComponent = playerentity.get(PlayerComponent)
        logger.debug(f"{self.inputname}, current player name: {playercomp.name}")

        ##停止控制当前的
        playerentity.remove(PlayerComponent)

        #准备控制新的
        targetnpc = context.getnpc(name)
        if targetnpc is not None:
            npccomp: NPCComponent = targetnpc.get(NPCComponent)
            logger.debug(f"{self.inputname}: [{npccomp.name}] is now controlled by the player [{playername}]")
            targetnpc.add(PlayerComponent, playername)
        else:
            logger.error(f"{self.inputname}, npc is None")
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################     
class PlayerCommandAttack(PlayerInput):

    def __init__(self, name: str, game: RPGGame, playerproxy: PlayerProxy, targetname: str) -> None:
        super().__init__(name, game, playerproxy)
        self.targetname = targetname

    def execute(self) -> None:
        context = self.game.extendedcontext 
        dest = self.targetname
        playerentity = context.getplayer(self.playerproxy.name)
        if playerentity is None:
            logger.warning("debug_attack: player is None")
            return
        
        if playerentity.has(NPCComponent):
            npc_comp: NPCComponent = playerentity.get(NPCComponent)
            action = ActorAction(npc_comp.name, "FightActionComponent", [dest])
            playerentity.add(FightActionComponent, action)
            logger.debug(f"debug_attack: {npc_comp.name} add {action}")
            return
        
        elif playerentity.has(StageComponent):
            stage_comp: StageComponent = playerentity.get(StageComponent)
            action = ActorAction(stage_comp.name, "FightActionComponent", [dest])
            playerentity.add(FightActionComponent, action)
            logger.debug(f"debug_attack: {stage_comp.name} add {action}")
            return
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################     
class PlayerCommandLeaveFor(PlayerInput):

    def __init__(self, name: str, game: RPGGame, playerproxy: PlayerProxy, stagename: str) -> None:
        super().__init__(name, game, playerproxy)
        self.stagename = stagename

    def execute(self) -> None:
        context = self.game.extendedcontext
        stagename = self.stagename
        playerentity = context.getplayer(self.playerproxy.name)
        if playerentity is None:
            logger.warning("debug_leave: player is None")
            return
        
        npc_comp: NPCComponent = playerentity.get(NPCComponent)
        action = ActorAction(npc_comp.name, "LeaveForActionComponent", [stagename])
        playerentity.add(LeaveForActionComponent, action)
        newmemory = f"""{{
            "LeaveForActionComponent": ["{stagename}"]
        }}"""
        context.add_human_message_to_entity(playerentity, newmemory)
        logger.debug(f"debug_leave: {npc_comp.name} add {action}")

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################     
class PlayerCommandPrisonBreak(PlayerInput):

    def __init__(self, name: str, game: RPGGame, playerproxy: PlayerProxy) -> None:
        super().__init__(name, game, playerproxy)

    def execute(self) -> None:
        context = self.game.extendedcontext
        playerentity = context.getplayer(self.playerproxy.name)
        if playerentity is None:
            logger.warning("debug_leave: player is None")
            return
        
        npccomp: NPCComponent = playerentity.get(NPCComponent)
        currentstagename: str = npccomp.current_stage
        stageentity = context.getstage(currentstagename)
        if stageentity is None:
            logger.error(f"PrisonBreakActionSystem: {currentstagename} is None")
            return

        action = ActorAction(npccomp.name, PrisonBreakActionComponent.__name__, [currentstagename])
        playerentity.add(LeaveForActionComponent, action)
        newmsg = f"""{{"{PrisonBreakActionComponent.__name__}": ["{currentstagename}"]}}"""
        context.add_human_message_to_entity(playerentity, newmsg)
        logger.debug(f"debug_leave: {npccomp.name} add {action}")
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
class PlayerCommandBroadcast(PlayerInput):

    def __init__(self, name: str, game: RPGGame, playerproxy: PlayerProxy, content: str) -> None:
        super().__init__(name, game, playerproxy)
        self.content = content

    def execute(self) -> None:
        context = self.game.extendedcontext
        content = self.content
        playerentity = context.getplayer(self.playerproxy.name)
        if playerentity is None:
            logger.warning("debug_broadcast: player is None")
            return
        
        npc_comp: NPCComponent = playerentity.get(NPCComponent)
        action = ActorAction(npc_comp.name, "BroadcastActionComponent", [content])
        playerentity.add(BroadcastActionComponent, action)
        #playerentity.add(HumanInterferenceComponent, f'{npc_comp.name}大声说道：{content}')
        newmemory = f"""{{
            "BroadcastActionComponent": ["{content}"]
        }}"""
        context.add_human_message_to_entity(playerentity, newmemory)
        logger.debug(f"debug_broadcast: {npc_comp.name} add {action}")
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
class PlayerCommandSpeak(PlayerInput):

    def __init__(self, name: str, game: RPGGame, playerproxy: PlayerProxy, commandstr: str) -> None:
        super().__init__(name, game, playerproxy)
        self.commandstr = commandstr

    def execute(self) -> None:
        context = self.game.extendedcontext
        content = self.commandstr
        playerentity = context.getplayer(self.playerproxy.name)
        if playerentity is None:
            logger.warning("debug_speak: player is None")
            return
        
        npc_comp: NPCComponent = playerentity.get(NPCComponent)
        action = ActorAction(npc_comp.name, "SpeakActionComponent", [content])
        playerentity.add(SpeakActionComponent, action)
        newmemory = f"""{{
            "SpeakActionComponent": ["{content}"]
        }}"""
        context.add_human_message_to_entity(playerentity, newmemory)
        logger.debug(f"debug_speak: {npc_comp.name} add {action}")
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
class PlayerCommandWhisper(PlayerInput):

    def __init__(self, name: str, game: RPGGame, playerproxy: PlayerProxy, commandstr: str) -> None:
        super().__init__(name, game, playerproxy)
        self.commandstr = commandstr

    def execute(self) -> None:
        context = self.game.extendedcontext
        content = self.commandstr
        playerentity = context.getplayer(self.playerproxy.name)
        if playerentity is None:
            logger.warning("debug_whisper: player is None")
            return
        
        npc_comp: NPCComponent = playerentity.get(NPCComponent)
        action = ActorAction(npc_comp.name, "WhisperActionComponent", [content])
        playerentity.add(WhisperActionComponent, action)
        newmemory = f"""{{
            "WhisperActionComponent": ["{content}"]
        }}"""
        context.add_human_message_to_entity(playerentity, newmemory)
        logger.debug(f"debug_whisper: {npc_comp.name} add {action}")
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
class PlayerCommandSearch(PlayerInput):

    def __init__(self, name: str, game: RPGGame, playerproxy: PlayerProxy, targetname: str) -> None:
        super().__init__(name, game, playerproxy)
        self.targetname = targetname

    def execute(self) -> None:
        context = self.game.extendedcontext
        content = self.targetname
        playerentity = context.getplayer(self.playerproxy.name)
        if playerentity is None:
            logger.warning("debug_search: player is None")
            return
        
        npc_comp: NPCComponent = playerentity.get(NPCComponent)
        action = ActorAction(npc_comp.name, "SearchActionComponent", [content])
        playerentity.add(SearchActionComponent, action)
        newmemory = f"""{{
            "SearchActionComponent": ["{content}"]
        }}"""
        context.add_human_message_to_entity(playerentity, newmemory)
        logger.debug(f"debug_search: {npc_comp.name} add {action}")
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################