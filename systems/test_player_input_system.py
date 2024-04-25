from entitas import ExecuteProcessor #type: ignore
from auxiliary.extended_context import ExtendedContext
from loguru import logger
from rpg_game import RPGGame 
from auxiliary.player_proxy import PlayerProxy, get_player_proxy, TEST_PLAYER_NAME
from auxiliary.player_input_command import (
                          PlayerCommandAttack, 
                          PlayerCommandLeaveFor, 
                          PlayerCommandBroadcast, 
                          PlayerCommandSpeak, 
                          PlayerCommandWhisper, 
                          PlayerCommandSearch,
                          PlayerCommandPrisonBreak,
                          PlayerCommandPerception,
                          PlayerCommandSteal,
                          PlayerCommandTrade,
                          PlayerCommandCheckStatus)

from auxiliary.extended_context import ExtendedContext

############################################################################################################
def splitcommand(input_val: str, split_str: str)-> str:
    if split_str in input_val:
        return input_val.split(split_str)[1].strip()
    return input_val
############################################################################################################
class TestPlayerInputSystem(ExecuteProcessor):
    def __init__(self, context: ExtendedContext, rpggame: 'RPGGame') -> None:
        self.context: ExtendedContext = context
        self.rpggame = rpggame
############################################################################################################
    def execute(self) -> None:
        logger.debug("<<<<<<<<<<<<<  PlayerInputSystem  >>>>>>>>>>>>>>>>>")
        while True:
            playername = self.current_input_player()
            playerproxy = get_player_proxy(playername)
            if playerproxy is None:
                logger.warning("玩家不存在，或者玩家未加入游戏")
                break
            usrinput = input(f"[{playername}]:")
            rpggame = self.rpggame
            self.handle_player_input(rpggame, playerproxy, usrinput)
            logger.debug(f"{'=' * 50}")
            break
############################################################################################################
    #测试的先写死
    def current_input_player(self) -> str:
        return TEST_PLAYER_NAME
############################################################################################################
    def handle_player_input(self, rpggame: RPGGame, playerproxy: PlayerProxy, usrinput: str) -> None:
        
        assert playerproxy is not None
        assert rpggame is not None

        if not rpggame.started:
            logger.warning("请先/run")
            return
        
        if "/attack" in usrinput:
            command = "/attack"
            targetname = splitcommand(usrinput, command)           
            PlayerCommandAttack(command, rpggame, playerproxy, targetname).execute()
                        
        elif "/leave" in usrinput:
            command = "/leave"
            stagename = splitcommand(usrinput, command)
            PlayerCommandLeaveFor(command, rpggame, playerproxy, stagename).execute()
  
        elif "/broadcast" in usrinput:
            command = "/broadcast"
            content = splitcommand(usrinput, command)
            PlayerCommandBroadcast(command, rpggame, playerproxy, content).execute()
            
        elif "/speak" in usrinput:
            command = "/speak"
            content = splitcommand(usrinput, command)
            PlayerCommandSpeak(command, rpggame, playerproxy, content).execute()

        elif "/whisper" in usrinput:
            command = "/whisper"
            content = splitcommand(usrinput, command)
            PlayerCommandWhisper(command, rpggame,playerproxy, content).execute()

        elif "/search" in usrinput:
            command = "/search"
            propname = splitcommand(usrinput, command)
            PlayerCommandSearch(command, rpggame, playerproxy, propname).execute()

        elif "/prisonbreak" in usrinput:
            command = "/prisonbreak"
            PlayerCommandPrisonBreak(command, rpggame, playerproxy).execute()

        elif "/perception" in usrinput:
            command = "/perception"
            PlayerCommandPerception(command, rpggame, playerproxy).execute()
            logger.debug(f"{'=' * 50}")

        elif "/steal" in usrinput:
            command = "/steal"
            propname = splitcommand(usrinput, command)
            PlayerCommandSteal(command, rpggame, playerproxy, propname).execute()

        elif "/trade" in usrinput:
            command = "/trade"
            propname = splitcommand(usrinput, command)
            PlayerCommandTrade(command, rpggame, playerproxy, propname).execute()

        elif "/checkstatus" in usrinput:
            command = "/checkstatus"
            PlayerCommandCheckStatus(command, rpggame, playerproxy).execute()
############################################################################################################
