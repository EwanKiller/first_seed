
from entitas import ExecuteProcessor, Matcher #type: ignore
from auxiliary.extended_context import ExtendedContext
from loguru import logger
from auxiliary.components import stage_available_actions_register, npc_available_actions_register, StageComponent, NPCComponent
   
class PostActionSystem(ExecuteProcessor):
############################################################################################################
    def __init__(self, context: ExtendedContext) -> None:
        self.context: ExtendedContext = context
############################################################################################################
    def execute(self) -> None:
        logger.debug("<<<<<<<<<<<<<  PostActionSystem  >>>>>>>>>>>>>>>>>")
        # 在这里清除所有的行动
        self.remove_npc_actions()
        self.remove_stage_actions()
        self.test()
############################################################################################################
    def remove_stage_actions(self) -> None:
        entities = self.context.get_group(Matcher(all_of = [StageComponent], any_of = stage_available_actions_register)).entities
        for entity in entities:
            for actionsclass in stage_available_actions_register:
                if entity.has(actionsclass):
                    entity.remove(actionsclass)
############################################################################################################
    def remove_npc_actions(self) -> None:
        entities = self.context.get_group(Matcher(all_of = [NPCComponent], any_of = npc_available_actions_register)).entities
        for entity in entities:
            for actionsclass in npc_available_actions_register:
                if entity.has(actionsclass):
                    entity.remove(actionsclass)
############################################################################################################
    def test(self) -> None:
        stageentities = self.context.get_group(Matcher(any_of = stage_available_actions_register)).entities
        assert len(stageentities) == 0, f"Stage entities with actions: {stageentities}"
        npcentities = self.context.get_group(Matcher(any_of = npc_available_actions_register)).entities
        assert len(npcentities) == 0, f"NPC entities with actions: {npcentities}"
############################################################################################################

            

