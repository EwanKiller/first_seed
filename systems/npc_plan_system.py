

from entitas import Entity, Matcher, ExecuteProcessor #type: ignore
from auxiliary.components import (NPCComponent, 
                        FightActionComponent, 
                        SpeakActionComponent, 
                        LeaveForActionComponent, 
                        TagActionComponent, 
                        HumanInterferenceComponent,
                        MindVoiceActionComponent,
                        BroadcastActionComponent, 
                        WhisperActionComponent,
                        HumanInterferenceComponent, 
                        SearchActionComponent,
                        PlayerComponent)
from auxiliary.actor_action import ActorPlan
from auxiliary.prompt_maker import npc_plan_prompt
from auxiliary.extended_context import ExtendedContext
from loguru import logger

class NPCPlanSystem(ExecuteProcessor):
    """
    This class represents a system for handling NPC plans.

    Attributes:
    - context: The context in which the system operates.

    Methods:
    - __init__(self, context): Initializes the NPCPlanSystem object.
    - execute(self): Executes the NPC plan system.
    - handle(self, entity): Handles the plan for a specific NPC entity.
    """

    def __init__(self, context: ExtendedContext) -> None:
        """
        Initializes the NPCPlanSystem object.

        Parameters:
        - context: The context in which the system operates.
        """
        self.context = context

    def execute(self) -> None:
        """
        Executes the NPC plan system.
        """
        logger.debug("<<<<<<<<<<<<<  NPCPlanSystem  >>>>>>>>>>>>>>>>>")
        entities = self.context.get_group(Matcher(NPCComponent)).entities
        for entity in entities:
            if entity.has(HumanInterferenceComponent):
                entity.remove(HumanInterferenceComponent)
                logger.info(f"{entity.get(NPCComponent).name}本轮行为计划被人类接管。\n")
                continue
            if entity.has(PlayerComponent):
                logger.info(f"{entity.get(NPCComponent).name}正在被玩家控制，不执行自动计划。\n")
                continue

            #开始处理NPC的行为计划
            self.handle(entity)

    def handle(self, entity: Entity) -> None:
        """
        Handles the plan for a specific NPC entity.

        Parameters:
        - entity: The NPC entity to handle the plan for.
        """
        prompt = npc_plan_prompt(entity, self.context)
        comp = entity.get(NPCComponent)
        try:
            response = comp.agent.request(prompt)
            if response is None:
                logger.warning("Agent request is None.如果不是默认Player可能需要检查配置。")
                return
            actorplan = ActorPlan(comp.name, response)
            for action in actorplan.actions:
                if len(action.values) == 0:
                    continue
                match action.actionname:
                    case "FightActionComponent":
                        if not entity.has(FightActionComponent):
                            entity.add(FightActionComponent, action)

                    case "LeaveForActionComponent":
                        if not entity.has(LeaveForActionComponent):
                            entity.add(LeaveForActionComponent, action)

                    case "SpeakActionComponent":
                        if not entity.has(SpeakActionComponent):
                            entity.add(SpeakActionComponent, action)
                    
                    case "TagActionComponent":
                        if not entity.has(TagActionComponent):
                            entity.add(TagActionComponent, action)
                    
                    case "RememberActionComponent":
                        pass

                    case "MindVoiceActionComponent":
                        if not entity.has(MindVoiceActionComponent):
                            entity.add(MindVoiceActionComponent, action)

                    case "BroadcastActionComponent":
                        if not entity.has(BroadcastActionComponent):
                            entity.add(BroadcastActionComponent, action)

                    case "WhisperActionComponent":
                        if not entity.has(WhisperActionComponent):
                            entity.add(WhisperActionComponent, action)

                    case "SearchActionComponent":
                        if not entity.has(SearchActionComponent):
                            entity.add(SearchActionComponent, action)
                    case _:
                        logger.warning(f" {action.actionname}, Unknown action name")
                        continue

        except Exception as e:
            logger.exception(f"NPCPlanSystem: {e}")  
            return
        return
