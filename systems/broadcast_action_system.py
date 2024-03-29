
from entitas import Entity, Matcher, ReactiveProcessor, GroupEvent # type: ignore
from auxiliary.components import BroadcastActionComponent, StageComponent
from auxiliary.actor_action import ActorAction
from auxiliary.extended_context import ExtendedContext
from auxiliary.print_in_color import Color
from auxiliary.prompt_maker import broadcast_action_prompt
from typing import Optional
from loguru import logger

class BroadcastActionSystem(ReactiveProcessor):

    def __init__(self, context: ExtendedContext) -> None:
        super().__init__(context)
        self.context = context

    def get_trigger(self) -> dict[Matcher, GroupEvent]:
        return {Matcher(BroadcastActionComponent): GroupEvent.ADDED}

    def filter(self, entity: Entity) -> bool:
        return entity.has(BroadcastActionComponent)

    def react(self, entities: list[Entity]) -> None:
        logger.debug("<<<<<<<<<<<<<  BroadcastActionSystem  >>>>>>>>>>>>>>>>>")

        for entity in entities:
            self.handle(entity)  # 核心处理

        for entity in entities:
            entity.remove(BroadcastActionComponent)  # 必须移除！！！       

    def handle(self, entity: Entity) -> None:
        broadcastcomp: BroadcastActionComponent = entity.get(BroadcastActionComponent)
        stagecomp: Optional[StageComponent] = self.context.get_stagecomponent_by_uncertain_entity(entity) 
        if stagecomp is None or broadcastcomp is None:
            logger.error(f"BroadcastActionSystem: stagecomp or broadcastcomp is None!")
            return
        action: ActorAction = broadcastcomp.action
        for value in action.values:
            broadcast_say = broadcast_action_prompt(action.name, stagecomp.name, value, self.context)
            logger.info(f"{Color.HEADER}{broadcast_say}{Color.ENDC}")
            stagecomp.directorscripts.append(broadcast_say)
            

            
        
                