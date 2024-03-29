
from entitas import Entity, Matcher, ReactiveProcessor, GroupEvent # type: ignore
from auxiliary.components import SpeakActionComponent, NPCComponent, StageComponent
from auxiliary.actor_action import ActorAction
from auxiliary.extended_context import ExtendedContext
from auxiliary.print_in_color import Color
from auxiliary.prompt_maker import speak_action_prompt
from typing import Optional
from loguru import logger
from auxiliary.dialogue_rule import check_speak_enable, parse_taget_and_message
   
####################################################################################################
class SpeakActionSystem(ReactiveProcessor):

    def __init__(self, context: ExtendedContext) -> None:
        super().__init__(context)
        self.context = context

    def get_trigger(self) -> dict[Matcher, GroupEvent]:
        return {Matcher(SpeakActionComponent): GroupEvent.ADDED}

    def filter(self, entity: Entity) -> bool:
        return entity.has(SpeakActionComponent)

    def react(self, entities: list[Entity]) -> None:
        logger.debug("<<<<<<<<<<<<<  SpeakActionSystem  >>>>>>>>>>>>>>>>>")
        # 核心执行
        for entity in entities:
            self.handle(entity)  
        # 必须移除！！！
        for entity in entities:
            entity.remove(SpeakActionComponent)     
####################################################################################################
    def handle(self, entity: Entity) -> None:
        speak_comp: SpeakActionComponent = entity.get(SpeakActionComponent)
        speak_action: ActorAction = speak_comp.action
        for value in speak_action.values:
            tagret_message_pair = parse_taget_and_message(value)
            target: str = tagret_message_pair[0]
            message: str = tagret_message_pair[1]
            ##如果检查不过就能继续
            if not check_speak_enable(self.context, entity, target):
                continue
            ##拼接说话内容
            say_content: str = speak_action_prompt(speak_action.name, target, message, self.context)
            logger.info(f"{Color.HEADER}{say_content}{Color.ENDC}")
            ##添加场景事件，最后随着导演剧本走
            self.context.add_content_to_director_script_by_entity(entity, say_content)

       