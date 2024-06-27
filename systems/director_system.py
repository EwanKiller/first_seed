from typing import override
from entitas import Entity, Matcher, ExecuteProcessor #type: ignore
from auxiliary.components import StageComponent, ActorComponent, PlayerComponent
from my_entitas.extended_context import ExtendedContext
from loguru import logger
from auxiliary.director_component import StageDirectorComponent
from auxiliary.player_proxy import add_client_actor_message
from builtin_prompt.cn_builtin_prompt import stage_director_begin_prompt, stage_director_end_prompt, stage_director_event_wrap_prompt

class DirectorSystem(ExecuteProcessor):

    def __init__(self, context: ExtendedContext) -> None:
        self.context = context
###################################################################################################################
    @override
    def execute(self) -> None:
        self.handle()
        self.director_clear()
###################################################################################################################
    def handle(self) -> None:
        entities = self.context.get_group(Matcher(all_of=[StageComponent, StageDirectorComponent])).entities
        for entity in entities:
            logger.debug('=' *50)
            self.handle_stage(entity)
            logger.debug('=' *50)
            self.handle_actors_in_this_stage(entity)
            logger.debug('=' *50)
###################################################################################################################   
    def director_clear(self) -> None:
        entities = self.context.get_group(Matcher(all_of=[StageComponent, StageDirectorComponent])).entities
        for entity in entities:
            directorcomp: StageDirectorComponent = entity.get(StageDirectorComponent)
            directorcomp.clear()
###################################################################################################################
    def handle_stage(self, entitystage: Entity) -> None:
        assert entitystage.has(StageComponent)
        stagecomp: StageComponent = entitystage.get(StageComponent)
        directorcomp: StageDirectorComponent = entitystage.get(StageDirectorComponent)
        events2stage = directorcomp.to_stage(stagecomp.name, self.context)  
        for event in events2stage:
            logger.debug(f"director:{stagecomp.name}:{event}")
            self.context.safe_add_human_message_to_entity(entitystage, event)       
###################################################################################################################
    def handle_actors_in_this_stage(self, entitystage: Entity) -> None:
        assert entitystage.has(StageComponent)
        stagecomp: StageComponent = entitystage.get(StageComponent)
        actors_int_this_stage = self.context.actors_in_stage(stagecomp.name)
        for _entity in actors_int_this_stage:
            director_events_to_actor(self.context, _entity)
###################################################################################################################
def director_events_to_actor(context: ExtendedContext, actor_entity: Entity) -> None:
    stage_entity = context.safe_get_stage_entity(actor_entity)
    if stage_entity is None:
        return
    stage_director_comp: StageDirectorComponent = stage_entity.get(StageDirectorComponent)
    assert stage_director_comp is not None

     ### 添加消息！
    actor_comp: ActorComponent = actor_entity.get(ActorComponent)

    events2actor = stage_director_comp.to_actor(actor_comp.name, context)    
    if len(events2actor) == 0:
        return

    ### 标记开始
    context.safe_add_human_message_to_entity(actor_entity, stage_director_begin_prompt(stage_director_comp.name, len(events2actor)))

    for index, event in enumerate(events2actor):
        wrap_prompt = stage_director_event_wrap_prompt(event, index)
        logger.debug(f"director:{actor_comp.name}:{event}")
        context.safe_add_human_message_to_entity(actor_entity, wrap_prompt)

    ## 标记结束
    context.safe_add_human_message_to_entity(actor_entity, stage_director_end_prompt(stage_director_comp.name, len(events2actor)))

    # 通知客户端显示
    if actor_entity.has(PlayerComponent):
        events2player = stage_director_comp.to_player(actor_comp.name, context)
        for event in events2player:
            add_client_actor_message(actor_entity, event)
###################################################################################################################
    