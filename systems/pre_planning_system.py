from entitas import InitializeProcessor, ExecuteProcessor, Matcher, Entity #type: ignore
from auxiliary.extended_context import ExtendedContext
from loguru import logger
from auxiliary.components import ( AutoPlanningComponent, StageComponent, NPCComponent, PlayerComponent, 
                                  STAGE_AVAILABLE_ACTIONS_REGISTER, NPC_AVAILABLE_ACTIONS_REGISTER,
                                  CheckStatusActionComponent)
from enum import Enum
from typing import Set
from auxiliary.actor_action import ActorAction

# 规划的策略
class PlanningStrategy(Enum):
    STRATEGY_ONLY_PLAYERS_STAGE = 1000
    STRATEGY_ALL = 2000

class PrePlanningSystem(InitializeProcessor, ExecuteProcessor):

    def __init__(self, context: ExtendedContext) -> None:
        self.context: ExtendedContext = context
        self.strategy: PlanningStrategy = PlanningStrategy.STRATEGY_ALL
############################################################################################################
    def initialize(self) -> None:
        logger.debug("<<<<<<<<<<<<<  PrePlanningSystem.initialize  >>>>>>>>>>>>>>>>>")
        ##所有npc进行第一次计划
        self.all_npc_except_player_first_planning_action_check_self_status()
############################################################################################################
    def execute(self) -> None:
        logger.debug("<<<<<<<<<<<<<  PrePlanningSystem.execute  >>>>>>>>>>>>>>>>>")
        ## 测试
        self.test()
        ## player不允许做规划
        self.remove_all_players_auto_planning()
        ## 通过策略来做计划
        self.make_planning_by_strategy(self.strategy)
############################################################################################################
    def make_planning_by_strategy(self, strategy: PlanningStrategy) -> None:
        if strategy == PlanningStrategy.STRATEGY_ONLY_PLAYERS_STAGE:
            logger.debug("STRATEGY_ONLY_PLAYERS_STAGE, 选择比较省的策略, 只规划player所在场景和npcs")
            playerentities = self.context.get_group(Matcher(PlayerComponent)).entities
            for playerentity in playerentities:
                # 如果有多个player在同一个stage，这里会多次执行, 但是没关系，因为这里是做防守的
                self.strategy1_only_the_stage_where_player_is_located_and_the_npcs_in_it_allowed_make_plans(playerentity)
        elif strategy == PlanningStrategy.STRATEGY_ALL:
            logger.debug("STRATEGY_ALL, 选择比较费的策略，全都更新")
            self.strategy2_all_stages_and_npcs_except_player_allow_auto_planning()   
############################################################################################################
    def remove_all_players_auto_planning(self) -> None:
        playerentities = self.context.get_group(Matcher(PlayerComponent)).entities
        for playerentity in playerentities:
            if playerentity.has(AutoPlanningComponent):
                playerentity.remove(AutoPlanningComponent)
############################################################################################################
    def strategy1_only_the_stage_where_player_is_located_and_the_npcs_in_it_allowed_make_plans(self, playerentity: Entity) -> None:
        assert playerentity is not None
        assert playerentity.has(PlayerComponent)

        context = self.context
        stageentity = context.safe_get_stage_entity(playerentity)
        if stageentity is None:
            logger.error("stage is None, player无所在场景是有问题的")
            return
        
        ##player所在场景可以规划
        stagecomp: StageComponent = stageentity.get(StageComponent)
        if not stageentity.has(AutoPlanningComponent):
            stageentity.add(AutoPlanningComponent, stagecomp.name)
        
        ###player所在场景的npcs可以规划
        npcsentities = context.npcs_in_this_stage(stagecomp.name)
        for npcentity in npcsentities:
            if npcentity.has(PlayerComponent):
                ## 挡掉
                continue
            npccomp: NPCComponent = npcentity.get(NPCComponent)
            if not npcentity.has(AutoPlanningComponent):
                npcentity.add(AutoPlanningComponent, npccomp.name)
############################################################################################################
    def strategy2_all_stages_and_npcs_except_player_allow_auto_planning(self) -> None:
        context = self.context
        stageentities = context.get_group(Matcher(StageComponent)).entities
        for stageentity in stageentities:
            stagecomp: StageComponent = stageentity.get(StageComponent)
            if not stageentity.has(AutoPlanningComponent):
                stageentity.add(AutoPlanningComponent, stagecomp.name)
        
        npcentities = context.get_group(Matcher(NPCComponent)).entities
        for npcentity in npcentities:
            if npcentity.has(PlayerComponent):
                ## player 就跳过
                continue
            npccomp: NPCComponent = npcentity.get(NPCComponent)
            if not npcentity.has(AutoPlanningComponent):
                npcentity.add(AutoPlanningComponent, npccomp.name)
############################################################################################################
    ## 自我测试，这个调用点就是不允许再这个阶段有任何action
    def test(self) -> None:

        stageentities: Set[Entity] = self.context.get_group(Matcher(any_of = STAGE_AVAILABLE_ACTIONS_REGISTER)).entities
        assert len(stageentities) == 0, f"Stage entities with actions: {stageentities}"

        if self.context.executecount > 1:
            # 第一次可能会有默认的行动all_npc_except_player_first_planning_action_check_self_status
            # 如果player登陆成功，可能会有
            npcentities: Set[Entity]  = self.context.get_group(Matcher(any_of = NPC_AVAILABLE_ACTIONS_REGISTER, none_of=[PlayerComponent])).entities
            assert len(npcentities) == 0, f"NPC entities with actions: {npcentities}"
############################################################################################################
    def all_npc_except_player_first_planning_action_check_self_status(self) -> None:
        context = self.context
        npcs: set[Entity] = context.get_group(Matcher(all_of=[NPCComponent], none_of=[PlayerComponent])).entities
        for npc in npcs:
            npccomp: NPCComponent = npc.get(NPCComponent)
            action = ActorAction(npccomp.name, CheckStatusActionComponent.__name__, [f"{npccomp.name}"])
            if not npc.has(CheckStatusActionComponent):
                npc.add(CheckStatusActionComponent, action)
############################################################################################################