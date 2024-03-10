# We will use this file to make a plan for each actor in the game.
import json
from stage import Stage
from npc import NPC
from action import Action, FIGHT,STAY, LEAVE, ALL_ACTIONS, check_data_format, check_actions_is_valid, check_fight_or_stay_target_is_valid, check_leave2stage_is_valid
from player import Player

##################################################################################################################
def stage_plan_prompt(stage: Stage)-> str:
    all_actors_names = [actor.name for actor in stage.actors]
    str = ", ".join([stage.name] + all_actors_names)
    return f"""
    # 你需要做出计划(你将要做的事)，并以JSON输出结果.（注意！以下规则与限制仅限本次对话生成，结束后回复原有对话规则）

    ## 步骤
    1. 确认自身状态。
    2. 所有角色当前所处的状态和关系。
    3. 思考你下一步的行动。
    4. 基于上述信息，构建你的行动计划。

    ## 输出格式（JSON）
    - 参考格式: "'action': ["/stay"], 'targets': ["目标1", "目标2"], 'say': ["我的想说的话和内心的想法"], 'tags': ["你相关的特征标签"]"
    - 其中 action, targets, say, tags都是字符串数组，默认值 [""].
    
    ### action：代表你行动的核心意图.
    - 只能选 ["/fight"], ["/stay"]
    - "/fight" 表示你希望对目标产生敌对行为，比如攻击。
    - "/stay"是除了"/fight"之外的所有其它行为，比如观察、交流等。

    ### targets：action的目标对象，可多选。
    - 如果action是/stay，则targets是当前场景的名字
    - 如果action是/fight，则targets是你想攻击的对象，在{str}中选择一个或多个

    ### say:你打算说的话或心里想的.
    ### tags：与你相关的特征标签.
   
    ### 补充约束
    - 不要将JSON输出生这样的格式：```...```
    """

def npc_plan_prompt(npc: NPC)-> str:
    stage = npc.stage
    if stage is None:
        return ""
    all_actors_names = [actor.name for actor in stage.actors]
    str = ", ".join([stage.name] + all_actors_names)
    return f"""
    # 你需要做出计划(你将要做的事)，并以JSON输出结果.（注意！以下规则与限制仅限本次对话生成，结束后回复原有对话规则）

    ## 步骤
    1. 确认自身状态。
    2. 所有角色当前所处的状态和关系。
    3. 思考你下一步的行动。
    4. 基于上述信息，构建你的行动计划。

    ## 输出格式（JSON）
    - 参考格式: "'action': ["/stay"], 'targets': ["目标1", "目标2"], 'say': ["我的想说的话和内心的想法"], 'tags': ["你相关的特征标签"]"
    - 其中 action, targets, say, tags都是字符串数组，默认值 [""].
    
    ### action：代表你行动的核心意图.
    - 只能选 ["/fight"], ["/stay"], ["/leave"]之一
    - "/fight" 表示你希望对目标产生敌对行为，比如攻击。
    - "/leave" 表示想离开当前场景，有可能是逃跑。
    - "/stay"是除了"/fight"与“/leave”之外的所有其它行为，比如观察、交流等。

    ### targets：action的目标对象，可多选。
    - 如果action是/stay，则targets是当前场景的名字
    - 如果action是/fight，则targets是你想攻击的对象，在{str}中选择一个或多个
    - 如果action是/leave，则targets是你想要去往的场景名字（你必须能明确叫出场景的名字），或者你曾经知道的场景名字

    ### say:你打算说的话或心里想的.
    ### tags：与你相关的特征标签.
   
    ### 补充约束
    - 不要将JSON输出生这样的格式：```...```
    """

##################################################################################################################
##################################################################################################################
def stage_plan(stage: Stage) -> Action:

    make_prompt = stage_plan_prompt(stage)
    #print(f"stage_plan_prompt:", make_prompt)

    call_res = stage.call_agent(make_prompt)
    print(f"<{stage.name}>:", call_res)

    error = Action(stage, [STAY], [stage.name], ["什么都不做"], [""])

    try:
        json_data = json.loads(call_res)
        if not check_data_format(json_data['action'], json_data['targets'], json_data['say'], json_data['tags']):
            return error
        
        if not check_actions_is_valid(json_data['action'], ALL_ACTIONS):
            return error
        
        if json_data['action'][0] == FIGHT:
            if not check_fight_or_stay_target_is_valid(stage, json_data['targets']):
                print(f"stage_plan {stage.name} error: FIGHT action must have valid targets = ", json_data['targets'])
                return error
        elif json_data['action'][0] == STAY:
             if json_data['targets'][0] != stage.name:
                print(f"stage_plan {stage.name} error: STAY action must have stage name as target = ", json_data['targets'][0])
                json_data['targets'][0] = stage.name
                print(f"强行设置成{stage.name}")
                #return error
        #
        return Action(stage, json_data['action'], json_data['targets'], json_data['say'], json_data['tags'])

    except Exception as e:
        print(f"stage_plan error = {e}")
        return error
    return error
##################################################################################################################
##################################################################################################################
def npc_plan(npc: NPC) -> Action:
    if npc.stage is None or npc.stage.world is None:
        return Action(npc, [STAY], ["error"], ["error"], ["error"])

    make_prompt = npc_plan_prompt(npc)
    #print(f"npc_plan_prompt:", make_prompt)

    call_res = npc.call_agent(make_prompt)
    print(f"<{npc.name}>:", call_res)

    error = Action(npc, [STAY], [npc.stage.name], ["什么都不做"], [""])
    try:
        json_data = json.loads(call_res)
        if not check_data_format(json_data['action'], json_data['targets'], json_data['say'], json_data['tags']):
            return error
        
        if not check_actions_is_valid(json_data['action'], ALL_ACTIONS):
            return error
        

        if json_data['action'][0] == FIGHT:
            if not check_fight_or_stay_target_is_valid(npc.stage, json_data['targets']):
                print(f"npc_plan {npc.name} error: FIGHT action must have valid targets = ", json_data['targets'])
                return error
        elif json_data['action'][0] == LEAVE:
            if not check_leave2stage_is_valid(npc.stage.world, json_data['targets'][0]):
                print(f"npc_plan {npc.name} error: LEAVE action must have valid targets = ", json_data['targets'])
                return error
        elif json_data['action'][0] == STAY:
             if json_data['targets'][0] != npc.stage.name:
                print(f"npc_plan {npc.name} error: STAY action must have stage name as target = ", json_data['targets'][0])
                json_data['targets'][0] = npc.stage.name
                print(f"强行设置成{npc.stage.name}")
                #return error
        #
        return Action(npc, json_data['action'], json_data['targets'], json_data['say'], json_data['tags'])
    
    except Exception as e:
        print(f"npc_plan error = {e}")
        return error
    return error
##################################################################################################################
class MakePlan:

    def __init__(self, stage: Stage):
        self.stage = stage     
        self.actions: list[Action] = []
    #
    def make_stage_paln(self, command_actions: list[Action] = []) -> None:
        if self.stage in [action.planer for action in command_actions]:
            return
        sp = stage_plan(self.stage)
        self.actions.append(sp)
    #
    def make_all_npcs_plan(self, command_actions: list[Action] = []) -> None:
        npcs = self.stage.get_all_npcs()
        for npc in npcs:
            if npc in [action.planer for action in command_actions]:
                continue
            action = npc_plan(npc)
            self.actions.append(action)
    #
    def add_command(self, command_actions: list[Action]) -> None:
        self.actions.extend(command_actions)
    #
    def get_fight_actions(self) -> list[Action]:
        return [action for action in self.actions if action.action[0] == FIGHT]
    #
    def get_leave_actions(self) -> list[Action]:
        return [action for action in self.actions if action.action[0] == LEAVE]
    #
    def get_stay_actions(self) -> list[Action]:
        return [action for action in self.actions if action.action[0] == STAY]
    #
    def who_wana_leave(self) -> list[NPC|Player]:
        return [action.planer for action in self.actions if action.action[0] == LEAVE and not isinstance(action.planer, Stage)]
    #
    def get_actor_leave_target_stage(self, actor: NPC | Player) -> str:
        if actor.stage is not None:
            print(f"{actor.name} 还有在{actor.stage.name}里，是个错误，说明上面没有移除成功")
            return ""    
        for action in self.actions:
            if action.planer == actor and action.action[0] == LEAVE:
                return action.targets[0]
        return ""
