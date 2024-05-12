import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
import os
from loguru import logger
import json
from typing import List, Dict, Any
from budding_world.excel_data import ExcelDataNPC, ExcelDataStage, ExcelDataProp
from budding_world.npc_editor import ExcelEditorNPC
from budding_world.stage_editor import ExcelEditorStage
from budding_world.utils import serialization_prop

EDITOR_WORLD_TYPE = "World"
EDITOR_PLAYER_TYPE = "Player"
EDITOR_NPC_TYPE = "NPC"
EDITOR_STAGE_TYPE = "Stage"

################################################################################################################
class ExcelEditorWorld:
    def __init__(self, worldname: str, data: List[Any], npc_data_base: Dict[str, ExcelDataNPC], prop_data_base: Dict[str, ExcelDataProp], stage_data_base: Dict[str, ExcelDataStage]) -> None:
        # 根数据
        self.name: str = worldname
        self.data: List[Any] = data
        self.npc_data_base = npc_data_base
        self.prop_data_base = prop_data_base
        self.stage_data_base = stage_data_base

        #笨一点，先留着吧。。。
        self.worlds: List[Any] = []
        self.players: List[Any] = []
        self.npcs: List[Any] = []
        self.stages: List[Any] = []

        #真正的构建数据
        self.editor_worlds: List[ExcelEditorNPC] = []
        self.editor_players: List[ExcelEditorNPC] = []
        self.editor_npcs: List[ExcelEditorNPC] = []
        self.editor_stages: List[ExcelEditorStage] = []
        self.editor_props: List[ExcelDataProp] = []

        ##把数据分类
        self.classify_data(self.worlds, self.players, self.npcs, self.stages)
        ##根据分类各种处理。。。
        self.editor_worlds = self.create_worlds(self.worlds)
        self.editor_players = self.create_players(self.players)
        self.editor_npcs = self.create_npcs(self.npcs)
        self.editor_stages = self.create_stages(self.stages)

        ##提取全部的道具。
        self.editor_props = self.parse_props_from_npc(self.editor_worlds) + self.parse_props_from_npc(self.editor_players) + self.parse_props_from_npc(self.editor_npcs) + self.parse_props_from_stage(self.editor_stages)
        logger.debug(f"World: {self.name} has {len(self.editor_props)} props.")


    def parse_props_from_npc(self, npcs: List[ExcelEditorNPC]) -> List[ExcelDataProp]:
        res = []
        for npc in npcs:
            for prop in npc.excelprops:
                if prop not in res:
                    res.append(prop)
        return res
    
    def parse_props_from_stage(self, stages: List[ExcelEditorStage]) -> List[ExcelDataProp]:
        res = []
        for stage in stages:

            for prop in stage.props_in_stage:
                if prop not in res:
                    res.append(prop)
            
            for prop in stage.interactive_props:
                if prop not in res:
                    res.append(prop)

        return res

    #先将数据分类
    def classify_data(self, out_worlds: List[Any], out_players: List[Any], out_npcs: List[Any], out_stages: List[Any]) -> None:
        #
        out_worlds.clear()
        out_players.clear()
        out_npcs.clear()
        out_stages.clear()
        #
        for item in self.data:
            if item["type"] == EDITOR_WORLD_TYPE:
                out_worlds.append(item)
            elif item["type"] == EDITOR_PLAYER_TYPE:
                out_players.append(item)
            elif item["type"] == EDITOR_NPC_TYPE:
                out_npcs.append(item)
            elif item["type"] == EDITOR_STAGE_TYPE:
                out_stages.append(item)

    def create_worlds(self, worlds: List[Any]) -> List[ExcelEditorNPC]:
        return self.create_npcs(worlds)

    def create_players(self, players: List[Any]) -> List[ExcelEditorNPC]:
        return self.create_npcs(players)
       
    def create_npcs(self, npcs: List[Any]) -> List[ExcelEditorNPC]:
        res: List[ExcelEditorNPC] = []
        for item in npcs:
            editor_npc = ExcelEditorNPC(item, self.npc_data_base, self.prop_data_base)
            res.append(editor_npc)
        return res

    def create_stages(self, stages: List[Any]) -> List[ExcelEditorStage]:
        res: List[ExcelEditorStage] = []
        for item in stages:
            editor_stage = ExcelEditorStage(item, self.npc_data_base, self.prop_data_base, self.stage_data_base)
            res.append(editor_stage)
        return res

    #最后生成JSON
    def serialization(self) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        output["worlds"] = [editor_npc.serialization() for editor_npc in self.editor_worlds]
        output["players"] = [editor_npc.serialization() for editor_npc in self.editor_players]
        output["npcs"] = [editor_npc.serialization() for editor_npc in self.editor_npcs]
        output["stages"] = [editor_stage.serialization() for editor_stage in self.editor_stages]
        output["database"] = self.data_base()

        version_sign = input("请输入版本号:")
        if version_sign == "":
            version_sign = "ewan"
            logger.warning(f"使用默认的版本号: {version_sign}")
        
        output["version"] = version_sign
        return output
    
    def data_base(self) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        npc_data_base = self.editor_worlds + self.editor_players + self.editor_npcs
        output["npcs"] = [data.serialization() for data in npc_data_base]
        output["stages"] = [data.serialization() for data in self.editor_stages]
        output["props"] = []
        for prop in self.editor_props:
            output["props"].append(serialization_prop(prop))
        return output
    
    def write(self, directory: str) -> bool:
        builddata = self.serialization()    
        logger.warning(builddata)
        builddata_json = json.dumps(builddata, indent=4, ensure_ascii = False)
        try:
            filename = f"{self.name}.json"
            path = os.path.join(directory, filename)
            # 确保目录存在
            os.makedirs(directory, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as file:
                file.write(builddata_json)
                return True
        except Exception as e:
            logger.error(f"An error occurred: {e}") 
        return False

