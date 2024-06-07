from loguru import logger
import os
from typing import Dict, List, Optional
from auxiliary.file_def import PropFile, ActorArchiveFile, StageArchiveFile

class FileSystem:

    def __init__(self, name: str) -> None:
        self.name = name
        self.rootpath = ""
        # 拥有的道具
        self.propfiles: Dict[str, List[PropFile]] = {}
        # 知晓的NPC 
        self.actor_archives: Dict[str, List[ActorArchiveFile]] = {}
        # 知晓的Stage
        self.stage_archives: Dict[str, List[StageArchiveFile]] = {}
################################################################################################################
    ### 必须设置根部的执行路行
    def set_root_path(self, rootpath: str) -> None:
        if self.rootpath != "":
            raise Exception(f"[filesystem]已经设置了根路径，不能重复设置。")
        self.rootpath = rootpath
        #logger.debug(f"[filesystem]设置了根路径为{rootpath}")
################################################################################################################
    ### 删除
    def delete_file(self, filepath: str) -> None:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.error(f"[{filepath}]的文件删除失败。")
            return
################################################################################################################
    def write_file(self, filepath: str, content: str) -> None:
        try:
            if not os.path.exists(filepath):
                os.makedirs(os.path.dirname(filepath), exist_ok=True) # 没有就先创建一个！
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        except FileNotFoundError as e:
            logger.error(f"[{filepath}]的文件不存在。")
            return
        except Exception as e:
            logger.error(f"[{filepath}]的文件写入失败。")
            return
################################################################################################################
    def prop_file_path(self, ownersname: str, filename: str) -> str:
        return f"{self.rootpath}{ownersname}/props/{filename}.json"
################################################################################################################
    ## 写一个道具的文件
    def write_prop_file(self, propfile: PropFile) -> None:
        self.delete_file(self.prop_file_path(propfile.ownersname, propfile.name))
        content = propfile.content()
        self.write_file(self.prop_file_path(propfile.ownersname, propfile.name), content)
################################################################################################################
    ## 添加一个道具文件
    def add_prop_file(self, propfile: PropFile) -> None:
        self.propfiles.setdefault(propfile.ownersname, []).append(propfile)
        self.write_prop_file(propfile)
################################################################################################################
    def get_prop_files(self, ownersname: str) -> List[PropFile]:
        return self.propfiles.get(ownersname, [])
################################################################################################################
    def get_prop_file(self, ownersname: str, propname: str) -> Optional[PropFile]:
        propfiles = self.get_prop_files(ownersname)
        for file in propfiles:
            if file.name == propname:
                return file
        return None
################################################################################################################
    def has_prop_file(self, ownersname: str, propname: str) -> bool:
        return self.get_prop_file(ownersname, propname) is not None
################################################################################################################
    def exchange_prop_file(self, from_owner: str, to_owner: str, propname: str) -> None:
        findownersfile = self.get_prop_file(from_owner, propname)
        if findownersfile is None:
            logger.error(f"{from_owner}没有{propname}这个道具。")
            return
        # 文件得从管理数据结构中移除掉
        self.propfiles[from_owner].remove(findownersfile)
        # 文件重新写入
        self.delete_file(self.prop_file_path(from_owner, propname))
        self.add_prop_file(PropFile(propname, to_owner, findownersfile.prop))
################################################################################################################
    def actor_archive_path(self, ownersname: str, filename: str) -> str:
        return f"{self.rootpath}{ownersname}/npcs/{filename}.json"
################################################################################################################
    ## 添加一个你知道的NPC
    def add_actor_archive(self, npcarchive: ActorArchiveFile) -> Optional[ActorArchiveFile]:
        files = self.actor_archives.setdefault(npcarchive.ownersname, [])
        for file in files:
            if file.npcname == npcarchive.npcname:
                # 名字匹配，先返回，不添加。后续可以复杂一些
                return None
        files.append(npcarchive)
        return npcarchive
################################################################################################################
    def has_actor_archive(self, ownersname: str, npcname: str) -> bool:
        return self.get_actor_archive(ownersname, npcname) is not None
################################################################################################################
    def get_actor_archive(self, ownersname: str, npcname: str) -> Optional[ActorArchiveFile]:
        files = self.actor_archives.get(ownersname, [])
        for file in files:
            if file.npcname == npcname:
                return file
        return None
################################################################################################################
    ## 写一个道具的文件
    def write_actor_archive(self, npcarchive: ActorArchiveFile) -> None:
        ## 测试
        self.delete_file(self.actor_archive_path(npcarchive.ownersname, npcarchive.name))
        content = npcarchive.content()
        self.write_file(self.actor_archive_path(npcarchive.ownersname, npcarchive.name), content)
################################################################################################################
    def stage_archive_path(self, ownersname: str, filename: str) -> str:
        return f"{self.rootpath}{ownersname}/stages/{filename}.json"
################################################################################################################
    def add_stage_archive(self, stage_archive: StageArchiveFile) -> None:
        files = self.stage_archives.setdefault(stage_archive.ownersname, [])
        for file in files:
            if file.stagename == stage_archive.stagename:
                # 名字匹配，先返回，不添加。后续可以复杂一些
                return
        files.append(stage_archive)
################################################################################################################
    def write_stage_archive(self, known_stage_file: StageArchiveFile) -> None:
        ## 测试
        self.delete_file(self.stage_archive_path(known_stage_file.ownersname, known_stage_file.name))
        content = known_stage_file.content()
        self.write_file(self.stage_archive_path(known_stage_file.ownersname, known_stage_file.name), content)
################################################################################################################
    def get_stage_archive(self, ownersname: str, stagename: str) -> Optional[StageArchiveFile]:
        stagelist = self.stage_archives.get(ownersname, [])
        for file in stagelist:
            if file.stagename == stagename:
                return file
        return None
################################################################################################################
    def has_stage_archive(self, ownersname: str, stagename: str) -> bool:
        return self.get_stage_archive(ownersname, stagename) is not None
################################################################################################################