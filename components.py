
from collections import namedtuple

###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
WorldComponent = namedtuple('WorldComponent', 'name agent')
StageComponent = namedtuple('StageComponent', 'name agent events')
NPCComponent = namedtuple('NPCComponent', 'name agent current_stage')
###############################################################################################################################################

###############################################################################################################################################
SpeakActionComponent = namedtuple('SpeakActionComponent', 'action')
FightActionComponent = namedtuple('FightActionComponent', 'action')
LeaveActionComponent = namedtuple('LeaveActionComponent', 'action')

