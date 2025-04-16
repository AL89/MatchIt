from pydantic import BaseModel, Field, computed_field
from typing import List, Optional, Union, Tuple
from functools import cached_property
from .padel_player import Player

class Match(BaseModel,validate_assignment=True):
    team1: List[Player] = Field(min_length=2,max_length=2)
    team2: List[Player] = Field(min_length=2,max_length=2)
    team1_score: int = 0
    team2_score: int = 0
    tournament_name: Optional[str] = None
    game_type: Optional[str] = None
    padel_round: int = 1

    @computed_field
    def winner(self) -> Optional[List[Player]]:
        if self.team1_score > self.team2_score:
            winner_team = self.team1
        elif self.team2_score > self.team1_score:
            winner_team = self.team2
        else:
            winner_team = None
        return winner_team
    
    @computed_field
    def loser(self) -> Union[List[Player],None]:
        winner_team = self.winner
        # if winner is None:
        #     return None
        if not winner_team:
            return False
        return self.team1 if winner_team == self.team2 else self.team2

    @computed_field
    def result(self) -> Tuple[int]:
        return (self.team1_score,self.team2_score)

    def __repr__(self):
        repr_str = [
            f"{' and '.join([p.name for p in self.team1])}: {self.team1_score}",
            f"{' and '.join([p.name for p in self.team2])}: {self.team2_score}",
        ]
        return "\n".join(repr_str)
    
    @classmethod
    def from_player_list(cls,player_list:List[Player],round_no:int=1,**kwargs):
        return cls(
            team1 = player_list[:2],team2=player_list[2:],
            tournament_name=kwargs.get('tournament_name',None),game_type=kwargs.get('game_type',None),padel_round=round_no
        )

class Round(BaseModel,validate_assignment=True):
    round_no: int = 1
    matches: List[Match]
    sitovers: Optional[List[Player]] = None
    # team_pairings = List[Tuple[Player]]

    @computed_field
    @cached_property
    def player_list(self) -> List[Player]:
        players = []
        for m in self.matches:
            players += m.team1 + m.team2
        return players
    
    @computed_field
    @cached_property
    def team_pairings(self) -> List[Tuple[Player]]:
        players = self.player_list.copy()
        pairings = []
        for i in range(1,len(players),2):
            pair = (players[i-1],players[i])
            pairings.append(pair)
        return pairings
    
    # @field_validator('matches',mode='before')
    # @classmethod
    # def update_matches(cls,v:List[Match]) -> List[Match]:
    #     for m in v:
    #         m.update_player_scores()
    #     return v

    def is_equal_player_list(self,other_player_list:List[Player]) -> bool:
        return self.player_list == other_player_list

    def is_pair(self,team_pair:Tuple[Player]) -> bool:
        for pair in self.team_pairings:
            if pair[0] in team_pair and pair[1] in team_pair:
                return True
        return False
        # return team_pair in self.team_pairings

    def is_combination(self,combination:Tuple[Player]):
        pair1 = combination[:2]
        pair2 = combination[2:]
        all_combs = [self.is_pair(pair1),self.is_pair(pair2)]
        # all_combs = [self.player_list[i]==combination[i] for i in range(len(combination))]
        return any(all_combs)

    @classmethod
    def from_player_list(cls,player_list:List[Player],round_no:int=1,**kwargs):
        if len(player_list) < 4:
            raise ValueError('Cannot create matches. There must be at least 4 players')
        
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        
        player_list = list(chunks(player_list,4))
        matches = []
        sit_overs = None
        for m in player_list:
            if len(m) < 4:
                sit_overs = m
                break

            matches.append(
                Match(
                    team1 = m[:2],team2 = m[2:],
                    tournament_name=kwargs.get('tournament_name','Padel Event'),
                    game_type=kwargs.get('game_type','Match'),
                    padel_round=round_no
                )
            )
        return cls(round_no=round_no,matches=matches,sit_overs=sit_overs)

def __str__(self):
    return f"Round {self.round_no}"

def __repr__(self):
    repr_str = [str(self)]
    repr_str.append(f"Matches: {len(self.matches)}")
    if self.sit_overs:
        repr_str(f"Sit overs: {len(self.sit_overs)}")
    return "\n".join(repr_str) 