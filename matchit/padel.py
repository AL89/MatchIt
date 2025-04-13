#%%
from pydantic import BaseModel, Field, computed_field, model_validator, field_validator
from typing import Optional, Union, List, Tuple, Dict, Any, Literal
import pandas as pd
from numpy.random import choice
from itertools import combinations
from functools import cached_property
import random

class Player(BaseModel):
    name: str
    win: int = 0
    draw: int = 0
    loss: int = 0
    points: int = 0

def __eq__(self, other:"Player"):
    return self.name == other.name

def __str__(self):
    return self.name

def __repr__(self):
    repr_str = [str(self)]
    if self.win:
        repr_str.append(f"Wins: {self.win}")
    if self.loss:
        repr_str.append(f"Loss: {self.loss}")
    if self.draw:
        repr_str.append(f"Draw: {self.draw}")
    if self.points:
        repr_str.append(f"Points: {self.points}")
    return "\n".join(repr_str)

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
    sit_overs: Optional[List[Player]] = None
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

class Tournament(BaseModel):
    id: int = 0
    name: str
    player_list: List[Player] = Field(min_length=4)

class Event(Tournament):
    round: int = 0
    rounds: Union[List[Round],List] = []
    play_by: Literal['points','win'] = 'points'
    # _teammate_history: Union[Dict,Dict[str,List[str]]] = {}
    round_sitovers: Union[Dict,Dict[int, List[Player]]] = {}
    event_type: str = 'Padel Event'
    
    @computed_field
    def max_rounds(self) -> int:
        return len(self.rounds) - 1

    @computed_field
    def current_round(self) -> Optional[Round]:
        return self.rounds[-1] if self.rounds else None

    # @computed_field
    def standings(self,sort_by:Literal['points','win']='points',return_type:str='dataframe') -> Union[pd.DataFrame,Dict[str,Dict]]:
        if not self.rounds:
            return None
        sort_by_options = ['points','win','draw','loss']
        sort_by = [sort_by] + [sb for sb in sort_by_options if sb != sort_by]
        self.update_player_scores()
        # self.update_player_list(method=sort_by)
        player_standings = [player.model_dump() for player in self.player_list]
        df = pd.DataFrame.from_records(player_standings).sort_values(by=sort_by,ascending=False)
        df = df.rename(columns={c:c.capitalize() for c in df.columns})
        df = df.set_index('Name')
        if return_type == 'dataframe':
            return df
        else:   # dict
            return df.to_dict('index')
        # return player_standings
    
    def update_player_scores(self):
        for player in self.player_list:
            player.points = 0
            player.win = 0
            player.loss = 0
            player.draw = 0

        for round in self.rounds:
            for m in round.matches:
                for player in m.team1:
                    player.points += m.team1_score
                for player in m.team2:
                    player.points += m.team2_score
                
                if not m.winner:
                    for player in m.team1 + m.team2:
                        player.draw += 1
                else:
                    for player in m.winner:
                        player.win += 1
                    for player in m.loser:
                        player.loss += 1

    def update_player_list(self,method:Literal['round','points','win']='round'):
        if method == 'round':
            self.player_list = self.current_round.player_list if self.rounds else self.player_list
        elif method == 'points': 
            # self.player_list = self.update_player_list(method='round')
            self.player_list.sort(key=lambda p: getattr(p,'points'),reverse=True)
        else:   # wins
            self.player_list.sort(key=lambda p: getattr(p,'win'),reverse=True)

    def randomize_new_round(self,round_no:int=1,**kwargs) -> None:
        players = self.players_list #.copy()
        random.shuffle(players)
        new_round = Round.from_player_list(players,round_no=round_no,**kwargs)
        self.rounds.append(new_round)
    
    def next_round(self):
        self.update_player_scores()
        self.update_player_list(method='round')
        
        total_players = len(self.player_list)
        full_match_count = (total_players // 4) * 4
        overflow = total_players % 4
        
        sitover_idx_start = (self.round - 1) * overflow % total_players if overflow else 0
        sitovers = []

        if overflow:
            sitovers = self.player_list[sitover_idx_start:sitover_idx_start+overflow]
            if len(sitovers) < overflow:
                sitovers += self.player_list[0:overflow - len(sitovers)]
        
        # Remaining players for match generation
        active_players = [p for p in self.player_list if p not in sitovers]
        matches = []

        for i in range(0,len(active_players),4):
            match_players = active_players[i:i+4]
            matches.append(
                Match.from_player_list(player_list=match_players,round_no=self.round,tournament_name=self.name,game_type=self.event_type)
            )

        self.rounds.append(
            Round(round_no=self.round,matches=matches)
        )
        self.update_player_list(method='round')
        
class Americano(Event):
    event_type: str = 'Americano'

    @field_validator('player_list')
    @classmethod
    def validate_player_count(cls,v:List[Player]) -> List[Player]:
        if not len(v) % 4 == 0:
            raise ValueError('An Americano tournament event must have the number of players divisible by 4.')
        return v

    def next_round(self):
        self.update_player_scores()
        self.update_player_list(method='round')
        player_list = self.player_list
        round_player_list = []
        self.round = len(self.rounds) + 1 if self.rounds else 1

        matches = []
        while len(player_list) >= 4:
            combination = [player_list[0]] + player_list[2:4] + [player_list[1]]
            if self.rounds:
                randomize_count = 0
                while any([r.is_combination(combination) for r in self.rounds]):
                    random.shuffle(player_list)
                    combination = [player_list[0]] + player_list[2:4] + [player_list[1]]

                    if self.max_rounds + 1 <= self.round: # len(self.players):
                        randomize_count += 1
                    
                    if randomize_count == 3:       # 
                        break

            team1 = combination[:2]
            team2 = combination[2:]
            matches.append(
                Match(padel_round=self.round,team1=team1,team2=team2,tournament_name=self.name,game_type=self.game_type)
                        # Match.from_player_list(player_list=list(combination),round_no=self.round,tournament_name=self.name,game_type='Americano')
            )
            for p in combination:
                player_list.remove(p)
                round_player_list.append(p)

        self.rounds.append(
            Round(round_no=self.round,matches=matches)
        )
        self.update_player_list(method='round')

class Mexicano(Event):
    event_type: str = 'Mexicano'

    @field_validator('player_list')
    @classmethod
    def validate_player_count(cls,v:List[Player]) -> List[Player]:
        if not len(v) % 4 == 0:
            raise ValueError('A Mexicano tournament event must have the number of players divisible by 4.')
        return v

    def next_round(self):
        self.update_player_scores()
        self.update_player_list(method='points')
        player_list = self.player_list
        self.round = len(self.rounds) + 1 if self.rounds else 1

        matches = []
        for i in range(0,len(player_list),4):
            group = player_list[i:i+4]

            # Form teams: Best + worst vs 2nd + 3rd
            team1 = [group[0],group[-1]]
            team2 = [group[1],group[2]]

            matches.append(
                Match(padel_round=self.round,team1=team1,team2=team2,tournament_name=self.name,game_type=self.event_type)
                        # Match.from_player_list(player_list=list(combination),round_no=self.round,tournament_name=self.name,game_type='Americano')
            )

        self.rounds.append(
            Round(round_no=self.round,matches=matches)
        )
        self.update_player_list(method='points')

# %%


if __name__ == '__main__':
    # Sample players
    players = [Player(name=name) for name in ['A','B','C','D','E','F','G','H']]

    mexicano_event = Mexicano(id=1,name='Americano Event',player_list=players)
    mexicano_event.next_round()

    # player1 = Player(id=1,name='Andreas')
    # player2 = Player(id=2,name='Cindy')
    # player3 = Player(id=3,name='Michael')
    # player4 = Player(id=4,name='Tijana')
    # team1 = [player1,player2]
    # team2 = [player3,player4]
    # match_p = Match(id=1,team1=team1,team2=team2)
    # match_p.scores.team1_score = 17
    # match_p.scores.team2_score = 15

# %%
