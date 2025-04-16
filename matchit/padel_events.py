from pydantic import BaseModel, computed_field, field_validator, Field
from typing import Union, List, Dict, Literal, Optional, Tuple
import pandas as pd
import random
from .padel_match import Match, Round
from .padel_player import Player

def new_padel_round(player_list:List[Player],round_no:int=1,**kwargs) -> Round:
    total_players = len(player_list)
    overflow = total_players % 4
        
    sitover_idx_start = (round_no - 1) * overflow % total_players if overflow else 0
    sitovers = []

    if overflow:
        sitovers = player_list[sitover_idx_start:sitover_idx_start+overflow]
        if len(sitovers) < overflow:
            sitovers += player_list[0:overflow - len(sitovers)]
    
    # Remaining players for match generation
    active_players = [p for p in player_list if p not in sitovers]
    matches = []

    for i in range(0,len(active_players),4):
        match_players = active_players[i:i+4]
        matches.append(
            Match.from_player_list(
                player_list=match_players,round_no=round_no,tournament_name=kwargs.get('tournament_name','Padel Event'),game_type=kwargs.get('game_type','Padel')
            )
        )
    
    return Round(round_no=round_no,matches=matches,sitovers=sitovers)

class EventPlayer(Player):
    points: int = 0

    def __repr__(self) -> str:
        repr_str = super().__repr__().split('\n')
        if self.points:
            repr_str.append(f"Points: {self.points}")
        return "\n".join(repr_str)

class Event(BaseModel):
    event_name: str
    round: int = 0
    rounds: Union[List[Round],List] = []
    play_by: Literal['points','win'] = 'points'
    player_list: List[EventPlayer] = Field(min_length=4)
    round_sitovers: Union[Dict,Dict[int, List[EventPlayer]]] = {}
    
    @computed_field
    def event_type(self) -> str:
        return 'Padel Event'

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

    def update_player_list(self,method:Literal['round','points','seeding','win']='round'):
        if method == 'round':
            self.player_list = self.current_round.player_list if self.rounds else self.player_list
        elif method == 'points': 
            # self.player_list = self.update_player_list(method='round')
            self.player_list.sort(key=lambda p: getattr(p,'points'),reverse=True)
        elif method == 'seeding':
            self.player_list.sort(key=lambda p: getattr(p,'seeding_score'),reverse=True)
        else:   # wins
            self.player_list.sort(key=lambda p: getattr(p,'win'),reverse=True)

    def randomize_new_round(self,round_no:int=1,**kwargs) -> None:
        random.shuffle(self.players_list)
        new_round = Round.from_player_list(self.players_list,round_no=round_no,**kwargs)
        self.rounds.append(new_round)
    
    def next_round(self):
        self.update_player_scores()
        self.update_player_list(method='round')
        
        padel_round = new_padel_round(player_list=self.player_list,round_no=self.round,tournament_name=self.event_name,game_type='Padel')
        self.rounds.append(padel_round)
        # total_players = len(self.player_list)
        # overflow = total_players % 4
        
        # sitover_idx_start = (self.round - 1) * overflow % total_players if overflow else 0
        # sitovers = []

        # if overflow:
        #     sitovers = self.player_list[sitover_idx_start:sitover_idx_start+overflow]
        #     if len(sitovers) < overflow:
        #         sitovers += self.player_list[0:overflow - len(sitovers)]
        
        # # Remaining players for match generation
        # active_players = [p for p in self.player_list if p not in sitovers]
        # matches = []

        # for i in range(0,len(active_players),4):
        #     match_players = active_players[i:i+4]
        #     matches.append(
        #         Match.from_player_list(player_list=match_players,round_no=self.round,tournament_name=self.name,game_type=self.event_type)
        #     )

        # self.rounds.append(
        #     Round(round_no=self.round,matches=matches,sitovers=sitovers)
        # )
        # self.update_player_list(method='round')
    
    def create_n_rounds(self,n_rounds=4):
        pass

class Americano(Event):
    @computed_field
    def event_type(self) -> str:
        return 'Americano'

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
    @computed_field
    def event_type(self) -> str:
        return 'Mexicano'

    @field_validator('player_list')
    @classmethod
    def validate_player_count(cls,v:List[Player]) -> List[Player]:
        if not len(v) % 4 == 0:
            raise ValueError('A Mexicano tournament event must have the number of players divisible by 4.')
        return v

    def next_round(self):
        self.update_player_scores()
        self.update_player_list(method='points')
        self.round = len(self.rounds) + 1 if self.rounds else 1

        matches = []
        for i in range(0,len(self.player_list),4):
            group = self.player_list[i:i+4]

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
