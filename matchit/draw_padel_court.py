import drawsvg as dw
from typing import List, Optional

def svg_padel_court(
        team1:Optional[List[str]] = None,
        team2:Optional[List[str]]=None,
        team1_score:int=0,
        team2_score:int=0,
        court_name:Optional[str]=None
    ) -> dw.Drawing:
    add_height = 20 if court_name else 0
    d = dw.Drawing(width=210,height=add_height + 110)

    # Background
    back_rec = dw.Rectangle(x=0,y=0,width=210,height=add_height + 110,fill='#2476C2')
    d.append(back_rec)

    # Add padel court title
    if court_name:
        d.append(
            dw.Text(f"Court: {court_name}",16,210/2,20/2,center=True,fill='white')
        )

    # Draw Padel Court
    rec = dw.Rectangle(x=5,y=add_height+5,width=200,height=100,fill='#2476C2',stroke='white', stroke_width=2)
    rec.append_title('Padel court') # Tooltip
    d.append(rec)

    # Vertical service lines
    for i in [35,175]:
        d.append(
            dw.Line(i,add_height + 5,i,add_height + 105,stroke='white',stroke_width=2)
        )

    # Center horizontal line
    d.append(
        dw.Line(35,add_height + 110/2,175,add_height + 110/2,stroke='white',stroke_width=2)
    )

    for i in [210/2 - 1, 210/2 + 1.5]:
        d.append(
            dw.Line(i,add_height + 5,i,add_height + 105,stroke='white',stroke_width=1)
        )

    # Add player names
    if team1 and team2:
        for n,i in zip(team1,[5 + add_height + 100/4,5 + add_height + 100/2 + 100/4]):
            d.append(
                dw.Text(n,8,5+200/4,i,fill='white',center=True)
            )

        for n,i in zip(team2,[5 + add_height + 100/4,5 + add_height + 100/2 + 100/4]):
            d.append(
                dw.Text(n,8,5+200/2+200/4,i,fill='white',center=True)
            )
    
    # Add scores
    x_scores = [5+200/4 + 25, 5 + 200/2 + 200/4 - 25]
    for i in x_scores:
        rec_score = dw.Rectangle(x=i-10,y=5+add_height + 100/2-10,width=20,height=20,fill='white',center=True)
        d.append(rec_score)
    
    if team1_score > 0 or team2_score > 0:
        if team1_score > team2_score:
            team1_color = 'green'
            team2_color = 'red'
        elif team2_score > team1_score:
            team1_color = 'red'
            team2_color = 'green'
        else:
            team1_color = team2_color = 'gray'

        for s,c, i in zip([team1_score,team2_score],[team1_color,team2_color],x_scores):
            rec_score = dw.Rectangle(x=i-10,y=5+add_height+100/2-10,width=20,height=20,fill=c,center=True)
            d.append(rec_score)
            text_score = dw.Text(str(s),12,i,5+add_height+100/2,fill='white',center=True)
            d.append(text_score)
    
    return d