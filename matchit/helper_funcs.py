def df_cell_color(val):
    color = 'black'
    if val < 0:
        color = 'red'
    elif val > 0:
        color = 'green'
    return f'color: {color}'