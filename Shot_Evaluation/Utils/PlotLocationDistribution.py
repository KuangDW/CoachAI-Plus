from .CoordinatesProcess import transform_coordinates
from .DrawCourt import plot_location_court

def plot_location_distribution(df_Real, df_Output, player_name):
    df_Real = df_Real.apply(transform_coordinates, axis=1, args=(player_name,))
    plot_location_court(df_Real, df_Output, player_name)
