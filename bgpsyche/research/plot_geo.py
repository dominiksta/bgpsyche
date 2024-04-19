import typing as t

import geopandas as gpd
import pandas as pd
from bgpsyche.util.const import HERE

def plot_val_by_country(
        mpl_axis: t.Any,
        iso22value: t.Dict[str, float],
        geopandas_plot_kw = {},
        axis_off = True
):
    countries_gdf = gpd.read_file(HERE / 'research' / 'ne_50m_admin_0_countries.geojson')
    df = pd.DataFrame([
        { 'iso_a2': iso2, 'val': val } for iso2, val in iso22value.items()
    ])
    to_plot = countries_gdf.merge(df, on='iso_a2')
    to_plot.plot(
        column='val', cmap='magma_r', ax=mpl_axis, **geopandas_plot_kw,
    )
    if axis_off: mpl_axis.set_axis_off()
