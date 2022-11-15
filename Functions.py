import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots

def show_data(data, what=None):
    if what is None:
        what = ['head', 'shapes', 'col_types', 'nans', 'stats', 'unique_vals']
    
    if 'head' in what:
        print('Head:')
        display(data.head())
        
    if 'shapes' in what:
        print('\nNrows:', data.shape[0])
        print('Ncols:', data.shape[1])
        print('rows/cols ratio:', data.shape[0]/data.shape[1], "\n")
        
    if 'col_types' in what:
        print('Col types:')
        display(pd.DataFrame(data.dtypes).transpose())
        print('Number of integer columns:', len(data.dtypes[data.dtypes == 'int64']))
        print('Number of float columns:', len(data.dtypes[data.dtypes == 'float64']))
        print('Number of string columns:', len(data.dtypes[data.dtypes == 'object']), "\n")
        
    if 'nans' in what:
        print('NaNs:')
        display(pd.DataFrame(data.isna().sum()).transpose())     
        
    if 'stats' in what:
        print('\nStatistics:')
        display(data.describe())
        
    if 'unique_vals' in what:
        print('\nUnique values (object):\n')
        print(data.select_dtypes(['object']).apply(lambda x: x.unique()))
        print('\nUnique values (integer):\n')
        print(data.select_dtypes(['int64']).apply(lambda x: x.unique()))


def corr_heatmap(df):
    color_background = '#F5F5F5'
    color_gridlines = '#DCDCDC'
        
    fig = px.imshow(df.corr().round(3), text_auto=True, color_continuous_scale='RdBu_r')#'deep'
    fig.update_traces(opacity=0.8)
    fig.update_layout(
        coloraxis_showscale=False,
        paper_bgcolor=color_background,
        plot_bgcolor=color_background)
    fig.update_yaxes(gridcolor=color_gridlines, title='')
    fig.update_xaxes(linecolor=color_gridlines)
    return(fig)


def show_plots(df, y_name, vars_subset=None, what=None):
    
    y = df.loc[:, y_name].to_numpy()
    df = df.drop(columns=[y_name])
    
    # Presets:
    if vars_subset is None:
        vars_subset = df.columns.tolist()
        
    if what is None:
        what = ['str_dist', 'num_dist', 'str_dist_vs_y', 'num_dist_vs_y']
    
    color_background = '#F5F5F5'
    color_gridlines = '#DCDCDC'
    colors_in_use = [
        '#2C3E50', '#537EA2', '#858F84', '#42A593',
        '#873E23', '#CFD1A1', '#6A744F', '#BDBDC5',
        '#7EA253', '#EDB676', '#C26D40'
    ]+px.colors.qualitative.Safe
    
    dtypes_num = ['int64', 'int32', 'int16', 'float64', 'float32', 'float16']
    dtypes_str = ['object', 'category']
    vars_num = df.loc[:, vars_subset].select_dtypes(include=dtypes_num).columns
    vars_str = df.loc[:, vars_subset].select_dtypes(include=dtypes_str).columns
    
    # Lists
    str_dist_lst = [None]*len(vars_str)
    str_dist2_lst = []
    num_dist_lst = [None]*len(vars_num)
    num_scat_lst = []
    mix_violin_lst = []
    str_dist_vs_y_lst = []
    num_dist_vs_y_lst = []
    
    # For single categorical:
    if 'str_dist' in what:
        for i in range(0, len(vars_str)):
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df.loc[:, vars_str[i]],
                                       name=vars_str[i],
                                       showlegend=True))
            fig.update_traces(marker_color=colors_in_use[0],
                              marker_line_color='rgb(8,48,107)',
                              marker_line_width=1.5,
                              opacity=0.8)
            fig.update_layout(xaxis_type='category',
                              xaxis_title=vars_str[i],
                              paper_bgcolor=color_background,
                              plot_bgcolor=color_background)
            fig.update_yaxes(gridcolor=color_gridlines)
            fig.update_xaxes(linecolor=color_gridlines)
            str_dist_lst[i] = fig
        
    # For 2 categorical:
    if 'str_dist2' in what:
        for i in range(0, len(vars_str)):
            for j in range(0, len(vars_str)):
                if i == j:
                    continue
                else:
                    fig = px.histogram(df, x=vars_str[i], color=vars_str[j],
                                       color_discrete_sequence=colors_in_use[1:])
                    fig.update_traces(marker_line_color='rgb(8,48,107)',
                                      marker_line_width=1.5,
                                      opacity=0.8)
                    fig.update_layout(xaxis_type='category',
                                      xaxis_title=vars_str[i],
                                      paper_bgcolor=color_background,
                                      plot_bgcolor=color_background)
                    fig.update_yaxes(gridcolor=color_gridlines, title='')
                    fig.update_xaxes(linecolor=color_gridlines)
                    str_dist2_lst.append(fig)
            
    # For single numerical:
    if 'num_dist' in what:
        for i in range(0, len(vars_num)):
            fig2 = ff.create_distplot(
                hist_data=[df[vars_num[i]].dropna()],
                group_labels=[vars_num[i]],
                show_hist=True,
                show_rug=False
            )

            fig = make_subplots(rows=1, cols=3)
            fig.add_trace(go.Histogram(
                fig2['data'][0], marker_color=colors_in_use[0], marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5, opacity=0.8#, name=vars_num[i]
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                fig2['data'][1], marker_color=colors_in_use[0], marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5, opacity=0.8#, name=vars_num[i]
            ), row=1, col=2)
            fig.add_trace(go.Violin(
                y=df.loc[:, vars_num[i]], box_visible=True, meanline_visible=True,
                marker_color=colors_in_use[0], marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5, 
                opacity=0.8, name=vars_num[i]
            ), row=1, col=3)
            fig.update_layout(
                paper_bgcolor=color_background,
                plot_bgcolor=color_background,
                showlegend=False)
            fig.update_yaxes(
                gridcolor=color_gridlines,
                zerolinecolor=color_gridlines,
                title='')
            fig.update_xaxes(
                gridcolor=color_gridlines,
                title='')
            num_dist_lst[i] = fig

    # For 2 numeric:
    if 'num_scat' in what:
        for i in range(0, len(vars_num)):
            for j in range(0, len(vars_num)):
                if i == j:
                    continue
                else:
                    fig = px.scatter(df,
                                     x=df[vars_num[i]],
                                     y=df[vars_num[j]],
                                     trendline='ols')
                    fig.update_layout(paper_bgcolor=color_background,
                                      plot_bgcolor=color_background)
                    fig.update_traces(marker_color=colors_in_use[0],
                                      opacity=0.8)
                    fig.update_yaxes(gridcolor=color_gridlines,
                                     zerolinecolor=color_gridlines,
                                     title=vars_num[j])
                    fig.update_xaxes(gridcolor=color_gridlines,
                                     zerolinecolor=color_gridlines,
                                     title=vars_num[i])
                    num_scat_lst.append(fig)
        
    # 1 categorical, 1 numeric:
    if 'mix_violin' in what:
        for i in range(0, len(vars_str)):
            for j in range(0, len(vars_num)):
                fig = px.violin(df, y=vars_num[j], color=vars_str[i],
                                color_discrete_sequence=colors_in_use,
                                box=True, points='outliers')
                fig.update_traces(opacity=0.8)
                fig.update_layout(xaxis_title=vars_num[j],
                                  showlegend=True,
                                  paper_bgcolor=color_background,
                                  plot_bgcolor=color_background)
                fig.update_yaxes(gridcolor=color_gridlines, title='')
                fig.update_xaxes(linecolor=color_gridlines)
                mix_violin_lst.append(fig)
    
    if 'str_dist_vs_y' in what:
        if y.dtype in dtypes_num:
            for i in range(0, len(vars_str)):
                fig = px.violin(df, y=y, color=vars_str[i],
                                color_discrete_sequence=colors_in_use,
                                box=True, points='outliers')
                fig.update_traces(opacity=0.8)
                fig.update_layout(xaxis_title=y_name,
                                  showlegend=True,
                                  paper_bgcolor=color_background,
                                  plot_bgcolor=color_background)
                fig.update_yaxes(gridcolor=color_gridlines, title='')
                fig.update_xaxes(linecolor=color_gridlines)
                str_dist_vs_y_lst.append(fig)
        else:
            for i in range(0, len(vars_str)):
                fig = px.histogram(df, x=vars_str[i], color=y,
                                       color_discrete_sequence=colors_in_use[1:])
                fig.update_traces(marker_line_color='rgb(8,48,107)',
                                      marker_line_width=1.5,
                                      opacity=0.8)
                fig.update_layout(xaxis_type='category',
                                      xaxis_title=vars_str[i],
                                      paper_bgcolor=color_background,
                                      plot_bgcolor=color_background)
                fig.update_yaxes(gridcolor=color_gridlines, title='')
                fig.update_xaxes(linecolor=color_gridlines)
                str_dist_vs_y_lst.append(fig)
        
    if 'num_dist_vs_y' in what:
        if y.dtype in dtypes_num:
            for i in range(0, len(vars_num)):
                fig = px.scatter(df,
                                 x=df[vars_num[i]],
                                 y=y,
                                 trendline='ols')
                fig.update_layout(paper_bgcolor=color_background,
                                  plot_bgcolor=color_background)
                fig.update_traces(marker_color=colors_in_use[0],
                                  opacity=0.8)
                fig.update_yaxes(gridcolor=color_gridlines,
                                 zerolinecolor=color_gridlines,
                                 title=y_name)
                fig.update_xaxes(gridcolor=color_gridlines,
                                 zerolinecolor=color_gridlines,
                                 title=vars_num[i])
                num_dist_vs_y_lst.append(fig)
        else:
            for i in range(0, len(vars_num)):
                fig = px.violin(df, y=vars_str[i], color=y,
                                color_discrete_sequence=colors_in_use,
                                box=True, points='outliers')
                fig.update_traces(opacity=0.8)
                fig.update_layout(xaxis_title=y_name,
                                  showlegend=True,
                                  paper_bgcolor=color_background,
                                  plot_bgcolor=color_background)
                fig.update_yaxes(gridcolor=color_gridlines, title='')
                fig.update_xaxes(linecolor=color_gridlines)
                num_dist_vs_y_lst.append(fig)
    
    final_dict = dict(
        str_dist = str_dist_lst,
        str_dist2 = str_dist2_lst,
        num_dist = num_dist_lst,
        num_scat = num_scat_lst,
        mix_violin = mix_violin_lst,
        str_dist_vs_y = str_dist_vs_y_lst,
        num_dist_vs_y = num_dist_vs_y_lst
    )
    return(final_dict)