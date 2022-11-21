import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots

def show_data(df, what=None):
    """Shows dataset's descriptive statistics and other important characteristics. 
    
    Parameters
    ----------
        df : dataset,
        what : which type of information to show,
               default=['head', 'shapes', 'col_types', 'nans', 'stats', 'unique_vals'].
               Available options are:
               - 'head' : shows 5 first rows of the dataset,
               - 'shapes' : shows number of rows, columns and rows/columns ratio,
               - 'col_types' : shows pandas columns types,
               - 'nans' : shows number of NaNs for each column. Does not count Inf, -Inf,
               - 'stats' : shows basic statistics ( from data.describe() ),
               - 'unique_vals' : unique categories for every column in the dataset
                                 (restricted by the print length).
        
    Returns
    ----------
        According to the 'what' argument.
    """
    
    if what is None:
        what = ['head', 'shapes', 'col_types', 'nans', 'stats', 'unique_vals']
    
    if 'head' in what:
        print('Head:')
        display(df.head())
        
    if 'shapes' in what:
        print('\nNrows:', df.shape[0])
        print('Ncols:', df.shape[1])
        print('rows/cols ratio:', df.shape[0]/df.shape[1], "\n")
        
    if 'col_types' in what:
        print('Col types:')
        display(pd.DataFrame(df.dtypes).transpose())
        print('Number of integer columns:', len(df.dtypes[df.dtypes == 'int64']))
        print('Number of float columns:', len(df.dtypes[df.dtypes == 'float64']))
        print('Number of string columns:', len(df.dtypes[df.dtypes == 'object']), "\n")
        
    if 'nans' in what:
        print('NaNs:')
        display(pd.DataFrame(df.isna().sum()).transpose())     
        
    if 'stats' in what:
        print('\nStatistics:')
        display(df.describe())
        
    if 'unique_vals' in what:
        print('\nUnique values (object):\n')
        print(df.select_dtypes(['object']).apply(lambda x: x.unique()))
        print('\nUnique values (integer):\n')
        print(df.select_dtypes(['int64']).apply(lambda x: x.unique()))

def corr_heatmap(df):
    """Visualizes correlation matrix of all numeric variables.
    
    Parameters
    ----------
        df : dataset.
        
    Returns
    ----------
        Heatmap from correlation matrix.
    """
    
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

def quantitative_eda(df, stim_vec, imp_vec):
    """Visualizes a summary of Quantitative EDA.
    
    Parameters
    ----------
        df : dataset,
        stim_vec : list of stimulant type of each variable in dataset. Accepted types are:
                   ['d', 's', 'm', 'n'] - (di)stimulant, mixed, none/explainable,
        stim_vec : list of importance type of each variable in dataset. Accepted types are:
                   ['h', 'm', 'l', 'n'] - high, medium, low, none/explainable.
        
    Returns
    ----------
        Barplot distribution of predefined stimulants,
        Barplot distribution of predefined importance,
        DataFrame with information about each column.
    """
    
    
    color_background = '#F5F5F5'
    color_gridlines = '#DCDCDC'
    colors_in_use = [
        '#2C3E50', '#537EA2', '#858F84', '#42A593',
        '#873E23', '#CFD1A1', '#6A744F', '#BDBDC5',
        '#7EA253', '#EDB676', '#C26D40'
    ]+px.colors.qualitative.Safe

    data_business_eda = pd.DataFrame(dict(
        cols = df.columns.tolist(),
        stim = stim_vec,
        imp = imp_vec
    ))
    
    data_business_eda['imp'] = data_business_eda['imp'].replace(
        {'h':'high', 'l':'low', 'm':'medium', 'n':'none/explainable'}
    )
    data_business_eda['stim'] = data_business_eda['stim'].replace(
        {'d':'distimulant', 's':'stimulant', 'm':'mixed', 'n':'none/explainable'}
    )

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data_business_eda['imp']))
    fig.update_traces(
            marker_color=colors_in_use,
            marker_line_width=1.5,
            opacity=0.8
    )
    fig.update_layout(
            xaxis_type='category',
            xaxis_title='Variable importance',
            paper_bgcolor=color_background,
            plot_bgcolor=color_background
    )
    fig.update_yaxes(gridcolor=color_gridlines)
    fig.update_xaxes(linecolor=color_gridlines)

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=data_business_eda['stim']))
    fig2.update_traces(
            marker_color=colors_in_use,
            marker_line_width=1.5,
            opacity=0.8
    )
    fig2.update_layout(
            xaxis_type='category',
            xaxis_title='Variable type',
            paper_bgcolor=color_background,
            plot_bgcolor=color_background
    )
    fig2.update_yaxes(gridcolor=color_gridlines)
    fig2.update_xaxes(linecolor=color_gridlines)
    
    fig.show()
    fig2.show()
    
    display(data_business_eda.transpose())

def show_plots(df, y_name, vars_subset=None, what=None):
    """Creates a dictionary of lists, where each list element is a single plot.
    
    Each element of dictionary represent a certain category of plots. Availablee options are:
        - 'cat_dist' : list of barplots for each categorical variable,
        - 'cat_dist2' : list of barplots for each unique combinations of 2 categorical variables,
        - 'num_dist' : list of violin+density+histogram plot for numerical each variable,
        - 'num_dist2' : list of scatterplots for each unique combinations of 2 numerical variables,
        - 'cat_dist_vs_y' : list of plots (bar or violin) for each categorical variable against explainable
                            variable,
        - 'num_dist_vs_y' : list of plots (scatter or violin) for each numerical variable against explainable
                            variable,
        - 'mix_violin' : list of violin plots for each unique combinations of 1 categorical and 1 numerical
                         variables.
        
        To show all the plots, use for loop:
        for i in plots['num_dist']:
            i.show()
    
    Parameters
    ----------
        df : dataset,
        y_name : explainable variable,
        vars_subset: selected variable(s) from the dataset besides explainable variable, default=None (all)
        what: which types of lists should be returned, 
              default=['str_dist', 'num_dist', 'str_dist_vs_y', 'num_dist_vs_y']. Available options are:
              ['str_dist', 'str_dist2', 'num_dist', 'num_dist2', 
               'str_dist_vs_y', 'num_dist_vs_y', 'mix_violin'].

    Returns
    ----------
        A dictionary of lists.
    """
    
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
                fig = px.violin(df, y=vars_num[i], color=y,
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


def show_plots_single(df, y_name, vars_subset=None,
                      ncols=3, h_space=0.02, p_height=1000, f_size=8):
    """Creates plots of variables' distribution and distribution against explainable variable.
    
    Available plots are:
    - 'cat_single' : all categorical columns on the same grid,
    - 'cat_split' : all categorical columns on different grids,
    - 'num_single' : all numerical columns on the same grid,
    - 'num_split' : all numerical columns on different grids,
    - 'cat_vs_y_single' : all categorical columns against explainable variable on the same grid,
    - 'cat_vs_y_split' : all categorical columns against explainable variable on different grids,
    - 'num_vs_y_single' : all numerical columns against explainable variable on the same grid,
    - 'num_vs_y_split' : all numerical columns against explainable variable on different grids.
    
    Parameters
    ----------
        df : dataset,
        y_name : explainable variable,
        vars_subset: selected variable(s) from the dataset besides explainable variable, default=None (all),
        ncols : number of plot's columns, default=3,
        h_space : spacing between subplots, default=0.02,
        p_height : height of single plot, default=1000,
        f_size : yaxis title's font size of single plot, default=8.

    Returns
    ----------
        A dictionary of plots.
    """
    
    y = df.loc[:, y_name].to_numpy()
    #df = df.drop(columns=[y_name])
    
    if vars_subset is None:
        vars_subset = df.columns.tolist()
        
    color_background = '#F5F5F5'
    color_gridlines = '#DCDCDC'
    colors_in_use = [
        '#2C3E50', '#537EA2', '#858F84', '#42A593',
        '#873E23', '#CFD1A1', '#6A744F', '#BDBDC5',
        '#7EA253', '#EDB676', '#C26D40'
    ]+(px.colors.qualitative.Safe+
       px.colors.qualitative.Pastel+
       px.colors.qualitative.Prism+
       px.colors.qualitative.Antique+
       px.colors.qualitative.Vivid+
       px.colors.qualitative.Plotly
      )
    
    dtypes_num = ['int64', 'int32', 'int16', 'float64', 'float32', 'float16']
    dtypes_str = ['object', 'category']
    vars_num = df.loc[:, vars_subset].select_dtypes(include=dtypes_num).columns
    vars_str = df.loc[:, vars_subset].select_dtypes(include=dtypes_str).columns
    
    
    # single plot, all categorical #########################################################
    fig = go.Figure()
    for i in range(0, len(vars_str)):
        fig.add_trace(go.Histogram(x=df.loc[:, vars_str[i]], 
                                   name=vars_str[i],
                                   marker=dict(color=colors_in_use[min(i, len(colors_in_use))]),
                                   showlegend=True))
        fig.update_layout(
            xaxis_type='category',
            xaxis_title='',
            paper_bgcolor=color_background,
            plot_bgcolor=color_background
        )
        
    fig.update_yaxes(gridcolor=color_gridlines)
    fig.update_xaxes(linecolor=color_gridlines)
    fig.update_traces(
        #marker_color=colors_in_use,
        marker_line_width=1.5,
        opacity=0.8
    )
    
    if len(fig['data']) % ncols == 0:
        dim_1 = int(len(fig['data'])/ncols)
    else:
        dim_1 = int(len(fig['data'])/ncols)+1
    dim_2 = ncols
    
    fig2 = make_subplots(rows=dim_1, cols=dim_2, horizontal_spacing=h_space)
    
    for i in range(dim_2):
        for j in range(dim_1):
            if j+(i*dim_1) >= len(fig['data']):
                continue
            fig2.append_trace(fig['data'][j+(i*dim_1)], j+1, i+1)
            fig2.update_yaxes(title_text=vars_str[j+(i*dim_1)], row=j+1, col=i+1)
            fig2.update_yaxes(title_font_size=f_size, row=j+1, col=i+1)
            fig2.update_yaxes(title_font_color='black', row=j+1, col=i+1)
        
    fig2.update_layout(
        xaxis_type='category',
        height=p_height,
        paper_bgcolor=color_background,
        plot_bgcolor=color_background#,
        #yaxis=dict(font=dict(size=6))
    )
    fig2.update_yaxes(gridcolor=color_gridlines, showticklabels=False)
    fig2.update_xaxes(linecolor=color_gridlines, showticklabels=False)
    
    
    # single plot, all numerical ############################################################
    fig3 = go.Figure()
    for i in range(0, len(vars_num)):
        fig3.add_trace(go.Violin(
            y=df.loc[:, vars_num[i]], box_visible=True, meanline_visible=True,
            marker=dict(color=colors_in_use[min(i, len(colors_in_use))]),
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, opacity=0.8, 
            name=vars_num[i], showlegend=True
        ))
        fig3.update_layout(
            xaxis_type='category',
            xaxis_title='',
            paper_bgcolor=color_background,
            plot_bgcolor=color_background
        )
        fig3.update_yaxes(gridcolor=color_gridlines)
        fig3.update_xaxes(linecolor=color_gridlines)
    
    
    if len(vars_num) % ncols == 0:
        dim_1 = int(len(fig3['data'])/ncols)
    else:
        dim_1 = int(len(fig3['data'])/ncols)+1
    dim_2 = ncols
    
    fig4 = make_subplots(rows=dim_1, cols=dim_2, horizontal_spacing=h_space)
    for i in range(dim_2):
        for j in range(dim_1):
            if j+(i*dim_1) >= len(fig3['data']):
                continue
            fig4.append_trace(fig3['data'][j+(i*dim_1)], j+1, i+1)
            fig4.update_yaxes(title_text=vars_num[j+(i*dim_1)], row=j+1, col=i+1)
            fig4.update_yaxes(title_font_size=f_size, row=j+1, col=i+1)
            fig4.update_yaxes(title_font_color='black', row=j+1, col=i+1)
        
    fig4.update_layout(
        paper_bgcolor=color_background,
        height=p_height,
        plot_bgcolor=color_background
    )
    fig4.update_yaxes(gridcolor=color_gridlines, showticklabels=False)
    fig4.update_xaxes(linecolor=color_gridlines, showticklabels=False)
    
    
    # all categorical against y ##############################################################
    fig5 = go.Figure()
    if y.dtype in dtypes_num:
        for i in range(0, len(vars_str)):
            fig5.add_trace(go.Violin(y=y, x=df.loc[:, vars_str[i]],
                                     box_visible=True, meanline_visible=True,
                                     name=vars_str[i],
                                     marker=dict(color=colors_in_use[min(i, len(colors_in_use))]),
                                     marker_line_width=1.5, opacity=0.8,
                                     marker_line_color='rgb(8,48,107)'))
            fig5.update_layout(xaxis_title=y_name,
                               showlegend=True,
                               paper_bgcolor=color_background,
                               plot_bgcolor=color_background)
            fig5.update_yaxes(gridcolor=color_gridlines, title='')
            fig5.update_xaxes(linecolor=color_gridlines)
    else:
        for i in range(0, len(vars_str)):
            fig5.add_trace(go.Histogram(
                y=y, x=df.loc[:, vars_str[i]], 
                name=vars_str[i], histfunc="count",
                marker=dict(color=colors_in_use[min(i, len(colors_in_use))]),
                showlegend=True, marker_line_width=1.5,
                opacity=0.8))
            fig5.update_layout(
                xaxis_type='category',
                xaxis_title='',
                paper_bgcolor=color_background,
                plot_bgcolor=color_background
            )
            fig5.update_yaxes(gridcolor=color_gridlines)
            fig5.update_xaxes(linecolor=color_gridlines)
    
    
    if len(fig5['data']) % ncols == 0:
        dim_1 = int(len(fig5['data'])/ncols)
    else:
        dim_1 = int(len(fig5['data'])/ncols)+1
    dim_2 = ncols
    
    fig6 = make_subplots(rows=dim_1, cols=dim_2)
    
    for i in range(dim_2):
        for j in range(dim_1):
            if j+(i*dim_1) >= len(fig5['data']):
                continue
            fig6.append_trace(fig5['data'][j+(i*dim_1)], j+1, i+1)
            fig6.update_yaxes(title_text=vars_str[j+(i*dim_1)], row=j+1, col=i+1)
            fig6.update_yaxes(title_font_size=f_size, row=j+1, col=i+1)
            fig6.update_yaxes(title_font_color='black', row=j+1, col=i+1)
        
    fig6.update_layout(
        xaxis_type='category',
        height=p_height,
        paper_bgcolor=color_background,
        plot_bgcolor=color_background
    )
    fig6.update_yaxes(gridcolor=color_gridlines, showticklabels=False)
    fig6.update_xaxes(linecolor=color_gridlines, showticklabels=False)
    
    # all numerical against y ###################################################################
    fig7 = go.Figure()
    if y.dtype in dtypes_num:
        for i in range(0, len(vars_num)):
            fig7.add_trace(go.Scatter(
                x=df[vars_num[i]], y=y,
                opacity=0.8, name=vars_num[i], mode='markers',
                marker=dict(color=colors_in_use[min(i, len(colors_in_use))]),
            ))
            fig7.update_layout(xaxis_title='',
                               yaxis_title=vars_num[i],
                               showlegend=True,
                               paper_bgcolor=color_background,
                               plot_bgcolor=color_background)
            fig7.update_yaxes(gridcolor=color_gridlines, title='')
            fig7.update_xaxes(linecolor=color_gridlines)
    else:
        for i in range(0, len(vars_num)):
            fig7.add_trace(go.Violin(x=y, y=df.loc[:, vars_num[i]],
                                     box_visible=True, meanline_visible=True,
                                     name=vars_str[i],
                                     marker=dict(color=colors_in_use[min(i, len(colors_in_use))]),
                                     marker_line_width=1.5, opacity=0.8,
                                     marker_line_color='rgb(8,48,107)'))
            fig7.update_layout(xaxis_title=y_name,
                               showlegend=True,
                               paper_bgcolor=color_background,
                               plot_bgcolor=color_background)
            fig7.update_yaxes(gridcolor=color_gridlines, title='')
            fig7.update_xaxes(linecolor=color_gridlines)
            
            
    if len(vars_num) % ncols == 0:
        dim_1 = int(len(fig7['data'])/ncols)
    else:
        dim_1 = int(len(fig7['data'])/ncols)+1
    dim_2 = ncols
    
    fig8 = make_subplots(rows=dim_1, cols=dim_2)
    for i in range(dim_2):
        for j in range(dim_1):
            if j+(i*dim_1) >= len(fig7['data']):
                continue
            fig8.append_trace(fig7['data'][j+(i*dim_1)], j+1, i+1)
            fig8.update_yaxes(title_text=vars_num[j+(i*dim_1)], row=j+1, col=i+1)
            fig8.update_yaxes(title_font_size=f_size, row=j+1, col=i+1)
            fig8.update_yaxes(title_font_color='black', row=j+1, col=i+1)
        
    fig8.update_layout(
        height=p_height,
        paper_bgcolor=color_background,
        plot_bgcolor=color_background
    )
    fig8.update_yaxes(gridcolor=color_gridlines, showticklabels=False)
    fig8.update_xaxes(linecolor=color_gridlines, showticklabels=False)

    
    final_dict = dict(
        cat_single = fig,
        cat_split = fig2,
        num_single = fig3,
        num_split = fig4,
        cat_vs_y_single = fig5,
        cat_vs_y_split = fig6,
        num_vs_y_single = fig7,
        num_vs_y_split = fig8
    )
    
    return final_dict

def show_outliers(df, n_std=3):
    """Creates a DataFrame, where each column shows upper boudaries, from which observations for
    a specified column are considered outliers, and number of outliers.
    
    Parameters
    ----------
        df : dataset,
        n_std : number of standard deviations from the mean as criterion for an observation being
                considered outlier.

    Returns
    ----------
        DataFrame of upper outliers bounds and counts.
    """
    
    outliers_dict = {}
    
    for col in df.columns.to_list():
        mean = df[col].mean()
        sd = df[col].std()
        X_filtered = df[df[col] > mean+(n_std*sd)]
        outliers_dict[col] = [
            df.shape[0]-df[(df[col] <= mean+(n_std*sd))].shape[0],
            round((df.shape[0]-df[(df[col] <= mean+(n_std*sd))].shape[0])/df.shape[0], 3),
            X_filtered[col].min()
        ]
        
    return(pd.DataFrame(outliers_dict, index=['Count', 'Percentage', 'Min']))