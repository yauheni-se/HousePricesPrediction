import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

from IPython.display import display
from fitter import Fitter, get_common_distributions, get_distributions
from scipy import stats

def prepare_data(df, y, vars_subset=None, what=None):
    """Performs the most handful data preparation operations. 
    
    Parameters
    ----------
        df : dataset,
        y : name of the explainable variable,
        vars_subset : selected variable(s) from the dataset and the explainable variable,
                      default=None (all)
        what : which type of preparation to perform,
               default=['dummify', 'standardize', 'stimulate'].
               Available options are:
               - 'dummify' : convert categorical features to series of dummies,
                             while dropping the most frequent category
               - 'standardize' : apply standartization to non-binary columns,
               - 'stimulate' : change the sign of correlation with y variable.
                               For binary columns, values switch places (0 becames 1, 1 becames 0).
                               For numeric columns, values are multiplied by -1.
        
    Returns
    ----------
        Prepared dataset.
    """
    if what is None:
        what = ['dummify', 'standardize', 'stimulate']
    
    if vars_subset is None:
        vars_subset = df.columns.tolist()
        
    dtypes_num = ['int64', 'int32', 'int16', 'float64', 'float32', 'float16']
    dtypes_str = ['object', 'category']
    vars_num = df.loc[:, vars_subset].select_dtypes(include=dtypes_num).columns
    vars_str = df.loc[:, vars_subset].select_dtypes(include=dtypes_str).columns

    binary_cols = df.loc[:, vars_num].apply(lambda x: x.unique().tolist() in [[1, 0], [0,1]])
    binary_cols = binary_cols[binary_cols].index.to_list()

    non_binary_cols = df.loc[:, vars_num].apply(lambda x: x.unique().tolist() not in [[1, 0], [0,1]])
    non_binary_cols = non_binary_cols[non_binary_cols].index.to_list()
    vars_num = df.loc[:, vars_num].loc[:, non_binary_cols].columns
    
    if 'dummify' in what:
        vars_to_drop = []
        for i in vars_str:
            vars_to_drop.append(i+'_'+df[i].value_counts().index[0])
        df = pd.get_dummies(df, columns=vars_str, prefix=vars_str)
        df = df.drop(columns=vars_to_drop)
        
    if 'standardize' in what:
        df[vars_num] = df[vars_num].apply(lambda x: stats.zscore(x))
        
    if 'stimulate' in what:
        df_corr = df.corr()[y].to_frame(name='corr').reset_index()
        vars_to_stimulate = df_corr.loc[df_corr['corr']<0, :]['index'].to_list()
    
        if len(vars_to_stimulate) != 0:
            vars_to_stimulate_bin = [x for x in vars_to_stimulate if x in binary_cols]
            vars_to_stimulate_norm = [x for x in vars_to_stimulate if x not in binary_cols]
        
            if len(vars_to_stimulate_norm) != 0:
                df[vars_to_stimulate_norm] = df[vars_to_stimulate_norm].apply(lambda x: -x)
            if len(vars_to_stimulate_norm) != 0:
                df[vars_to_stimulate_bin] = df[vars_to_stimulate_bin].apply(
                    lambda x: x.replace({0 : 2, 1 : 3}).replace({2 : 1, 3 : 0})
                )
                print('binary values for ', vars_to_stimulate_bin, ' columns were switched')
    return df


def do_call(what, *args, **kwargs):
    """Helper function to call scipy's distributions in an R's manner (from string).
    
    Parameters
    ----------
        what : the name of the distribution,
        args : array of numeric values to transform,
        kwargs : distribution parameters given to the scipy.stats.
        
    Returns
    ----------
        Array of transformed numeric values.
    """
    return getattr(getattr(stats, what), 'cdf')(*args, **kwargs)

def plot_dist_comparison(df, vars_num, vars_normed, best_dist_vals, 
                         ncols=3, h_space=0.02, p_height=1000, f_size=8):
    """Helper function that creates plots of variables' distribution.
    
    Parameters
    ----------
        df : dataset,
        vars_num : selected numeric variable(s) from the dataset,
        vars_normed : normalized selected numeric variable(s)' values
        best_dist_vals : density values of best fitted distributions,
        ncols : number of plot's columns, default=3,
        h_space : spacing between subplots, default=0.02,
        p_height : height of single plot, default=1000,
        f_size : yaxis title's font size of single plot, default=8.

    Returns
    ----------
        Distribution(s) plot for each selected variable.
    """
    
    color_background = '#F5F5F5'
    color_gridlines = '#DCDCDC'
    colors_in_use = [
        '#537EA2', #'#42A593', #'#858F84', '#2C3E50', 
        '#873E23', #'#CFD1A1',
        #'#6A744F', 
        #'#BDBDC5',
        #'#7EA253', 
        '#EDB676', '#C26D40'
    ]+(px.colors.qualitative.Safe+
       px.colors.qualitative.Pastel+
       px.colors.qualitative.Prism+
       px.colors.qualitative.Antique+
       px.colors.qualitative.Vivid+
       px.colors.qualitative.Plotly
      )

    if len(vars_num) % ncols == 0:
        dim_1 = int(len(vars_num)/ncols)
    else:
        dim_1 = int(len(vars_num)/ncols)+1
    dim_2 = ncols
    fig = make_subplots(rows=dim_1, cols=dim_2, horizontal_spacing=h_space)
    
    unique_distribs_lst = []
    
    for i in range(dim_2):
        for j in range(dim_1):
            if j+(i*dim_1) >= len(vars_num):
                continue
            tmp_ind = 0
            for k in best_dist_vals[vars_num[j+(i*dim_1)]]:
                
                if k in unique_distribs_lst:
                    show_on_legend = False
                else:
                    unique_distribs_lst.append(k)
                    show_on_legend = True
                
                fig.append_trace(
                    go.Scatter(
                        x=vars_normed[vars_num[j+(i*dim_1)]],
                        y=best_dist_vals[vars_num[j+(i*dim_1)]][k],
                        line=dict(width=3),
                        marker=dict(color=colors_in_use[min(tmp_ind+1, len(colors_in_use))]),
                        name=k,
                        legendgroup=k,
                        showlegend = show_on_legend,
                        opacity=0.8
                    ), j+1, i+1)
                tmp_ind += 1
            fig.append_trace(
                go.Histogram(
                    x=df[vars_num[j+(i*dim_1)]], autobinx=False, histnorm='probability density',
                    marker_line_color='rgb(8,48,107)',
                    marker_line_width=1.5,
                    marker=dict(color=colors_in_use[0]),
                    showlegend=False,
                    opacity=0.8,
                    name=vars_num[j+(i*dim_1)]
                ), j+1, i+1)
            fig.update_yaxes(title_text=vars_num[j+(i*dim_1)], row=j+1, col=i+1)
            fig.update_yaxes(title_font_size=f_size, row=j+1, col=i+1)
            fig.update_yaxes(title_font_color='black', row=j+1, col=i+1)
    fig.update_layout(
        height=p_height,
        paper_bgcolor=color_background,
        plot_bgcolor=color_background
    )
    fig.update_yaxes(gridcolor=color_gridlines, showticklabels=False)
    fig.update_xaxes(linecolor=color_gridlines, showticklabels=False)
    fig.show()
    
def fit_distributions(df, 
                      vars_subset=None, distribs_lst=None,
                      criterion='sumsquare_error',
                      refit=False, keep_original=False, show_plot=True, 
                      n_best=1, ncols=3, h_space=0.02, p_height=1000, f_size=8):
    """Automatically selects the best distribution for each numeric column of the dataset.
    
    Categorical and 0-1 columns will be ignored.
    
    Parameters
    ----------
        df : dataset,
        vars_subset : selected numeric variables from the dataset,
                      default=all numeric, non-binary columns,
        distribs_lst : list of supposed candidates for distribution fit,
                       default=distributions from get_common_distributions(),
        criterion : which criterion to use, default='sumsquare_error'. Available options are:
                    'sumsquare_error', 'aic', 'bic', 'kl_div', 'ks_statistic' 'ks_pvalue',
        refit : whether to apply transformation to the selected variables, default=False.
        keep_original : whether to keep variables before transformation or replace them,
                        default=False.
        show_plot : whether to show distribution comparison for each variable, default=True.
        n_best : how many best distributions to show, default=1,
        vars_subset: selected variable(s) from the dataset besides explainable variable, 
                     default=None (all),
        ncols : number of plot's columns, default=3,
        h_space : spacing between subplots, default=0.02,
        p_height : height of single plot, default=1000,
        f_size : yaxis title's font size of single plot, default=8.

    Returns
    ----------
        DataFrame with information regarding the best distribution for each selected variable,
        Distribution(s) plot for each selected variable,
        DataFrame with transformed selected variables.
    """
    best_params = dict()
    best_dist = dict()
    best_dist_vals = dict()
    vars_normed = dict()
    vars_normed_hist = dict()
    
    if distribs_lst is None:
        distribs_lst = get_common_distributions()
    
    if vars_subset is None:
        vars_subset = df.columns.tolist()
    
    dtypes_num = ['int64', 'int32', 'int16', 'float64', 'float32', 'float16']
    vars_num = df.loc[:, vars_subset].select_dtypes(include=dtypes_num).columns
    
    non_binary_cols = df.loc[:, vars_num].apply(lambda x: x.unique().tolist() not in [[1, 0], [0,1]])
    non_binary_cols = non_binary_cols[non_binary_cols].index.to_list()
    vars_num = df.loc[:, vars_num].loc[:, non_binary_cols].columns
    
    for i in range(len(vars_num)):
        f = Fitter(df[vars_num[i]].values, distributions=distribs_lst)
        f.fit(progress=False);
        best_dist[vars_num[i]], best_params[vars_num[i]] = list(
            f.get_best(method = criterion).items()
        )[0]
        top_n_dist_vals = dict()
        top_n_dist = f.summary(Nbest=n_best, plot=False).index.to_list()
        for j in top_n_dist:
            if j in f.fitted_pdf.keys():
                top_n_dist_vals[j] = f.fitted_pdf[j]
            else:
                continue
        best_dist_vals[vars_num[i]] = top_n_dist_vals
    
        vars_normed[vars_num[i]] = f.x
        vars_normed_hist[vars_num[i]] = f._data
        

    best_dist_df = pd.DataFrame(pd.Series(best_dist, name='best distribution')).transpose()
    display(best_dist_df)
    
    best_dist_df_summary = best_dist_df.transpose().value_counts().to_frame(name='count').reset_index()
    best_dist_df_summary['percentage'] = best_dist_df_summary['count']/best_dist_df_summary['count'].sum()
    display(best_dist_df_summary)

    if show_plot:
        plot_dist_comparison(df, vars_num, vars_normed, best_dist_vals, ncols, h_space, p_height, f_size)

    if refit:
        df_trnsf = pd.DataFrame()
        for i in range(len(vars_num)):
            df_trnsf_single = df.loc[:, vars_num[i]].to_frame().reset_index().sort_values(by=vars_num[i])
            df_trnsf_single[vars_num[i]+'_trnsf'] = do_call(
                best_dist[vars_num[i]],
                df_trnsf_single[vars_num[i]],
                **best_params[vars_num[i]]
            )
            df_trnsf_single = df_trnsf_single.sort_values(by='index').drop(columns='index')
            df_trnsf = pd.concat([df_trnsf, df_trnsf_single], axis=1)

        if not keep_original:
            df_trnsf = df_trnsf.loc[:, df_trnsf.columns.str.endswith('_trnsf')]
        df = pd.concat([df.drop(columns=vars_num), df_trnsf], axis=1)
        return df