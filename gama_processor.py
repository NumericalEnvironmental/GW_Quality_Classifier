###################################################################################
#
# GAMA_processor.py
#
# by Walt McNab
#
# employ machine learning tools to gain insights into water quality datasets
#
# GAMA data = California Groundwater Ambient Monitoring and Assessment Program
#
###################################################################################


from copy import *
from numpy import *
from pandas import *
from datetime import *
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
from sklearn import naive_bayes
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats


options.mode.chained_assignment = None


#####################################
#
# analysis classes and functions
#
#####################################


class Classify:

    def __init__(self):
        # instantiate classifier objects
        rf_obj = ensemble.RandomForestClassifier(n_estimators=2000)   		            # random forest (max depth = 12 is effective; trial & error)
        svm_obj = svm.SVC()                                                         	    # support vector machine	
        L_obj = linear_model.LogisticRegression()  					    # logistic regression	
        NN_obj = neighbors.KNeighborsClassifier() 					    # nearest neighbor
        NB_obj = naive_bayes.GaussianNB()
        self.clf_objs = [rf_obj, svm_obj, L_obj, NN_obj, NB_obj] 			    # set up parameter lists
        self.name = ['RandForest', 'SVM', 'Logistic', 'NearNeighbr', 'NaiveBayes']

    def ApplyClassif(self, train_ts_df, test_ts_df, comp_labels):	

        # set up data sets for classification 
        features = deepcopy(comp_labels)
        features.append('APPROXIMATE LATITUDE')
        features.append('APPROXIMATE LONGITUDE')
        features.append('DATE')
        feature_train_df = train_ts_df[features]                                            # training data set
        feature_train_df['DATE'] = feature_train_df['DATE'].apply(lambda x: x.toordinal())    
        scaler = preprocessing.StandardScaler().fit(feature_train_df.values)                # scale the training data set (for some classifiers)                                                                            
        X = scaler.transform(feature_train_df.values)
        Y = train_ts_df['group']
        feature_test_df = test_ts_df[features]                                              # test data set
        feature_test_df['DATE'] = feature_test_df['DATE'].apply(lambda x: x.toordinal())   
        Z = scaler.transform(feature_test_df.values)

        # apply various classifiers, taking advantage of the consistent interface presented by scikit-learn classes ...
        for i, clf_obj in enumerate(self.clf_objs):
            clf_obj.fit(X, Y)
            train_ts_df[self.name[i]] = clf_obj.predict(X)
            test_ts_df[self.name[i]] = clf_obj.predict(Z)
            print self.name[i] + ' classifier score =', clf_obj.score(X, Y)				

        return train_ts_df, test_ts_df


class PlotClassify:
    
    ### class to facilitate plotting classifier performance against water quality trend groups (stacked histograms)
	
    def __init__(self, group):
        self.width = 0.9
        self.num_classf = len(group)
        self.group = group
        self.bar_locs = arange(self.num_classf)
        self.color = ['dodgerblue', 'lime', 'yellow', 'orange', 'red']          # note that this is hard-wired for maximum of 5 trend groups

    def DrawClassHist(self, name, h):
        plt.figure(0)
        legend_entry = []
        base = zeros(self.num_classf, float)
        for i in xrange(self.num_classf):
            p = plt.bar(self.bar_locs-self.width/2, h[i], self.width, color=self.color[i], linewidth=0.0, bottom=base)
            legend_entry.append(p[0])
            base += h[i]
        plt.ylabel('Count')
        plt.title(name)
        plt.xticks(self.bar_locs, tuple(self.group))
        plt.legend(tuple(legend_entry), tuple(self.group))
        plt.show()

		
class CompareHistograms:

    ### class to facilitate plotting of histograms for
    # (1) water quality trends groups against principal components and time
    # (2) dC/dt measured by individual sample event versus whole-time-series average per well

    def __init__(self, group=[]):
        if len(group): self.label = []                                          # (1)
        else: self.label = ['dC/dt, individual', 'dC/dt, lumped avgs']          # (2)
        for grp in group: self.label.append('Class = ' + str(grp))
        self.color = ['dodgerblue', 'lime', 'yellow', 'orange', 'red']          # note that this is hard-wired for maximum of 5 trend groups
        self.alpha = 0.5

    def DrawPlot(self, plot_set, bins, fig_title, xlabel, ylabel):
        plt.figure(0)
        for i, p in enumerate(plot_set):
            plt.hist(p, bins, color=self.color[i], alpha=self.alpha, label=self.label[i])
        plt.legend(loc='upper right')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(fig_title)
        plt.show()
		
		
def RunPCA(df, analyte_list, num_comps):
    # run principal components analysis; write loadings and display contributions to variance
    col_label = []                                                                                          # set up column labels
    for i in xrange(num_comps): col_label.append('Component_' + str(i))
    X = log10(df[analyte_list].values)
    pca_loadings = PCA(n_components=num_comps).fit(X)                                                       # compute LOADINGS
    pca_loadings_df = DataFrame(pca_loadings.components_)
    pca_loadings_df.columns = analyte_list
    pca_loadings_df.to_csv('pca_loadings.csv')
    pca_scores = PCA(n_components=num_comps).fit_transform(X)                                               # compute SCORES
    pca_scores_df = DataFrame(pca_scores)
    pca_scores_df.columns = col_label
    df = concat([df, pca_scores_df], axis=1)                                                                # append scores to dataframe
    print 'PCA data set contributions to variance = ', cumsum(pca_loadings.explained_variance_ratio_)       # display contributions to variance
    return df, col_label		


def TempAnalysis(train_df, test_df, analyte_t):
    # temporal analysis (of "analyte_t"; e.g., chloride)
    combo_df = concat([train_df, test_df], axis=0)               		# lump training and test data (so that PCA results are consistent among both sets)
    combo_df = combo_df.dropna(subset=[analyte_t])    
    combo_df.reset_index(inplace=True, drop=True)
    combo_ts_df = TimesSeries(combo_df, analyte_t)              		# compute actual concentration changes with time
    return combo_ts_df


def AssignComponents(combo_ts_df, analyte_list, num_comps, dtmin, dtmax):	
	
    # conduct PCA, assign component values to data set(s), and label trend groups
    packed_ts_df = combo_ts_df.dropna(axis=0, subset=analyte_list, how='any')
    packed_ts_df = packed_ts_df[(packed_ts_df['dt']>=dtmin) & (packed_ts_df['dt']<=dtmax)]    
    packed_ts_df.reset_index(inplace=True, drop=True)                                
    packed_ts_df, comp_labels = RunPCA(packed_ts_df, analyte_list, num_comps)

    # assign to trend groups, based on dC/dt bin
    packed_ts_df['group'] = sign(packed_ts_df['dC_dt']) * (abs(packed_ts_df['dC_dt'])>0.35) \
        + sign(packed_ts_df['dC_dt']) * (abs(packed_ts_df['dC_dt'])>2.25) 
    group = list(set(packed_ts_df['group']))
    group.sort()
    
    # split dataframes back into training data and test data
    train_ts_df = packed_ts_df[packed_ts_df['COUNTY']!='KINGS']
    test_ts_df = packed_ts_df[packed_ts_df['COUNTY']=='KINGS'] 
    train_ts_df.reset_index(inplace=True, drop=True)    
    test_ts_df.reset_index(inplace=True, drop=True)

    return train_ts_df, test_ts_df, group, comp_labels	

	
	
#####################################
#
# utility functions
#
#####################################
		

def ReadGama(analyte_list, headings, file_name):

    # read entire gama data file
    gama_df = read_csv(file_name, sep='\t')
    gama_df['CHEMICAL'] = ['SODIUM' if isinstance(x,float) else x for x in gama_df['CHEMICAL']]     # correct sodium label ('NA' --> 'SODIUM')
    mask = gama_df['CHEMICAL'].isin(analyte_list)                                                   # limit chemicals in dataframe to those on analytes list
    gama_df = gama_df[mask]
    gama_df['DATE'] = to_datetime(gama_df['DATE'])                                                  # format dates
    gama_df = gama_df[gama_df['DATASET']!='EDF']                                               	    # exclude shallow environmental monitor wells from this analysis

    # handle unit conversions and non-detects
    ppm_match = array(gama_df['UNITS']=='MG/L')                                                  
    gama_df['CONC'] = ppm_match*gama_df['RESULT'] + (1-ppm_match)*0.001*gama_df['RESULT']       
    nd_match = array(gama_df['QUALIFIER']=='<')
    gama_df['CONC'] = nd_match*0.5*gama_df['CONC'] + (1-nd_match)*gama_df['CONC']
    gama_df = gama_df[(gama_df['CONC']>0.)]
    gama_df = gama_df[(gama_df['CONC']<=2000.)]                                                     # exclude obvious outliers    

    # create pivot table
    gama_df = pivot_table(gama_df, values='CONC', index=headings, columns=['CHEMICAL'])
    gama_df.reset_index(inplace=True)

    return gama_df


def CollapsedSets(df, headings, analyte_list):
    # return dataframes that have been collapsed to (1) historic median concentrations for each well, and (2) changes over time for selected analyte
    headings.remove('DATE')
    medians_df = df.groupby(headings, as_index=False)[analyte_list].median()                        # group by well, as appropriate
    pub_df = medians_df[(medians_df['DATASET']=='DDW') | (medians_df['DATASET']=='DWR')]            # split into public water supply and not- components
    nonpub_df = medians_df[(medians_df['DATASET']!='DDW') & (medians_df['DATASET']!='DWR')]
    pub_df['APPROXIMATE LONGITUDE'] += random.uniform(-0.5, 0.5, size=len(pub_df)) * 0.005          # add wiggles to public water supply well locations
    pub_df['APPROXIMATE LATITUDE'] += random.uniform(-0.5, 0.5, size=len(pub_df)) * 0.005
    medians_df = concat([pub_df, nonpub_df], axis=0)                                                # recombine dataframes
    well_locs_df = medians_df[['WELL NAME', 'APPROXIMATE LONGITUDE', 'APPROXIMATE LATITUDE']]       # save well locations (with wiggles) associated with well IDs
    deltas_df = df.groupby(headings, as_index=False)['dt', 'dC'].sum()                              # set up global dC/dt dataframe
    deltas_df['dCdt_avg'] = deltas_df['dC']/deltas_df['dt'] * 365.25
    deltas_df.drop(['APPROXIMATE LONGITUDE', 'APPROXIMATE LATITUDE'], axis=1, inplace=True)         # re-assign (updated) well locations
    deltas_df = merge(well_locs_df, deltas_df, how='inner', on=['WELL NAME'])   
    return medians_df, deltas_df

    
def TimesSeries(df, analyte_t):
    
    # compute changes in concentration per unit time, per sample event per well
    #timeseries_df = df.sort_values(['WELL NAME', 'DATE'], axis=0)                                  # sort entire dataframe first by well name, then by date
    timeseries_df = df.sort(columns=['WELL NAME', 'DATE'], axis=0)                                  # use this instead to be compatible with older versus of pandas
    timeseries_df.reset_index(inplace=True, drop=True)                                              # re-index the dataframe
    
    # define dt, dC, and dC/dt
    dates = timeseries_df['DATE'].apply(lambda x: x.toordinal())
    dt = array(dates)[1:] - array(dates)[:-1]
    dC = array(timeseries_df[analyte_t])[1:] - array(timeseries_df[analyte_t])[:-1]
    dC_dt = dC / dt
    
    # identify new wells encountered while sequentially sweeping data set
    well_match = array(timeseries_df['WELL NAME'][1:]==timeseries_df['WELL NAME'][:-1])
    timeseries_df = timeseries_df.ix[1:]                                                            # first sample removed, by implication
    timeseries_df['dt'] = dt
    timeseries_df['dC'] = dC
    timeseries_df['dC_dt'] = dC_dt * 365.25
    timeseries_df = timeseries_df[well_match==True]
    
    # re-index again and return
    timeseries_df.reset_index(inplace=True, drop=True)
    return timeseries_df
    
def PlotGroupHistograms(train_ts_df, group, comp_labels, hist_plots):

    # this function address relationships between trend groups and (1) principal
    # components, as well as (2) time using histograms, ANOVA, and Kruskal-Wallis H-test

    stat_summary = []
    
    # plot probability distributions for principal components per group
    comp_set_df = train_ts_df[comp_labels + ['group']]
    bins = linspace(-4, 4, 50)
    for i, comp in enumerate(comp_labels):
        plot_set = []
        for grp in group: plot_set.append(array(comp_set_df[comp_set_df['group']==grp][comp]))
        hist_plots.DrawPlot(plot_set, bins, comp, 'Value', 'Count')    
        stat_summary.append(stats.f_oneway(plot_set[0], plot_set[1], plot_set[2], plot_set[3], plot_set[4])) 

    # plot probability distributions verses time
    time_set_df = train_ts_df[['DATE', 'group']]
    time_set_df['t'] = train_ts_df['DATE'].apply(lambda x: x.toordinal())
    bin_min = time_set_df['t'].min()
    bin_max = time_set_df['t'].max()
    bins =  linspace(bin_min, bin_max, 50)
    plot_time = []
    for grp in group: plot_time.append(array(time_set_df[time_set_df['group']==grp]['t']  ))
    hist_plots.DrawPlot(plot_time, bins, 'Time', 'Ordinal Day', 'Count')
    stat_summary.append(stats.kruskal(plot_time[0], plot_time[1], plot_time[2], plot_time[3], plot_time[4])) 

    # write present value set to text file
    anova_file = open('ANOVA.txt', 'w')
    anova_file.writelines(['Test', '\t', 'Factor', '\t', '(test stat. and p-value)', '\n'])
    for i, comp in enumerate(comp_labels):
        anova_file.writelines(['1-way ANOVA', '\t', comp, '\t', str(stat_summary[i]), '\n'])        
    anova_file.writelines(['Kruskal-Wallis', '\t', 'Time', '\t', str(stat_summary[-1]), '\n'])
    anova_file.close()



#####################################
#
# main script
#
#####################################


def Process_gama():

    # definitions
    data_files = ['gama_all_kern.txt', 'gama_all_kings.txt', 'gama_all_tulare.txt', 'gama_all_fresno.txt']          # tab-delimated county data sets, from GAMA
    analyte_list = ['ALKB', 'CA', 'CL', 'FE', 'K', 'MG', 'MN', 'NO3N', 'SODIUM', 'SO4']                             # include these analytes
    headings = ['WELL NAME', 'APPROXIMATE LATITUDE', 'APPROXIMATE LONGITUDE', 'DATE', 'DATASET', 'COUNTY']          # for dataframe prcoessing
    analyte_t = 'CL' 			                  	# analyte with which to conduct time series analysis
    dtmin = 30.                                                 # minimum and maximum time intervals for defining a change in concentration
    dtmax = 5.*365.25
    num_comps = 5                                               # number of components to be considered in principal components analysis (PCA)
	
    # read GAMA data file (COUNTY=KINGS is pulled out as the test set, hard-wired into this test code)
    for i, data_file in enumerate(data_files):                                          # extract GAMA data from multiple county data sets
        print 'Reading', data_file
        county_gama_df = ReadGama(analyte_list, headings, data_file)
        if not i: gama_df = county_gama_df
        else: gama_df = concat([gama_df, county_gama_df], axis=0)
    print 'Splitting into training and test data sets.'
    test_df = gama_df[gama_df['COUNTY']=='KINGS']
    train_df = gama_df[gama_df['COUNTY']!='KINGS']	
	
    # combine data sets and quantify rates of change in chloride concentrations
    print 'Processing time series.'
    combo_ts_df = TempAnalysis(train_df, test_df, analyte_t)
    time_series_summary_df = combo_ts_df[combo_ts_df['DATASET']!='Site']
    time_series_summary_df.to_csv('time_series_summary.csv')                            # write to output file
  
    # run PCA analyses on data sets to set up input for regression
    print 'Running PCA on composite data set.'
    train_ts_df, test_ts_df, group, comp_labels = AssignComponents(combo_ts_df, analyte_list, num_comps, dtmin, dtmax)       # note: analytical NAs are dropped here
	
    # setup for classification  
    print 'Working on classification.'
    clf = Classify() 									# set up classification handler object
    train_ts_df, test_ts_df = clf.ApplyClassif(train_ts_df, test_ts_df, comp_labels)	# run classification functions
    train_ts_df.to_csv('train_ts.csv')   
    test_ts_df.to_csv('test_ts.csv')	
	
    # create plots showing relationships between trend groups and key parameters
    hist_plots = CompareHistograms(group) 			                        # set up class for plotting histograms
    PlotGroupHistograms(train_ts_df, group, comp_labels, hist_plots)	                # create plots

    # summarize match of classifiers to groups
    clf_hst = PlotClassify(group)                                                       # set up plotting object
    for i, name in enumerate(clf.name):
        h = []
        for grp in group:
            grp_frame = train_ts_df[train_ts_df['group']==int(grp)]
            h.append(histogram(array(grp_frame[name]), bins=5, range=(-2, 2))[0])
        h = transpose(array(h))
        clf_hst.DrawClassHist(name, h)

    # summarize general trends in data sets (to support analyses, above)
    print 'Summarizing historic data trends, for mapping'
    medians_df, deltas_df = CollapsedSets(combo_ts_df, headings, analyte_list)
    medians_df.to_csv('medians.csv')
    deltas_df.to_csv('deltas.csv')

    # plot histograms comparing dC/dt
    dCdtHisto = CompareHistograms()
    ps0 = time_series_summary_df['dC_dt']
    ps1 = deltas_df[deltas_df['COUNTY']!='KINGS']['dCdt_avg']
    plot_set = [ps0, ps1]
    bins = linspace(-10, 10, 70)
    fig_title = 'dC/dt'
    dCdtHisto.DrawPlot(plot_set, bins, fig_title, 'dCdt (mg/L/year)', 'Count')

    print 'Done.'


##### run script #####
Process_gama()

