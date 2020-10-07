import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler as scaler 
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def catstats(column_of_interest, data, ax):
    data["percent"] = [1] * len(data)
    a = data[[column_of_interest, "percent"]]
    a = a.groupby(column_of_interest)["percent"].count().to_frame()
    a.percent = ((a.percent / a.percent.sum()) * 100).apply(lambda x: round(x, 2))
    veri_sorted = a.sort_values("percent",ascending=False)
    veri_sorted.percent.plot(kind='bar',color = '#000088' , ax = ax)
    percent_of_CP = ["{}%".format(row["percent"]) for name,row in veri_sorted.iterrows()]
    for i,child in enumerate(ax.get_children()[:veri_sorted.index.size]):
         ax.text(i,child.get_bbox().y1+1,percent_of_CP[i], horizontalalignment ='center')
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.patch.set_facecolor('#FFFFFF')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['left'].set_linewidth(1)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    title = "univariate stats & percentage for " + column_of_interest
    ax.set_title(title)
    ax.legend()

def catplot(list_of_columns, data, figsize):
    nrows, ncols = 13, 2
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    list_like = np.array(list_of_columns)
    matrix_like = list_like.reshape((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            ax = axs[i][j]
            column_of_interest = matrix_like[i][j]
            catstats(column_of_interest=column_of_interest, data=data, ax=ax)
            

            
def remove_low_std(X, std_thrs):
        std_frame = X.describe().loc["std"].to_frame()
        std_frame_thrs = std_frame[std_frame["std"] < std_thrs]
        low_std_features = list(std_frame_thrs.index)
        return low_std_features, std_frame
    
    
# utils functions
def reduce_class_of_worker(class_of_worker):
    class_of_worker_dict = {
        " Federal government": "government",
        " Local government": "government",
        " Never worked": "others",
        " Not in universe": "not_in_universe", 
        " Private": "private",
        " Self-employed-incorporated": "self",
        " Self-employed-not incorporated": "self", 
        " State government": "government",
        " Without pay": "others"}
    return class_of_worker_dict[class_of_worker]

def reduce_education(education):
    education_dict = {
        " 10th grade": "low",
        " 11th grade": "low",
        " 12th grade no diploma": "low",
        " 1st 2nd 3rd or 4th grade": "low",
        " 5th or 6th grade": "low",
        " 7th and 8th grade": "low",
        " 9th grade": "low",
        " Associates degree-academic program": "median",
        " Associates degree-occup /vocational": "median",
        " Bachelors degree(BA AB BS)": "high",
        " Children": "low",
        " Doctorate degree(PhD EdD)": "top",
        " High school graduate": "median",
        " Less than 1st grade": "low",
        " Masters degree(MA MS MEng MEd MSW MBA)": "top",
        " Prof school degree (MD DDS DVM LLB JD)": "median",
        " Some college but no degree": "median"}
    return education_dict[education]

def reduce_marital_stat(marital_status):
    marital_status_dict = {
        " Divorced": "not_married",
        " Married-A F spouse present": "married",
        " Married-civilian spouse present": "married",
        " Married-spouse absent": "married",
        " Never married": "not_married",
        " Separated": "not_married",
        " Widowed": "not_married"
    }
    return marital_status_dict[marital_status]


def reduce_major_industry_code(industry_code):
    major_industry_code_dict = {
        " Agriculture": "primary",
        " Armed Forces": "public",
        " Business and repair services": "services",
        " Communications": "services",
        " Construction": "industry",
        " Education": "public",
        " Entertainment": "public",
        " Finance insurance and real estate": "finance",
        " Forestry and fisheries": "primary",
        " Hospital services": "health",
        " Manufacturing-durable goods": "industry",
        " Manufacturing-nondurable goods": "industry",
        " Medical except hospital": "health",
        " Mining": "primary",
        " Not in universe or children": "not_in_universe",
        " Other professional services": "services",
        " Personal services except private HH": "services",
        " Private household services": "services",
        " Public administration": "public",
        " Retail trade": "services",
        " Social services": "services",
        " Transportation": "services" ,
        " Utilities and sanitary services": "services",
        " Wholesale trade": "services"
    }
    return major_industry_code_dict[industry_code]

def reduce_major_occupation_code(occupation_code):
    major_occupation_code_dict = {
        " Adm support including clerical": "field" ,
        " Armed Forces": "field" ,
        " Executive admin and managerial": "not_field",
        " Farming forestry and fishing": "field",
        " Handlers equip cleaners etc ": "field",
        " Machine operators assmblrs & inspctrs": "field",
        " Not in universe": "not_in_universe",
        " Other service": "not_field",
        " Precision production craft & repair": "field",
        " Private household services": "field",
        " Professional specialty": "not_field",
        " Protective services": "field",
        " Sales": "not_field",
        " Technicians and related support": "field",
        " Transportation and material moving": "field"}
    
    return major_occupation_code_dict[occupation_code]

def reduce_race(race):
    if race != " White":
        return "minorities"
    return "white"

def reduce_full_or_part_time_employment_stat(time_employment):
    time_employment_dict = {
        " Children or Armed Forces": "not_in_universe",
        " Full-time schedules": "full",
        " Not in labor force": "not_in_universe",
        " PT for econ reasons usually FT": "partial",
        " PT for econ reasons usually PT": "partial",
        " PT for non-econ reasons usually FT": "partial",
        " Unemployed full-time": "full",
        " Unemployed part- time": "full"
    }
    return time_employment_dict[time_employment]

def reduce_detailed_household_summary_in_household(householder):
    if householder != " Householder":
        return "not_householder"
    return "householder"

def reduce_country_birth(country):
    if country != " United-States":
        return "OUT"
    return "UN"

reduce_country_of_birth_self = reduce_country_of_birth_mother =  reduce_country_of_birth_father = lambda s: reduce_country_birth(s)

def reduce_citizenship(citizenship):
    if citizenship != " Foreign born- Not a citizen of U S ":
        return "citizen"
    return "no_citizen"

def reduce_tax_filer_stat(taxe):
    if taxe != " Nonfiler":
        return "filer"
    return "not_filer"
        
  
    
def imputer_transform(X, categoric_missing_features):
    for var in categoric_missing_features:
        imputer =  SimpleImputer(strategy="most_frequent")
        X[var] = imputer.fit_transform(X[var].values.reshape(-1, 1))
    return X

def outlier_transform(X):
    num_features = list(X.select_dtypes(include=['float64', 'int64']))
    X[num_features] = scaler().fit_transform(X[num_features])
    for var in X.select_dtypes(include=['float64', 'int64']):
        # scaler before 
        X = X[np.abs(X[var] - X[var].mean()) <= (3 * X[var].std())]
    return X

def label_encoder_transform(X):
    cat_features = list(X.select_dtypes(exclude=['float64', 'int64']))
    for var in cat_features:
        X[var] = LabelEncoder().fit_transform(X[var].values)
    return X 


def missing_values(X, thrs=80):
    total = X.isnull().sum().sort_values(ascending=False)
    percent = X.isnull().sum()/X.isnull().count().sort_values(ascending=False)*100
    missing_data = pd.concat([total,percent], axis=1, keys=['total', 'percentage'])
    features_of_interest = list(missing_data[(percent<=thrs)].index)
    return features_of_interest, missing_data