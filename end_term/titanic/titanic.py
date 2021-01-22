import pandas as pd
import os
from time import process_time
# Models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# GridSearchCV
from sklearn.model_selection import GridSearchCV
#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
#from sklearn import feature_selection
#from sklearn import model_selection

# Deal with titanic_data
def titanic_data_deal():
    train_num = 891
    # df create
    try:
        test_df = pd.read_csv(os.path.join(os.getcwd(), 'test.csv'))
        train_df = pd.read_csv(os.path.join(os.getcwd(), 'train.csv'))
    except FileNotFoundError:
        test_df = pd.read_csv(os.path.join(os.getcwd(), 'titanic', 'test.csv'))
        train_df = pd.read_csv(os.path.join(os.getcwd(), 'titanic', 'train.csv'))
    data_df = train_df.append(test_df)

    # Engineering features
    '''
    # Cleaning name and extracting Title
    for name_string in data_df['Name']:
        data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

    # Replacing rare titles with more common ones
    mapping = { 'Mlle': 'Miss',
                'Major': 'Mr',
                'Col': 'Mr',
                'Sir': 'Mr',
                'Don': 'Mr',
                'Mme': 'Miss',
                'Jonkheer': 'Mr',
                'Lady': 'Mrs',
                'Capt': 'Mr',
                'Countess':'Mrs',
                'Ms': 'Miss',
                'Dona': 'Mrs'
            }
    data_df.replace({'Title': mapping}, inplace=True)
    titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
    for title in titles:
        age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]
        data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute
    '''

    # Substituting Age values in TRAIN_DF and TEST_DF:
    train_df['Age'] = data_df['Age'][:train_num]
    test_df['Age'] = data_df['Age'][train_num:]
    # Dropping Title feature
    #data_df.drop('Title', axis = 1, inplace = True)
    # Adding Family_Size
    data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']
    train_df['Family_Size'] = data_df['Family_Size'][:train_num]
    test_df['Family_Size'] = data_df['Family_Size'][train_num:]
    # Adding Family_Survival
    data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])
    data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)
    DEFAULT_SURVIVAL_VALUE = 0.5
    data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

    # Using data_df['Last_Name', 'Fare'] to find 'Family_Survival'
    for _, grp_df in data_df[['Survived', 'Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                            'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
        if (len(grp_df) != 1):
            # A Family group is found.
            for ind, row in grp_df.iterrows():
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0
    print("Number of passengers with family survival information:",
        data_df.loc[data_df['Family_Survival']!=0.5].shape[0])

    # Using data_df['Ticket'] to find 'Family_Survival'
    for _, grp_df in data_df.groupby('Ticket'):
        if (len(grp_df) != 1):
            for ind, row in grp_df.iterrows():
                if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                    smax = grp_df.drop(ind)['Survived'].max()
                    smin = grp_df.drop(ind)['Survived'].min()
                    passID = row['PassengerId']
                    if (smax == 1.0):
                        data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                    elif (smin==0.0):
                        data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0
    print("Number of passenger with family/group survival information: "
        +str(data_df[data_df['Family_Survival']!=0.5].shape[0]))

    train_df['Family_Survival'] = data_df['Family_Survival'][:train_num]
    test_df['Family_Survival'] = data_df['Family_Survival'][train_num:]

    # Making FARE BINS
    data_df['Fare'].fillna(data_df['Fare'].median(), inplace = True)
    # Making Bins
    data_df['FareBin'] = pd.qcut(data_df['Fare'], 5)
    label = LabelEncoder()
    data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])
    train_df['FareBin_Code'] = data_df['FareBin_Code'][:train_num]
    test_df['FareBin_Code'] = data_df['FareBin_Code'][train_num:]
    train_df.drop(['Fare'], 1, inplace=True)
    test_df.drop(['Fare'], 1, inplace=True)
    # Making AGE BINS
    data_df['AgeBin'] = pd.qcut(data_df['Age'], 4)
    label = LabelEncoder()
    data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])
    train_df['AgeBin_Code'] = data_df['AgeBin_Code'][:train_num]
    test_df['AgeBin_Code'] = data_df['AgeBin_Code'][train_num:]
    train_df.drop(['Age'], 1, inplace=True)
    test_df.drop(['Age'], 1, inplace=True)
    # Mapping SEX and cleaning data (dropping garbage)
    train_df['Sex'].replace(['male','female'],[0,1],inplace=True)
    test_df['Sex'].replace(['male','female'],[0,1],inplace=True)
    #train_df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
    #            'Embarked'], axis = 1, inplace = True)
    #test_df.drop(['Name','PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
     #           'Embarked'], axis = 1, inplace = True)
    train_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'Embarked', ], axis = 1, inplace = True)
    test_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'Embarked', ], axis = 1, inplace = True)

    print(train_df.info)
    print(test_df.info)
    return train_df, test_df

# Using GridSearchCV to find hyperparams
def knn_find_para(X, y, X_test):
    # Grid Search CV
    algorithm = ['auto']
    weights = ['uniform', 'distance']
    leaf_size = list(range(1, 50, 5))
    n_neighbors = [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22]
    hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 'n_neighbors': n_neighbors}
    gd = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=hyperparams, verbose=True, cv=10, scoring='roc_auc')
    gd.fit(X, y)
    print(gd.best_score_)
    print(gd.best_estimator_)
    #gd.best_estimator_.fit(X, y)
    #y_pred = gd.best_estimator_.predict(X_test)

# AI model fit, evaluate, predict store to csv
def package_ai(train_data_complete, train_label_complete, test_data, model, model_name):
    model.fit(train_data_complete, train_label_complete)
    print("Result of {} is {}".format(model_name, model.score(train_data_complete, train_label_complete)))

    # Output
    predicted_survived_list = []
    for predicted_survived in model.predict(test_data):
        predicted_survived_list.append(int(predicted_survived))

    output = pd.DataFrame({'PassengerId':range(892, 1309+1), 'Survived': predicted_survived_list})
    output.to_csv(os.path.join(os.getcwd(), 'titanic', '{}_submissions.csv'.format(model_name)), index=False)

if __name__ == '__main__':
    # Data deal and split
    train_df, test_df = titanic_data_deal()
    X = train_df.drop('Survived', 1)
    y = train_df['Survived']
    X_test = test_df.copy()

    std_scaler = StandardScaler()
    X = std_scaler.fit_transform(X)
    X_test = std_scaler.transform(X_test)

    # knn algo
    #knn_find_para(X, y, X_test)

    # package_ai
    ai_model_routing = {
        'SVC' : SVC(),
        'logist' : LogisticRegression(),
        'perceptron' : Perceptron(),
        'gaussian' : GaussianNB(),
        'decision_tree' : DecisionTreeClassifier(criterion='entropy'),
        'adaboost' : AdaBoostClassifier(),
        'MLP' : MLPClassifier(),
        'lasso' : Lasso(),
        'rf' : RandomForestClassifier(n_estimators=5, random_state=42, criterion='entropy'), # n_estimators=100/5
        'knn' : KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski',
                            metric_params=None, n_jobs=1, n_neighbors=6, p=2, weights='uniform'),
    }
    for model_name, model in ai_model_routing.items():
        start = process_time()
        package_ai(X, y, X_test, model, model_name)
        end = process_time()
        print('{} spent time:'.format(model_name), end-start)
        print()