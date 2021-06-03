import pandas
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from dataset.preparation.ground_truths import GroundTruthBuilder


class SoilClassifier(object):
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def classify(self):
        data = pandas.read_csv(self.csv_file)
        print(data.head(5))
        x = data.drop([GroundTruthBuilder.PH_GENERIC_CLASS, GroundTruthBuilder.PH_VEG_ROW_CROPS_CLASS,
                       GroundTruthBuilder.PH_FRUITS_NUTS_CLASS], axis=1)
        print(f'Data after dropping columns = \n{x.head(5)}')
        # x = x.loc[:, 'ph']
        x = x.iloc[:, 6:7]
        # x.reset_index()
        # x.fillna(x.mean(), inplace=True)
        # print(f'null rows = {x.isnull().any(axis=0)}')

        print(f'Data after selecting required features = \n{x.head(5)}')
        print(f'type of x = {type(x)}')
        # y = numpy.ravel(data['soil_type'])
        y = data[GroundTruthBuilder.PH_GENERIC_CLASS]
        # y.fillna('NA', inplace=True)
        print(f'Labels {GroundTruthBuilder.PH_GENERIC_CLASS} = \n{y}')

        self.classify_by(x, y)

    def classify_by(self, features, labels):
        # print(f'>>> isnan =\n {numpy.isnan(features).all()}')
        # nan_array = numpy.where(numpy.isnan(features))
        # print(f'>>> nan_array = {nan_array}')

        # print(f'>>> isfinite =\n {numpy.isfinite(features).all()}')
        # nf_array = numpy.where(~numpy.isfinite(features))
        # print(f'n>>> f_array = {nf_array}')
        # for e in nf_array:
        #     if False in e:
        #         print(f'found infinite: {e}')
        #         break

        x_train, x_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.30,
                                                                            random_state=31)
        #
        # print(f'x_train==\n{x_train}')
        # print(f'x_test==\n{x_test}')
        # print(f'y_train==\n{y_train}')
        # print(f'y_test==\n{y_test}')
        #
        svc = SVC()
        knn = KNeighborsClassifier(n_neighbors=3)

        print('Training the model using SVC & KNeighbors classifier...')
        svc.fit(x_train, y_train)
        knn.fit(x_train, y_train)

        print('Training finished')

        print('Making predictions on the test data using the trained models..')
        svc_preds = svc.predict(x_test)
        knn_preds = knn.predict(x_test)

        print(f'SVC accuracy = \n{accuracy_score(svc_preds, y_test)}')
        print(f'KNN accuracy = \n{accuracy_score(knn_preds, y_test)}')

        print(f'SVC confusion matrix = \n{confusion_matrix(svc_preds, y_test)}')
        print(f'KNN confusion matrix = \n{confusion_matrix(knn_preds, y_test)}')

        print(f'SVC classification report = \n{classification_report(svc_preds, y_test)}')
        print(f'KNN classification report = \n{classification_report(knn_preds, y_test)}')








