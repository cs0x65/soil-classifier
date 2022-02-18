import numpy
import pandas
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
import matplotlib.pyplot as plt

from dataset.preparation.ground_truths import GroundTruthBuilder


class SoilClassifier(object):
    def __init__(self, csv_file):
        self.csv_file = csv_file
        # panda dataframe
        self.data = None

    def classify(self):
        self.data = pandas.read_csv(self.csv_file)
        self.ph_binary_classify()
        self.ph_multilabel_classify()

        # x = data.drop([GroundTruthBuilder.PH_GENERIC_CLASS, GroundTruthBuilder.PH_VEG_ROW_CROPS_CLASS,
        #                GroundTruthBuilder.PH_FRUITS_NUTS_CLASS], axis=1)
        # print(f'Data after dropping columns = \n{x.head(5)}')
        # # x = x.loc[:, 'ph']
        # x = x.iloc[:, 6:7]
        # # x.reset_index()
        # # x.fillna(x.mean(), inplace=True)
        # # print(f'null rows = {x.isnull().any(axis=0)}')
        #
        # print(f'Data after selecting required features = \n{x.head(5)}')
        # print(f'type of x = {type(x)}')
        # # y = numpy.ravel(data['soil_type'])
        # y = data[GroundTruthBuilder.PH_GENERIC_CLASS]
        # # y.fillna('NA', inplace=True)
        # print(f'Labels {GroundTruthBuilder.PH_GENERIC_CLASS} = \n{y}')
        #
        # # self.classify_by(x, y)
        #
        # mulitlabel_y = data[['ph_veg_and_row_crops', 'ph_fruits_and_nuts']]
        # print(mulitlabel_y.head(5))
        # print(data.head(5))
        #
        # self.classify_by(x, mulitlabel_y)

    def ph_binary_classify(self):
        print('--Start of pH binary classification--')

        x = self.data.drop([GroundTruthBuilder.PH_GENERIC_CLASS, GroundTruthBuilder.PH_VEG_ROW_CROPS_CLASS,
                            GroundTruthBuilder.PH_FRUITS_NUTS_CLASS], axis=1)
        x = x.iloc[:, 6:7]
        y = self.data[GroundTruthBuilder.PH_GENERIC_CLASS]

        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.30, random_state=31)

        print('Training model: SVC')
        svc = SVC()
        svc.fit(x_train, y_train)
        print('Training SVC finished')

        print('Predictions: SVC')
        svc_preds = svc.predict(x_test)
        print(f'SVC accuracy = \n{accuracy_score(svc_preds, y_test)}')
        print(f'SVC confusion matrix = \n{confusion_matrix(svc_preds, y_test)}')
        print(f'SVC classification report = \n{classification_report(svc_preds, y_test)}')

        print('Training model: k-NN')
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train, y_train)
        print('Training k-NN finished')

        print('Predictions: k-NN')
        knn_preds = knn.predict(x_test)
        print(f'KNN accuracy = \n{accuracy_score(knn_preds, y_test)}')
        print(f'KNN confusion matrix = \n{confusion_matrix(knn_preds, y_test)}')
        print(f'KNN classification report = \n{classification_report(knn_preds, y_test)}')

        cm = confusion_matrix(knn_preds, y_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
        # disp.plot()
        # plt.show()
        #
        # plt.scatter(x_train, y_train)
        # plt.show()

        distances, indexes = knn.kneighbors(x_train)
        plt.plot(distances.mean(axis=1))
        # plt.show()

        outlier_indexes = numpy.where(distances.mean(axis=1) > 0)
        print(f'outlier_indexes = {outlier_indexes}')
        outlier_values = self.data.iloc[outlier_indexes]
        print(f'outlier_values = {outlier_values}')
        plt.scatter(x_train, y_train, color='b')
        plt.scatter(outlier_values['ph'], outlier_values['ph_class'], color='r')
        plt.show()

        print('--End of pH binary classification--')

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

        # print(f'x_train==\n{x_train}')
        # print(f'x_test==\n{x_test}')
        # print(f'y_train==\n{y_train}')
        # print(f'y_test==\n{y_test}')
        #

    def ph_multilabel_classify(self):
        print('--Start of pH multilabel classification--')

        x = self.data.drop([GroundTruthBuilder.PH_GENERIC_CLASS, GroundTruthBuilder.PH_VEG_ROW_CROPS_CLASS,
                            GroundTruthBuilder.PH_FRUITS_NUTS_CLASS], axis=1)
        x = x.iloc[:, 6:7]
        y = self.data[['ph_veg_and_row_crops', 'ph_fruits_and_nuts']]

        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.30, random_state=31)

        print('Training model: BinaryRelevance')
        # initialize binary relevance multi-label classifier with a gaussian naive bayes base classifier
        classifier = BinaryRelevance(GaussianNB())
        classifier.fit(x_train, y_train)
        print('Predictions: BinaryRelevance')
        predictions = classifier.predict(x_test)
        print(f'Multilabel BinaryRelevance accuracy = \n{accuracy_score(predictions, y_test)}')
        print(f'Multilabel BinaryRelevance classification report = \n{classification_report(predictions, y_test)}')
        # print(f'Multilabel confusion matrix = \n{confusion_matrix(predictions, y_test)}')

        print('Training model: ClassifierChain')
        classifier = ClassifierChain(GaussianNB())
        diff = set(y_test) - set(y_train)
        classifier.fit(x_train, y_train)
        print('Predictions: ClassifierChain')
        predictions = classifier.predict(x_test)
        print(f'Multilabel ClassifierChain accuracy = \n{accuracy_score(predictions, y_test)}')
        print(f'Multilabel ClassifierChain classification report = \n{classification_report(predictions, y_test)}')

        print('Training model: LabelPowerset GNB')
        classifier = LabelPowerset(GaussianNB())
        classifier.fit(x_train, y_train)
        print('Predictions: LabelPowerset GNB')
        predictions = classifier.predict(x_test)
        print(f'Multilabel LabelPowerset GNB accuracy = \n{accuracy_score(predictions, y_test)}')
        print(f'Multilabel LabelPowerset GNB classification report = \n{classification_report(predictions, y_test)}')

        print('Training model: LabelPowerset SVM')
        # serial vector machine base classifier
        classifier = LabelPowerset(SVC())
        classifier.fit(x_train, y_train)
        print('Predictions: LabelPowerset SVM')
        predictions = classifier.predict(x_test)
        print(f'Multilabel LabelPowerset SVM accuracy = \n{accuracy_score(predictions, y_test)}')
        print(f'Multilabel LabelPowerset SVM classification report = \n{classification_report(predictions, y_test)}')

        print('Training model: MLkNN')
        classifier = MLkNN(k=21)
        classifier.fit(x_train.values, y_train.values)

        print('Predictions: MLkNN')
        predictions = classifier.predict(x_test)
        print(f'Multilabel MLkNN accuracy = \n{accuracy_score(predictions, y_test)}')
        print(f'Multilabel MLkNN classification report = \n{classification_report(predictions, y_test)}')
        # print(f'Multilabel MLkNN confusion matrix = \n{confusion_matrix(predictions, y_test)}')

        print('--End of pH multilabel classification--')
