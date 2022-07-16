from typing import List

import pandas
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset

from classifier import utils
from dataset.preparation.ground_truths import GroundTruthBuilder, FeatureSet

accuracy_dict = dict()


class SoilClassifier(object):
    def __init__(self, csv_file):
        self.csv_file = csv_file
        # panda dataframe
        self.data = None

    def classify(self):
        self.data = pandas.read_csv(self.csv_file)

        MulticlassClassifier(self.data).single_feature_multiclass_classify(features=[
            FeatureSet.PH.value_as_str, FeatureSet.EC.value_as_str, FeatureSet.OC.value_as_str,
            FeatureSet.P.value_as_str,
            FeatureSet.FE.value_as_str, FeatureSet.MN.value_as_str]
        )
        MulticlassClassifier(self.data).multi_feature_multiclass_classify(FeatureSet.MACRO_NUTRIENTS.value_as_list)
        MulticlassClassifier(self.data).multi_feature_multiclass_classify(FeatureSet.MICRO_NUTRIENTS.value_as_list)
        MulticlassClassifier(self.data).multi_feature_multiclass_classify(FeatureSet.FERTILITY_SET.value_as_list)

        MultilabelClassifier(self.data).single_feature_multilabel_classify(features=[FeatureSet.PH.value_as_str])

        print(f'\n\n*** accuracy_dict = {accuracy_dict} \n\n***')


class MulticlassClassifier(object):
    def __init__(self, data):
        self.data = data

    def single_feature_multiclass_classify(self, features: List):
        """
        This method acts on a single feature and applies classification algorithms such that each record in the
        test dataset on prediction belongs to one & only one class from the classes supplied in the training data.
        :param param features: list of features for each of which the classification models are trained and predicted.
        :return:
        """
        print(f'--Start of single feature multi-class classification for: {features}--')

        for feature in features:
            flist = [feature]
            self._knn_classify(features=flist, binary=GroundTruthBuilder.is_binary_classifiable_feature_set(flist),
                               plot=False)
            self._svm_classify(features=flist, binary=GroundTruthBuilder.is_binary_classifiable_feature_set(flist),
                               plot=True)

        print(f'--End of single feature multi-class classification for: {features}--')

    def multi_feature_multiclass_classify(self, features: List):
        """
        This method acts on multiple features and applies classification algorithms such that each record in the
        test dataset on prediction belongs to one & only one class from the classes supplied in the training data.
        :param features: list of features together for which the classification models are trained and predicted.
        :return:
        """
        print(f'--Start of multiple features multi-class classification for: {features}--')

        self._knn_classify(features=features, binary=GroundTruthBuilder.is_binary_classifiable_feature_set(features),
                           plot=True)
        self._svm_classify(features=features, binary=GroundTruthBuilder.is_binary_classifiable_feature_set(features))
        # TODO uncomment once other changes are verified
        self._logistic_regression_classify(features=features, binary=GroundTruthBuilder.
                                           is_binary_classifiable_feature_set(features))
        self._random_forest_classify(features=features, binary=GroundTruthBuilder.is_binary_classifiable_feature_set(
            features))

        print(f'--End of multiple features multi-class classification for: {features}--')

    def _svm_classify(self, features: List, binary=False, plot=False):
        print(f'--Start of SVM classification for: {features} binary: {binary}--')
        x = utils.get_data_without_labels(self.data)
        x = x[features]
        y = utils.get_applicable_labels(data=self.data, features=features)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.30, random_state=31)

        print('Training model: SVC')
        if binary:
            # find the diff in accuracy vis-a-vis linear and regular SVC
            print('Training model: binary classification: LinearSVC')
            svc = LinearSVC()
        else:
            svc = SVC(decision_function_shape='ovo')
        svc.fit(x_train, y_train)
        print('Training SVC finished')

        print('Predictions: SVC')
        svc_preds = svc.predict(x_test)
        acc_score = accuracy_score(svc_preds, y_test)
        accuracy_dict['svm'] = [acc_score, features, binary]
        print(f'SVC accuracy = \n{acc_score}')
        print(f'SVC confusion matrix = \n{confusion_matrix(svc_preds, y_test)}')
        print(f'SVC classification report = \n{classification_report(svc_preds, y_test)}')

        print(f'--End of SVM classification for: {features} binary: {binary}--')

    def _knn_classify(self, features: List, binary=False, plot=False):
        print(f'--Start of k-NN classification for: {features} binary: {binary}--')
        x = utils.get_data_without_labels(self.data)
        x = x[features]
        y = utils.get_applicable_labels(data=self.data, features=features)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.30, random_state=31)

        print('Training model: k-NN')
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train, y_train)
        print('Training k-NN finished')

        print('Predictions: k-NN')
        knn_preds = knn.predict(x_test)
        acc_score = accuracy_score(knn_preds, y_test)
        accuracy_dict['knn'] = [acc_score, features, binary]
        print(f'KNN accuracy = \n{acc_score}')
        print(f'KNN confusion matrix = \n{confusion_matrix(knn_preds, y_test)}')
        print(f'KNN classification report = \n{classification_report(knn_preds, y_test)}')

        if plot:
            cm = confusion_matrix(knn_preds, y_test)
            utils.render_confusion_matrix(cm=cm, classifier=knn, features=features)
            utils.render_scatter_graph(cm=cm, classifier=knn, features=features, x_train=x_train, y_train=y_train)
            utils.render_avg_neighbor_distance(cm=cm, classifier=knn, features=features, x_train=x_train)
            # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
            # title = str(features) + ':'
            # disp.plot()
            # plt.title(title + ' confusion matrix')
            # plt.xlabel('Predicted classes')
            # plt.ylabel('Actual classes')
            # plt.show()

            # plt.scatter(x_train, y_train)
            # plt.title(title + ' scatter graph')
            # plt.show()

            # distances, indexes = knn.kneighbors(x_train)
            # plt.plot(distances.mean(axis=1))
            # plt.title(title + ' avg. neighbor distance')
            # plt.show()

            # outlier_indexes = numpy.where(distances.mean(axis=1) > 0)
            # print(f'outlier_indexes = {outlier_indexes}')
            # outlier_values = self.data.iloc[outlier_indexes]
            # print(f'outlier_values = {outlier_values}')
            # plt.scatter(x_train, y_train, color='b')
            # plt.scatter(
            #     outlier_values[features],
            #     outlier_values[GroundTruthBuilder.FEATURES_TO_LABELS_DICT[features[0]][0]],
            #     color='r'
            # )
            # plt.title(title + ' outliers')
            # plt.show()

        print(f'--End of k-NN classification for: {features} binary: {binary}--')

    def _logistic_regression_classify(self, features: List, binary=False, plot=False):
        print(f'--Start of LogisticRegression classification for: {features} binary: {binary}--')
        x = utils.get_data_without_labels(self.data)
        x = x[features]
        y = utils.get_applicable_labels(data=self.data, features=features)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.30, random_state=31)

        print('Training model: LogisticRegression')
        if binary:
            lr = LogisticRegression(random_state=0, multi_class='ovr')
        else:
            lr = LogisticRegression(random_state=0, multi_class='multinomial')
        lr.fit(x_train, y_train)
        print('Training LogisticRegression finished')

        print('Predictions: LogisticRegression')
        svc_preds = lr.predict(x_test)
        acc_score = accuracy_score(svc_preds, y_test)
        accuracy_dict['logr'] = [acc_score, features, binary]
        print(f'LogisticRegression accuracy = \n{acc_score}')
        print(f'LogisticRegression confusion matrix = \n{confusion_matrix(svc_preds, y_test)}')
        print(f'LogisticRegression classification report = \n{classification_report(svc_preds, y_test)}')

        print(f'--End of LogisticRegression classification for: {features} binary: {binary}--')

    def _random_forest_classify(self, features: List, binary=False, plot=False):
        print(f'--Start of RandomForest classification for: {features} binary: {binary}--')
        x = utils.get_data_without_labels(self.data)
        x = x[features]
        y = utils.get_applicable_labels(data=self.data, features=features)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.30, random_state=31)

        print('Training model: RandomForest')
        # decide on max_depth, num_features etc.
        if binary:
            rf = RandomForestClassifier(random_state=0)
        else:
            rf = RandomForestClassifier(random_state=0, n_estimators=1000, max_depth=10)
        rf.fit(x_train, y_train)
        print('Training RandomForest finished')

        print('Predictions: RandomForest')
        svc_preds = rf.predict(x_test)
        acc_score = accuracy_score(svc_preds, y_test)
        accuracy_dict['rf'] = [acc_score, features, binary]
        print(f'RandomForest accuracy = \n{acc_score}')
        print(f'RandomForest confusion matrix = \n{confusion_matrix(svc_preds, y_test)}')
        print(f'RandomForest classification report = \n{classification_report(svc_preds, y_test)}')

        print(f'--End of RandomForest classification for: {features} binary: {binary}--')


class MultilabelClassifier(object):
    def __init__(self, data):
        self.data = data

    def single_feature_multilabel_classify(self, features: List):
        print(f'--Start of single feature multi-label classification for: {features}--')

        for feature in features:
            self._binary_relevance_classify(features=[feature], plot=False)
            self._classifier_chain_classify(features=[feature], plot=False)
            self._label_power_set_classify(features=[feature], plot=False)
            self._multi_learn_knn_classify(features=[feature], plot=False)

        print(f'--End of single feature multi-label classification for: {features}--')

    def _binary_relevance_classify(self, features: List, plot=False):
        print(f'--Start of Binary Relevance multilabel classification for: {features}--')
        x = self.data.drop([GroundTruthBuilder.PH_GENERIC_CLASS, GroundTruthBuilder.PH_VEG_ROW_CROPS_CLASS,
                            GroundTruthBuilder.PH_FRUITS_NUTS_CLASS], axis=1)
        x = x[features]
        labels = GroundTruthBuilder.FEATURES_TO_LABELS_DICT[features[0]][1:]
        y = self.data[labels]

        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.30, random_state=31)

        print('Training model: BinaryRelevance')
        # initialize binary relevance multi-label classifier with a gaussian naive bayes base classifier
        classifier = BinaryRelevance(GaussianNB())
        classifier.fit(x_train, y_train)
        print('Predictions: BinaryRelevance')
        predictions = classifier.predict(x_test)
        acc_score = accuracy_score(predictions, y_test)
        accuracy_dict['binary_relevance'] = [acc_score, features]
        print(f'Multilabel BinaryRelevance accuracy = \n{acc_score}')
        print(f'Multilabel BinaryRelevance classification report = \n{classification_report(predictions, y_test)}')
        # print(f'Multilabel confusion matrix = \n{confusion_matrix(predictions, y_test)}')

        print(f'--End of Binary Relevance multilabel classification for: {features}--')

    def _classifier_chain_classify(self, features: List, plot=False):
        print(f'--Start of Classifier Chain multilabel classification for: {features}--')

        x = self.data.drop([GroundTruthBuilder.PH_GENERIC_CLASS, GroundTruthBuilder.PH_VEG_ROW_CROPS_CLASS,
                            GroundTruthBuilder.PH_FRUITS_NUTS_CLASS], axis=1)
        x = x[features]
        labels = GroundTruthBuilder.FEATURES_TO_LABELS_DICT[features[0]][1:]
        y = self.data[labels]

        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.30, random_state=31)

        print('Training model: ClassifierChain')
        classifier = ClassifierChain(GaussianNB())
        diff = set(y_test) - set(y_train)
        classifier.fit(x_train, y_train)
        print('Predictions: ClassifierChain')
        predictions = classifier.predict(x_test)
        acc_score = accuracy_score(predictions, y_test)
        accuracy_dict['classifier_chain'] = [acc_score, features]
        print(f'Multilabel ClassifierChain accuracy = \n{acc_score}')
        print(f'Multilabel ClassifierChain classification report = \n{classification_report(predictions, y_test)}')

        print(f'--End of Classifier Chain multilabel classification for: {features}--')

    def _label_power_set_classify(self, features: List, plot=False):
        print(f'--Start of Label Power Set multilabel classification for: {features}--')

        x = self.data.drop([GroundTruthBuilder.PH_GENERIC_CLASS, GroundTruthBuilder.PH_VEG_ROW_CROPS_CLASS,
                            GroundTruthBuilder.PH_FRUITS_NUTS_CLASS], axis=1)
        x = x[features]
        labels = GroundTruthBuilder.FEATURES_TO_LABELS_DICT[features[0]][1:]
        y = self.data[labels]

        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.30, random_state=31)

        classifier = LabelPowerset(GaussianNB())
        classifier.fit(x_train, y_train)
        print('Predictions: LabelPowerset GNB')
        predictions = classifier.predict(x_test)
        acc_score = accuracy_score(predictions, y_test)
        accuracy_dict['label_power_set_gnb'] = [acc_score, features]
        print(f'Multilabel LabelPowerset GNB accuracy = \n{acc_score}')
        print(f'Multilabel LabelPowerset GNB classification report = \n{classification_report(predictions, y_test)}')

        print('Training model: LabelPowerset SVM')
        # serial vector machine base classifier
        classifier = LabelPowerset(SVC())
        classifier.fit(x_train, y_train)
        print('Predictions: LabelPowerset SVM')
        predictions = classifier.predict(x_test)
        acc_score = accuracy_score(predictions, y_test)
        accuracy_dict['label_power_set_svm'] = [acc_score, features]
        print(f'Multilabel LabelPowerset SVM accuracy = \n{acc_score}')
        print(f'Multilabel LabelPowerset SVM classification report = \n{classification_report(predictions, y_test)}')

        print(f'--End of Label Power Set multilabel classification for: {features}--')

    def _multi_learn_knn_classify(self, features: List, plot=False):
        print(f'--Start of Multi-learn kNN multilabel classification for: {features}--')

        x = self.data.drop([GroundTruthBuilder.PH_GENERIC_CLASS, GroundTruthBuilder.PH_VEG_ROW_CROPS_CLASS,
                            GroundTruthBuilder.PH_FRUITS_NUTS_CLASS], axis=1)
        x = x[features]
        labels = GroundTruthBuilder.FEATURES_TO_LABELS_DICT[features[0]][1:]
        y = self.data[labels]

        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.30, random_state=31)

        print('Training model: MLkNN')
        classifier = MLkNN(k=21)
        classifier.fit(x_train.values, y_train.values)

        print('Predictions: MLkNN')
        predictions = classifier.predict(x_test)
        acc_score = accuracy_score(predictions, y_test)
        accuracy_dict['multilearn_knn'] = [acc_score, features]
        print(f'Multilabel MLkNN accuracy = \n{acc_score}')
        print(f'Multilabel MLkNN classification report = \n{classification_report(predictions, y_test)}')
        # print(f'Multilabel MLkNN confusion matrix = \n{confusion_matrix(predictions, y_test)}')

        print(f'--End of Label Multi-learn kNN multilabel classification for: {features}--')
