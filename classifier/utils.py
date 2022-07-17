from typing import List

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from dataset.preparation.ground_truths import GroundTruthBuilder


def __get_all_labels() -> List[str]:
    return [GroundTruthBuilder.PH_GENERIC_CLASS, GroundTruthBuilder.PH_VEG_ROW_CROPS_CLASS,
            GroundTruthBuilder.PH_FRUITS_NUTS_CLASS, GroundTruthBuilder.EC_GENERIC_CLASS,
            GroundTruthBuilder.P_GENERIC_CLASS, GroundTruthBuilder.OC_GENERIC_CLASS,
            GroundTruthBuilder.FE_GENERIC_CLASS, GroundTruthBuilder.MN_GENERIC_CLASS,
            GroundTruthBuilder.OPTIMAL_MACRO_NUTRIENTS_CLASS,
            GroundTruthBuilder.OPTIMAL_MICRO_NUTRIENTS_CLASS,
            GroundTruthBuilder.GENERIC_FERTILITY_CLASS,
            GroundTruthBuilder.PH_MACRO_MICRO_ALL_OPTIMAL_CLASS
            ]


def get_data_without_labels(data, features: List[str] = None):
    """
    :param data: the complete data as retrieved by pandas by loading csv file
    :param features: the set of features which are being used for a classification
    :return: the data by dropping labels which don't overlap with the given list of features; typically a classifier
    can use the generated set of labels for the next run/round of classification, and that's when the features can
    overlap with the labels.
    """
    labels_to_drop = __get_all_labels()
    if features:
        labels_to_drop = [label for label in labels_to_drop if label not in features]
    return data.drop(labels_to_drop, axis=1)


def get_applicable_labels(data, features: List[str], multilabel: bool = False):
    if len(features) == 1:
        if multilabel:
            # single feature multiple labels
            labels = data[GroundTruthBuilder.FEATURES_TO_LABELS_DICT[features[0]][1:]]
        else:
            # single feature multiple classes
            labels = data[GroundTruthBuilder.FEATURES_TO_LABELS_DICT[features[0]][0]]
    else:
        features_key = ','.join(features)
        labels = GroundTruthBuilder.FEATURES_TO_LABELS_DICT[features_key]
        if multilabel:
            # multiple features multiple labels
            labels = data[labels[1:]]
        else:
            # multiple features multiple classes
            labels = data[labels[0]]
    return labels


def render_confusion_matrix(cm, classifier, features: List[str]):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    title = str(features) + ':'
    disp.plot()
    plt.title(title + ' confusion matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('Actual labels')
    plt.show()


def render_scatter_graph(cm, classifier, features: List[str], x_train, y_train):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    title = str(features) + ':'
    plt.title(title + ' scatter graph')
    plt.scatter(x_train, y_train)
    plt.show()


def render_avg_neighbor_distance(cm, classifier, features: List[str], x_train):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    title = str(features) + ':'
    distances, indexes = classifier.kneighbors(x_train)
    plt.plot(distances.mean(axis=1))
    plt.title(title + ' avg. neighbor distance')
    plt.show()
