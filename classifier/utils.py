from typing import List

from dataset.preparation.ground_truths import GroundTruthBuilder


def get_data_without_labels(data):
    return data.drop([GroundTruthBuilder.PH_GENERIC_CLASS, GroundTruthBuilder.PH_VEG_ROW_CROPS_CLASS,
                      GroundTruthBuilder.PH_FRUITS_NUTS_CLASS, GroundTruthBuilder.EC_GENERIC_CLASS,
                      GroundTruthBuilder.P_GENERIC_CLASS, GroundTruthBuilder.OC_GENERIC_CLASS,
                      GroundTruthBuilder.FE_GENERIC_CLASS, GroundTruthBuilder.MN_GENERIC_CLASS,
                      GroundTruthBuilder.OPTIMAL_MACRO_NUTRIENTS_CLASS,
                      GroundTruthBuilder.OPTIMAL_MICRO_NUTRIENTS_CLASS],
                     axis=1)


def get_applicable_labels(data, features: List[str], multilabel: bool = False):
    if len(features) == 1:
        if multilabel:
            # single feature multiple labels
            labels = data[GroundTruthBuilder.FEATURES_TO_LABELS_DICT[features[0]][1:]]
        else:
            # single feature multiple classes
            labels = data[GroundTruthBuilder.FEATURES_TO_LABELS_DICT[features[0]][0]]
    else:
        features_key = ''
        for f in features:
            features_key = features_key + ',' + f
        features_key = features_key.lstrip(',')
        labels = GroundTruthBuilder.FEATURES_TO_LABELS_DICT[features_key]
        if multilabel:
            # multiple features multiple labels
            labels = data[labels[1:]]
        else:
            # multiple features multiple classes
            labels = data[labels[0]]
    return labels
