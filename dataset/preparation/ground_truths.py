import csv
from enum import Enum, unique
from typing import Dict, List
import os

import pandas


@unique
class FeatureSet(Enum):
    PH = 'ph'
    EC = 'ec'
    OC = 'oc'
    P = 'av_p'
    FE = 'av_fe'
    MN = 'av_mn'
    MACRO_NUTRIENTS = 'oc,av_p'
    MICRO_NUTRIENTS = 'av_fe,av_mn'
    COMPLETE_FERTILITY_SET = 'ph,ec,oc,av_p,av_fe,av_mn'
    # here we are trying to use the generated label/class values to do the classification -- this is kind of using
    # aggregated values as features to derive the fertility class
    PH_MACRO_MICRO_SET = 'oc,av_p,av_fe,av_mn,ph'

    def __str__(self):
        return self.value[0] if isinstance(self.value, tuple) else self.value

    @property
    def value_as_str(self):
        return self.__str__()

    @property
    def value_as_list(self) -> List:
        return self.value_as_str.split(',') if ',' in self.value_as_str else [self.value_as_str]


class GroundTruthBuilder:
    PH_GENERIC_CLASS = 'ph_class'
    PH_VEG_ROW_CROPS_CLASS = 'ph_veg_and_row_crops'
    PH_FRUITS_NUTS_CLASS = 'ph_fruits_and_nuts'

    EC_GENERIC_CLASS = 'ec_class'

    OC_GENERIC_CLASS = 'oc_class'
    P_GENERIC_CLASS = 'av_p_class'

    FE_GENERIC_CLASS = 'av_fe_class'
    MN_GENERIC_CLASS = 'av_mn_class'

    OPTIMAL_MACRO_NUTRIENTS_CLASS = 'optimal_macro_nutrients_class'
    OPTIMAL_MICRO_NUTRIENTS_CLASS = 'optimal_micro_nutrients_class'

    GENERIC_FERTILITY_CLASS = 'generic_fertility_class'

    PH_MACRO_MICRO_ALL_OPTIMAL_CLASS = 'ph_macro_micro_all_optimal_class'

    PH_AND_MACRO_MICRO_EITHER_OPTIMAL_CLASS = 'ph_and_macro_micro_either_optimal_class'

    # May need to specify what are the tolerable avg distances from the neighbors for them not to be outliers when using
    # kNN algorithm.
    FEATURES_TO_LABELS_DICT = {
        FeatureSet.PH.value_as_str: [PH_GENERIC_CLASS, PH_VEG_ROW_CROPS_CLASS, PH_FRUITS_NUTS_CLASS],
        FeatureSet.EC.value_as_str: [EC_GENERIC_CLASS],
        FeatureSet.OC.value_as_str: [OC_GENERIC_CLASS],
        FeatureSet.P.value_as_str: [P_GENERIC_CLASS],
        FeatureSet.FE.value_as_str: [FE_GENERIC_CLASS],
        FeatureSet.MN.value_as_str: [MN_GENERIC_CLASS],
        FeatureSet.MACRO_NUTRIENTS.value_as_str: [OPTIMAL_MACRO_NUTRIENTS_CLASS],
        FeatureSet.MICRO_NUTRIENTS.value_as_str: [OPTIMAL_MICRO_NUTRIENTS_CLASS],
        FeatureSet.COMPLETE_FERTILITY_SET.value_as_str: [GENERIC_FERTILITY_CLASS],
        FeatureSet.PH_MACRO_MICRO_SET.value_as_str: [PH_MACRO_MICRO_ALL_OPTIMAL_CLASS,
                                                     PH_AND_MACRO_MICRO_EITHER_OPTIMAL_CLASS]
    }

    BINARY_CLASSIFIABLE_FEATURES = [
        FeatureSet.MACRO_NUTRIENTS.value_as_str,
        FeatureSet.MICRO_NUTRIENTS.value_as_str,
        FeatureSet.COMPLETE_FERTILITY_SET.value_as_str,
        FeatureSet.PH_MACRO_MICRO_SET.value_as_str
    ]

    @staticmethod
    def is_binary_classifiable_feature_set(features: List[str]):
        return ','.join(features) in GroundTruthBuilder.BINARY_CLASSIFIABLE_FEATURES

    def __init__(self, in_csv_file: str, out_csv_file: str = None):
        self.in_csv_file = in_csv_file.strip()
        self.out_csv_file = out_csv_file.strip() if out_csv_file else f'{self.in_csv_file.split(".csv")[0]}_with_gt.csv'
        self.cleansed_out_file = f'{self.in_csv_file.split(".csv")[0]}_cleansed.csv'
        self.headers = None
        self.dataset_with_gd = list()

    def _cleanse_data(self):
        """
        TODO: currently on those columns are being cleansed which are actually being used in classification.
        If required, in the future cleanse all the columns.
        """
        print('--Start:Cleansing the data and replacing missing values--')
        data = pandas.read_csv(self.in_csv_file, na_values=['', ' ', None], skipinitialspace=True)
        # print(f'columns = {data.columns}')
        self._cleanse_macro_nutrients(data)
        self._cleanse_micro_nutrients(data)
        data.to_csv(self.cleansed_out_file, index=False)
        print('--End:Cleansing the data and replacing missing values--')

    @staticmethod
    def _cleanse_macro_nutrients(data):
        print('--Start:Cleansing macro-nutrients--')
        av_p = data['av_p']
        av_p.fillna(av_p.mean(), inplace=True)
        print(f'## max/min av_p = {av_p.max()} {av_p.min()}')
        av_k = data['av_k']
        av_k.fillna(av_k.mean(), inplace=True)
        print(f'## max/min av_k = {av_k.max()} {av_k.min()}')
        av_s = data['av_s']
        av_s.fillna(av_s.mean(), inplace=True)
        print(f'## max/min av_s = {av_s.max()} {av_s.min()}')
        print('--End:Cleansing macro-nutrients--')

    @staticmethod
    def _cleanse_micro_nutrients(data):
        print('--Start:Cleansing micro-nutrients--')
        av_cu = data['av_cu']
        av_cu.fillna(av_cu.mean(), inplace=True)
        print(f'## max/min av_cu = {av_cu.max()} {av_cu.min()}')
        av_mn = data['av_mn']
        av_mn.fillna(av_mn.mean(), inplace=True)
        print(f'## max/min av_mn = {av_mn.max()} {av_mn.min()}')
        av_zn = data['av_zn']
        av_zn.fillna(av_zn.mean(), inplace=True)
        print(f'## max/min av_zn = {av_zn.max()} {av_zn.min()}')
        av_fe = data['av_fe']
        av_fe.fillna(av_fe.mean(), inplace=True)
        print(f'## max/min av_fe = {av_fe.max()} {av_fe.min()}')
        print('--End:Cleansing micro-nutrients--')

    def build_all(self):
        self._cleanse_data()

        # build the ground truths from the cleansed dataset
        with open(self.cleansed_out_file) as csv_file:
            reader = csv.DictReader(csv_file)
            self.headers = reader.fieldnames
            print(f'## fieldnames before: {self.headers}')
            for row in reader:
                # print(f'Row no {reader.line_num}: {row}')
                self.build_ph_labels(row)
                self.build_ec_labels(row)
                self.build_oc_labels(row)
                self.build_p_labels(row)
                self.build_fe_labels(row)
                self.build_mn_labels(row)
                self.build_optimal_macro_nutrients_binary_label(row)
                self.build_optimal_micro_nutrients_binary_label(row)
                self.build_generic_fertility_label(row)
                self.build_ph_macro_micro_all_optimal_label(row)
                self.build_ph_and_macro_micro_either_optimal_label(row)
                self.dataset_with_gd.append(row)

            self.headers.append(GroundTruthBuilder.PH_GENERIC_CLASS)
            self.headers.append(GroundTruthBuilder.PH_VEG_ROW_CROPS_CLASS)
            self.headers.append(GroundTruthBuilder.PH_FRUITS_NUTS_CLASS)
            self.headers.append(GroundTruthBuilder.EC_GENERIC_CLASS)
            self.headers.append(GroundTruthBuilder.OC_GENERIC_CLASS)
            self.headers.append(GroundTruthBuilder.P_GENERIC_CLASS)
            self.headers.append(GroundTruthBuilder.FE_GENERIC_CLASS)
            self.headers.append(GroundTruthBuilder.MN_GENERIC_CLASS)
            self.headers.append(GroundTruthBuilder.OPTIMAL_MACRO_NUTRIENTS_CLASS)
            self.headers.append(GroundTruthBuilder.OPTIMAL_MICRO_NUTRIENTS_CLASS)
            self.headers.append(GroundTruthBuilder.GENERIC_FERTILITY_CLASS)
            self.headers.append(GroundTruthBuilder.PH_MACRO_MICRO_ALL_OPTIMAL_CLASS)
            self.headers.append(GroundTruthBuilder.PH_AND_MACRO_MICRO_EITHER_OPTIMAL_CLASS)

            self._write_to_csv()
            if os.path.exists(self.cleansed_out_file):
                print(f'--Deleting the temp file: {self.cleansed_out_file}--')
                os.remove(self.cleansed_out_file)

    def _write_to_csv(self):
        with open(self.out_csv_file, mode='w') as csv_file:
            print(f'Writing to CSV file: {self.out_csv_file}')
            print(f'Headers: {self.headers}')
            dict_writer = csv.DictWriter(csv_file, fieldnames=self.headers)
            dict_writer.writeheader()
            dict_writer.writerows(self.dataset_with_gd)
            print(f'Finished writing to the CSV file: {self.out_csv_file}')

    @staticmethod
    def build_ph_labels(row: Dict):
        ph = float(row['ph'])

        # Generic class based on ph value
        if ph < 6:
            # plants suffer when ph < 4.8
            ph_class = 'Acidic'
        elif ph > 7.3:
            ph_class = 'Alkaline'
        else:
            ph_class = 'Optimal'
        row[GroundTruthBuilder.PH_GENERIC_CLASS] = ph_class

        # Crop specific ph class
        row[GroundTruthBuilder.PH_VEG_ROW_CROPS_CLASS] = 5.8 <= ph <= 6.5
        row[GroundTruthBuilder.PH_FRUITS_NUTS_CLASS] = 5.5 <= ph <= 5.8

    @staticmethod
    def build_ec_labels(row: Dict):
        # electrical conductivity
        ec = float(row['ec'])

        # Generic class based on ec value
        if ec <= 0.5:
            # normal
            ec_class = 'Normal'
        elif 0.5 < ec <= 1.5:
            # marginally high: causes salt injury
            ec_class = 'Marginally High'
        else:
            ec_class = 'Excessive'
        row[GroundTruthBuilder.EC_GENERIC_CLASS] = ec_class

    @staticmethod
    def build_oc_labels(row: Dict):
        # organic carbon - it's an indirect measurement of nitrogen N2O
        oc = float(row['oc'])

        # Generic class based on oc value
        if oc < 0.5:
            # plants suffer when ph < 4.8
            oc_class = 'Low'
        elif oc > 0.75:
            oc_class = 'High'
        else:
            oc_class = 'Optimal'
        row[GroundTruthBuilder.OC_GENERIC_CLASS] = oc_class

    @staticmethod
    def build_p_labels(row: Dict):
        # phosphorous/P2O - one of the most important of macro nutrients
        av_p = float(row['av_p'])

        # Generic class based on av_p value
        if av_p < 10:
            av_p_class = 'Low'
        elif av_p > 24.6:
            av_p_class = 'High'
        else:
            av_p_class = 'Optimal'
        row[GroundTruthBuilder.P_GENERIC_CLASS] = av_p_class

    @staticmethod
    def build_fe_labels(row: Dict):
        # iron/ferrous/ferrous oxide - one of the important micro nutrients
        av_fe = float(row['av_fe'])

        # Generic class based on av_fe value
        if av_fe < 2.5:
            av_fe_class = 'Low'
        elif av_fe > 4.5:
            av_fe_class = 'High'
        else:
            av_fe_class = 'Optimal'
        row[GroundTruthBuilder.FE_GENERIC_CLASS] = av_fe_class

    @staticmethod
    def build_mn_labels(row: Dict):
        # manganese/manganese oxide - one of the important micro nutrients
        av_mn = float(row['av_mn'])

        # Generic class based on av_p value
        if av_mn < 1:
            av_mn_class = 'Low'
        elif av_mn > 2:
            av_mn_class = 'High'
        else:
            av_mn_class = 'Optimal'
        row[GroundTruthBuilder.MN_GENERIC_CLASS] = av_mn_class

    @staticmethod
    def build_optimal_macro_nutrients_binary_label(row: Dict):
        av_p = float(row['av_p'])
        oc = float(row['oc'])

        # pattern followed: lowest of medium range to highest of optimal range
        # medium 58.2843 - 78.4596, optimum 80.7013 - 112.085
        av_p_optimal = 10 <= av_p <= 24.6
        # medium 203.995 - 291.421, optimum 293.663 - 392.298
        oc_optimal = 0.5 <= oc <= 0.75

        row[GroundTruthBuilder.OPTIMAL_MACRO_NUTRIENTS_CLASS] = av_p_optimal and oc_optimal

    @staticmethod
    def build_optimal_micro_nutrients_binary_label(row: Dict):
        av_fe = float(row['av_fe'])
        av_mn = float(row['av_mn'])

        # pattern followed: lowest of medium range to highest of optimal range
        # medium 3.1 - 4.0, optimum 4.0 - 8.0
        av_fe_optimal = 2.5 <= av_fe <= 4.5
        av_mn_optimal = 1 <= av_mn <= 2

        row[GroundTruthBuilder.OPTIMAL_MICRO_NUTRIENTS_CLASS] = av_fe_optimal and av_mn_optimal

    @staticmethod
    def build_generic_fertility_label(row: Dict):
        ph = float(row['ph'])
        ec = float(row['ec'])
        oc = float(row['oc'])
        av_p = float(row['av_p'])
        av_fe = float(row['av_fe'])
        av_mn = float(row['av_mn'])

        row[GroundTruthBuilder.GENERIC_FERTILITY_CLASS] = (6 <= ph <= 7.3) and ec <= 0.5 and (0.5 <= oc <= 0.75) and \
                                                          (10 <= av_p <= 24.6) and (2.5 <= av_fe <= 4.5) and \
                                                          (1 <= av_mn <= 2)

    @staticmethod
    def build_ph_macro_micro_all_optimal_label(row: Dict):
        av_p = float(row['av_p'])
        oc = float(row['oc'])

        # pattern followed: lowest of medium range to highest of optimal range
        # medium 58.2843 - 78.4596, optimum 80.7013 - 112.085
        av_p_optimal = 10 <= av_p <= 24.6
        # medium 203.995 - 291.421, optimum 293.663 - 392.298
        oc_optimal = 0.5 <= oc <= 0.75

        av_fe = float(row['av_fe'])
        av_mn = float(row['av_mn'])

        # pattern followed: lowest of medium range to highest of optimal range
        # medium 3.1 - 4.0, optimum 4.0 - 8.0
        av_fe_optimal = 2.5 <= av_fe <= 4.5
        av_mn_optimal = 1 <= av_mn <= 2

        ph = float(row['ph'])
        optimal_ph = 6 >= ph <= 7.3

        row[GroundTruthBuilder.PH_MACRO_MICRO_ALL_OPTIMAL_CLASS] = av_p_optimal and oc_optimal and av_fe_optimal and \
                                                                   av_mn_optimal and optimal_ph

    @staticmethod
    def build_ph_and_macro_micro_either_optimal_label(row: Dict):
        av_p = float(row['av_p'])
        oc = float(row['oc'])

        # pattern followed: lowest of medium range to highest of optimal range
        # medium 58.2843 - 78.4596, optimum 80.7013 - 112.085
        av_p_optimal = 10 <= av_p <= 24.6
        # medium 203.995 - 291.421, optimum 293.663 - 392.298
        oc_optimal = 0.5 <= oc <= 0.75

        av_fe = float(row['av_fe'])
        av_mn = float(row['av_mn'])

        # pattern followed: lowest of medium range to highest of optimal range
        # medium 3.1 - 4.0, optimum 4.0 - 8.0
        av_fe_optimal = 2.5 <= av_fe <= 4.5
        av_mn_optimal = 1 <= av_mn <= 2

        ph = float(row['ph'])
        optimal_ph = 6 >= ph <= 7.3

        row[GroundTruthBuilder.PH_AND_MACRO_MICRO_EITHER_OPTIMAL_CLASS] = (av_p_optimal and oc_optimal) or \
                                                                          (av_fe_optimal and av_mn_optimal) \
                                                                          and optimal_ph
