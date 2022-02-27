import csv
from typing import Dict
import os

import pandas


class GroundTruthBuilder(object):
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

    # May need to specify what are the tolerable avg distances from the neighbors for them not to be outliers when using
    # kNN algorithm.
    FEATURES_TO_LABELS_DICT = {
        'ph': [PH_GENERIC_CLASS, PH_VEG_ROW_CROPS_CLASS, PH_FRUITS_NUTS_CLASS],
        'ec': [EC_GENERIC_CLASS],
        'oc': [OC_GENERIC_CLASS],
        'av_p': [P_GENERIC_CLASS],
        'av_fe': [FE_GENERIC_CLASS],
        'av_mn': [MN_GENERIC_CLASS],
        'av_p,av_k,av_s': [OPTIMAL_MACRO_NUTRIENTS_CLASS],
        'av_zn,av_cu,av_mn': [OPTIMAL_MICRO_NUTRIENTS_CLASS]
    }

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
        if av_mn < 2.5:
            av_mn_class = 'Low'
        elif av_mn > 4.5:
            av_mn_class = 'High'
        else:
            av_mn_class = 'Optimal'
        row[GroundTruthBuilder.MN_GENERIC_CLASS] = av_mn_class

    @staticmethod
    def build_optimal_macro_nutrients_binary_label(row: Dict):
        av_p = float(row['av_p'])
        av_k = float(row['av_k'])
        av_s = float(row['av_s'])

        # pattern followed: lowest of medium range to highest of optimal range
        # medium 58.2843 - 78.4596, optimum 80.7013 - 112.085
        av_p_optimal = 58.2843 <= av_p <= 112.085
        # medium 203.995 - 291.421, optimum 293.663 - 392.298
        av_k_optimal = 203.995 <= av_k <= 392.298

        row[GroundTruthBuilder.OPTIMAL_MACRO_NUTRIENTS_CLASS] = av_p_optimal and av_k_optimal and av_s <= 10

    @staticmethod
    def build_optimal_micro_nutrients_binary_label(row: Dict):
        av_zn = float(row['av_zn'])
        av_cu = float(row['av_cu'])
        av_mn = float(row['av_mn'])

        # pattern followed: lowest of medium range to highest of optimal range
        # medium 3.1 - 4.0, optimum 4.0 - 8.0
        av_zn_optimal = 3.1 <= av_zn <= 8.0

        row[GroundTruthBuilder.OPTIMAL_MICRO_NUTRIENTS_CLASS] = av_zn_optimal and av_cu >= 1.0 and av_mn >= 40
