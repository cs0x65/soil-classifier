accuracy_dict = {'ph': [{'knn': 1.0, 'binary': False}, {'svm-ovo': 1.0, 'binary': False},
                        {'binary_relevance': 0.8790979927465977, 'binary': False},
                        {'classifier_chain': 0.8790979927465977, 'binary': False},
                        {'label_power_set_gnb': 0.9536069517756472, 'binary': False},
                        {'label_power_set_svm': 1.0, 'binary': False}, {'multilearn_knn': 1.0, 'binary': False}],
                 'ec': [{'knn': 1.0, 'binary': False}, {'svm-ovo': 0.9970555495709003, 'binary': False}],
                 'oc': [{'knn': 1.0, 'binary': False}, {'svm-ovo': 1.0, 'binary': False}],
                 'av_p': [{'knn': 1.0, 'binary': False}, {'svm-ovo': 1.0, 'binary': False}],
                 'av_fe': [{'knn': 1.0, 'binary': False}, {'svm-ovo': 0.9985636827175123, 'binary': False}],
                 'av_mn': [{'knn': 1.0, 'binary': False}, {'svm-ovo': 0.9974146288915221, 'binary': False}],
                 'oc,av_p': [{'knn': 0.9974146288915221, 'binary': True},
                             {'linear-svm': 0.9269991741175626, 'binary': True},
                             {'logr-ovr': 0.9321699163345183, 'binary': True}, {'rf': 1.0, 'binary': True}],
                 'av_fe,av_mn': [{'knn': 0.9994613810190671, 'binary': True},
                                 {'linear-svm': 0.9905562138676434, 'binary': True},
                                 {'logr-ovr': 0.9898739631584617, 'binary': True}, {'rf': 1.0, 'binary': True}],
                 'ph,ec,oc,av_p,av_fe,av_mn': [{'knn': 0.9999281841358756, 'binary': True},
                                               {'linear-svm': 0.9999281841358756, 'binary': True},
                                               {'logr-ovr': 0.9999281841358756, 'binary': True},
                                               {'rf': 0.9999281841358756, 'binary': True}],
                 'oc,av_p,av_fe,av_mn,ph': [{'knn': 1.0, 'binary': True}, {'linear-svm': 1.0, 'binary': True},
                                            {'logr-ovr': 1.0, 'binary': True}, {'rf': 1.0, 'binary': True}]}


import json

print(json.dumps(accuracy_dict, indent=1))
