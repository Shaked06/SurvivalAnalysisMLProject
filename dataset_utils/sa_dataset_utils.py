from pycox.datasets import metabric, gbsg, flchain, nwtco, support, kkbox

DATASETS = ['FLCHAIN', 'GBSG', 'METABRIC', 'NWTCO',
            'NWTCO', 'SUPPORT', 'KKBOX', 'prostateSurvival', 'hepatoCellular']

PYCOX_DATASETS = ['FLCHAIN', 'GBSG', 'METABRIC', 'NWTCO',
                  'NWTCO', 'SUPPORT', 'KKBOX']

ASAUR_DATASETS = ['prostateSurvival', 'hepatoCellular']

PYCOX_DATASET_TO_FUNC = {'FLCHAIN': flchain,
                         'GBSG': gbsg,
                         'METABRIC': metabric,
                         'NWTCO': nwtco,
                         'SUPPORT': support,
                         'KKBOX': kkbox}


DATASET_TARGET_DECODER = {'FLCHAIN': ("futime", "death"),
                          'GBSG': ("duration", "event"),
                          'METABRIC': ("duration", "event"),
                          'NWTCO': ("edrel", "rel"),
                          'SUPPORT': ("duration", "event"),
                          'KKBOX': ("duration", "event"),
                          'prostateSurvival': ("survTime", "status"),
                          'hepatoCellular': ("OS", "Death")}

DATASET_BINARY_COLS = {'FLCHAIN': ["sex", "sample.yr", "flc.grp", "mgus"],
                       'GBSG': ["x0", "x1", "x2"],
                       'METABRIC': ["x4", "x5", "x6", "x7"],
                       'NWTCO': ["stage", "in.subcohort", "instit_2", "histol_2", "study_4"],
                       'SUPPORT': ["x1", "x2", "x3", "x4", "x5", "x6"],
                       'KKBOX': ["n_prev_churns", "payment_method_id", "is_auto_renew",
                                 "is_cancel", "age_at_start", "strange_age", "nan_days_since_reg_init",
                                 "no_prev_churns"],
                       'prostateSurvival': ["grade_mode"],
                       'hepatoCellular': ["Gender", "HBsAg", "Cirrhosis", "ALT", "AST", "AFP", 
                                          "Tumorsize", "Tumordifferentiation", "Vascularinvasion", 
                                          "Tumormultiplicity", "Capsulation", "TNM", "BCLC"]

                       }
NUM_FEATURES_PER_DATASET = {'FLCHAIN': 8,
                            'GBSG': 7,
                            'METABRIC': 9,
                            'NWTCO': 6,
                            'SUPPORT': 14,
                            'KKBOX': 13,
                            'prostateSurvival': 3,
                            'hepatoCellular': 43
                           }
