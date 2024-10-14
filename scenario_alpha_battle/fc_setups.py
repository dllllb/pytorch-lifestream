RAW = {'compression_task': 'composite_compression',
       'need_pretrain': False,
                    'common': dict(save_each=10),
                    'model_params': dict(
                                    training_model=dict(
                                             epochs=100,
                                         )                    
                                    ),  
                    'initial_assumption': [
                        'training_model',
                                              ]}

COMPOSITE_1 = {'compression_task': 'composite_compression',
               'need_pretrain': False,
                    'common': dict(save_each=-1),
                    'model_params': dict(pruning_model=dict(epochs=5,
                                                            pruning_iterations=5,
                                                            learning_rate=0.001,
                                                            importance='MagnitudeImportance',
                                                            pruner_name='magnitude_pruner',
                                                            importance_norm=1,
                                                            pruning_ratio=0.75,
                                                            finetune_params={'epochs': 10,
                                                                             'custom_loss': None}
                                                            ),
                                         low_rank_model=dict(epochs=20,
                                                             learning_rate=0.001,
                                                             hoyer_loss=0.2,
                                                             energy_thresholds=[0.9],
                                                             orthogonal_loss=5,
                                                             decomposing_mode='channel',
                                                             spectrum_pruning_strategy='energy',
                                                             finetune_params={'epochs': 10,
                                                                              'custom_loss': None}
                                                             ),
                                         training_model=dict(
                                             epochs=10,
                                         )                    
                                         ),  
                    'initial_assumption': [
                        'training_model',
                                            'pruning_model', 
                                            'low_rank_model',
                                              'pruning_model'
                                              ]}

COMPOSITE_2 = {'compression_task': 'composite_compression',
               'need_pretrain': False,
                    'common': dict(save_each=-1),
                    'model_params': dict(pruning_model=dict(epochs=20,
                                                            pruning_iterations=3,
                                                            learning_rate=0.001,
                                                            importance='MagnitudeImportance',
                                                            pruner_name='magnitude_pruner',
                                                            importance_norm=1,
                                                            pruning_ratio=0.75,
                                                            finetune_params={'epochs': 10,
                                                                             'custom_loss': None}
                                                            ),
                                         low_rank_model=dict(epochs=10,
                                                             learning_rate=0.001,
                                                             hoyer_loss=0.2,
                                                             energy_thresholds=[0.9],
                                                             orthogonal_loss=5,
                                                             decomposing_mode='channel',
                                                             spectrum_pruning_strategy='energy',
                                                             finetune_params={'epochs': 10,
                                                                              'custom_loss': None}
                                                             ),
                                         training_model=dict(
                                             epochs=30,
                                         )                    
                                         ),  
                    'initial_assumption': [
                        'training_model',
                        'low_rank_model',
                        'pruning_model',
                        'low_rank_model'
                    ]
}
TEST = {'compression_task': 'composite_compression',
        'need_pretrain': False,
                    'common': dict(save_each=-1),
                    'model_params': dict(pruning_model=dict(epochs=1,
                                                            pruning_iterations=3,
                                                            learning_rate=0.001,
                                                            importance='MagnitudeImportance',
                                                            pruner_name='magnitude_pruner',
                                                            importance_norm=1,
                                                            pruning_ratio=0.75,
                                                            finetune_params={'epochs': 1,
                                                                             'custom_loss': None}
                                                            ),
                                         low_rank_model=dict(epochs=1,
                                                             learning_rate=0.001,
                                                             hoyer_loss=0.2,
                                                             energy_thresholds=[0.9],
                                                             orthogonal_loss=5,
                                                             decomposing_mode='channel',
                                                             spectrum_pruning_strategy='energy',
                                                             finetune_params={'epochs': 1,
                                                                              'custom_loss': None}
                                                             ),
                                         training_model=dict(
                                             epochs=1,
                                         )                    
                                         ),  
                    'initial_assumption': [
                        'training_model',
                        'low_rank_model',
                        'pruning_model',
                    ]
}

SETUPS = {
    'raw': RAW,
    'test': TEST,
    'composite_1': COMPOSITE_1,
    'composite_2': COMPOSITE_2
}
