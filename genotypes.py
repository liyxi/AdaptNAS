from collections import namedtuple


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

arch_AdaptNAS = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_5x5', 0),
            ('sep_conv_5x5', 2), ('sep_conv_5x5', 3)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('skip_connect', 0),
            ('max_pool_3x3', 2), ('skip_connect', 0),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 1),
            ('dil_conv_3x3', 4), ('dil_conv_5x5', 1)],
    reduce_concat=range(2, 6))
