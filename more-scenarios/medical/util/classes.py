CLASSES = {
    'acdc': ['Right Ventricle', 'Myocardium', 'Left Ventricle'],
    'ukbb': ['Right Ventricle', 'Myocardium', 'Left Ventricle']
}


MASK = {
    'rv': 1,
    'myo': 2,
    'lv': 3
}


FRAME = {
    'frame01': 'sa_ED',
    'frame02': 'sa_ES'
}


"""
Sex

code meaning
0	Female
1	Male
"""
PRIMARY_DEMOGRAPHICS = {
    '31': 'sex',
    '21003': 'age',
    '21000': 'ethnicity'
}


"""
Ethnicity

coding	meaning	
-3	    Prefer not to answer
-1	    Do not know
1	    White
2	    Mixed
3	    Asian or Asian British
4	    Black or Black British
5	    Chinese
6	    Other ethnic group
1001	British
1002	Irish
1003	Any other white background
2001	White and Black Caribbean
2002	White and Black African
2003	White and Asian
2004	Any other mixed background
3001	Indian
3002	Pakistani
3003	Bangladeshi
3004	Any other Asian background
4001	Caribbean
4002	African
4003	Any other Black background
"""
ETHNNICITY_CODING = {
    '1': 'White',
    '3': 'Asian',
    '4': 'Black',
}


ETHNNICITY_CODING_REVERSED = {
    'White': '1',
    'Asian': '3',
    'Black': '4',
}


EXPERIMENTS_SPLITS = {
    '2': 80,
    '3': 26,
    '4': 18
}


EXPERIMENTS = [2, 3, 4]


EXPERIMENTS_DEFINITION = {
    '2': 'Sex Controlled',
    '3': 'Ethnicity Controlled',
    '4': 'Sex Ethnicity Controlled'
}


TESTSETS = ['sex', 'ethn']


SEEDS = [42, 43, 47, 53, 57, 61, 71, 73, 79, 83]


ETHNICITIES = ['white', 'asian', 'black']
