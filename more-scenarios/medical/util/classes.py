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
0	Female
1	Male
"""
PRIMARY_DEMOGRAPHICS = {
    '31': 'sex',
    '21003': 'age',
    '21000': 'ethnicity'
}


"""
coding	meaning	
-3	Prefer not to answer
-1	Do not know
1	White
2	Mixed
3	Asian or Asian British
4	Black or Black British
5	Chinese
6	Other ethnic group
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
    '5': 'Chinese',
}
