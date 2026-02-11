
from functools import partial
from pathlib import Path

from tqdm.auto import tqdm


def drawing_name_to_datetime(name):
    name2= name
    name2 = name2.replace('usd','')
    name2 = name2.split('-')[0]

    year = name2[0:4]
    month = name2[4:6]
    day = name2[6:8]
    hours = name2[8:10]
    minutes = name2[10:12]

    return {'year': year, 'month': month, 'day':day, 'hours':hours, 'minutes':minutes}

def whitelight_to_datetime(name):
    name2= name
    name2 = name2.replace('UPH','')

    year = name2[0:4]
    month = name2[4:6]
    day = name2[6:8]
    hours = name2[8:10]
    minutes = name2[10:12]
    seconds = name2[12:14]

    return {'year': year, 'month': month, 'day':day, 'hours':hours, 'minutes':minutes, "seconds": seconds}

def datetime_to_drawing_name(datetime):
    name = 'usd'
    name += f'{datetime["year"]}{datetime["month"]}{datetime["day"]}{datetime["hours"]}{datetime["minutes"]}'
    name += '.jpg'

    return name
def datetime_to_drawing_name2(datetime):
    name = 'usd'
    name += f'{datetime["year"]}{datetime["month"]}{datetime["day"]}{datetime["hours"]}{datetime["minutes"]}'
    name += '-HO-AV.jpg'

    return name

def datetime_to_whiteLight_string(datetime):
    name = 'UPH'
    name += f'{datetime["year"]}{datetime["month"]}{datetime["day"]}{datetime["hours"]}{datetime["minutes"]}'
    name += '.FTS'

    return name

def datetime_to_db_string(datetime):
    date = '-'.join([datetime['year'],datetime['month'], datetime['day']])
    time = ':'.join([datetime['hours'], datetime['minutes'], '00'])
    
    return f'{date} {time}'


def db_string_to_datetime(db_string):
    date_str, time_str = db_string.split(" ") 
    year, month, day = date_str.split('-')
    hours, minutes, seconds = time_str.split(':')
    
    return {'year': year, 'month': month, 'day':day, 'hours':hours, 'minutes':minutes, "seconds": seconds}

def whitelight_to_datetime(name):
    name2= name
    name2 = name2.replace('UPH','')

    year = name2[0:4]
    month = name2[4:6]
    day = name2[6:8]
    hours = name2[8:10]
    minutes = name2[10:12]
    seconds = name2[12:14]

    return {'year': year, 'month': month, 'day':day, 'hours':hours, 'minutes':minutes, "seconds": seconds}

def calcium_to_datetime(name):
    name2= name
    name2 = name2.replace('UCC','')

    year = name2[0:4]
    month = name2[4:6]
    day = name2[6:8]
    hours = name2[8:10]
    minutes = name2[10:12]
    seconds = name2[12:14]

    return {'year': year, 'month': month, 'day':day, 'hours':hours, 'minutes':minutes, "seconds": seconds}


def datetime_to_db_string(datetime):
    date = '-'.join([datetime['year'],datetime['month'], datetime['day']])
    time = ':'.join([datetime['hours'], datetime['minutes'], '00'])
    
    return f'{date} {time}'

def datetime_to_string(date):
    return f"{date['year']}-{date['month']}-{date['day']}"
