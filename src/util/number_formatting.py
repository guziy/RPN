__author__ = 'huziy'


def ordinal(value):
    """
    Source: http://code.activestate.com/recipes/576888-format-a-number-as-an-ordinal/
    Modified it a bit for python3....
    Converts zero or a *postive* integer (or their string
    representations) to an ordinal value.

    >>> for i in range(1, 13):
    ...     ordinal(i)
    '1st'
    '2nd'
    '3rd'
    '4th'
    '5th'
    '6th'
    '7th'
    '8th'
    '9th'
    '10th'
    '11th'
    '12th'

    >>> for i in (100, '111', '112', 1011):
    ...     ordinal(i)
    ...
    '100th'
    '111th'
    '112th'
    '1011th'

    """
    try:
        value = int(value)
    except ValueError:
        return value

    if value % 100 // 10 != 1:
        if value % 10 == 1:
            suffix = "st"
        elif value % 10 == 2:
            suffix = "nd"
        elif value % 10 == 3:
            suffix = "rd"
        else:
            suffix = "th"
    else:
        suffix = "th"

    ordval = "{}{}".format(value, suffix)

    return ordval


if __name__ == '__main__':
    for i in range(1, 13):
        print(ordinal(i))