

# should invert all the directions down the old path from exit points up to the enterpoint


to_invert = {
    "exitpoint-1": { # i.e should become an exit of the stream-1
        "description": "Near Notigi, outlet of the Rat lake",
        "lon": -99.345833333319618,
        "lat": 55.862500000000146,
        "acc_old": 98,
        "dir_old": 128,
        "dir_new": [1, 1, 2], # should be changed down the new downstream

        "enterpoint": { # Kind of close to the entry point of the stream
            "description": "near the natural water divide",
            "lon": -99.095833333319419,
            "lat": 56.645833333333435,
            "acc_old": 12506,
            "dir_old": 1,
            "dir_new": 8, # should be changed down the new downstream
        }
    }
}


to_modify = {
    "singlepoint-1": {
        "lon": -99.095833333319419,
        "lat": 56.654166666666768,
        "acc": 19,
        "dir_old": 4,
        "dir_new": 2
    },
    "singlepoint-2": {
        "lon": -99.087499999986079,
        "lat": 56.637500000000102,
        "acc": 62,
        "dir_old": 128,
        "dir_new": 32
    }
}