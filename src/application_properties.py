
__author__="huziy"
__date__ ="$26 mai 2010 12:23:59$"

import os

PROJECT_DIR = 'RPN'


def set_current_directory():
    dir = os.getcwd()
    while not dir.endswith(PROJECT_DIR):
        os.chdir('..')
        dir = os.getcwd()

if __name__ == "__main__":

    set_current_directory()
    print os.getcwd()
    print "Hello World"
