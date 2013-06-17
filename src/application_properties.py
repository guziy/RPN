
__author__="huziy"
__date__ ="$26 mai 2010 12:23:59$"

import os

PROJECT_DIR = 'RPN'


def set_current_directory():
    the_dir = os.getcwd()
    while not the_dir.endswith(PROJECT_DIR):
        os.chdir('..')
        the_dir = os.getcwd()
        if the_dir == "/":
            raise EnvironmentError("Was not able to set {0} as a current directory.".format(PROJECT_DIR))


if __name__ == "__main__":
    set_current_directory()
    print os.getcwd()
    print "Hello World"
