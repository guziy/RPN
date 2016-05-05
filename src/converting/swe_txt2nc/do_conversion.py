from pathlib import Path

from application_properties import main_decorator

# To convert cmc swe analysis data 1998 - 2013

@main_decorator
def main():
    in_path = "/HOME/huziy/skynet3_rech1/swe_ross_brown/cmc_1998-2013"


    p = Path(in_path)



if __name__ == '__main__':
    main()
