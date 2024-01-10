from build_collocation_database import *

args = argparse.Namespace(
    year = [2017],
    month = [1],
    modisfolder = "./test_data/MODIS/",
    caliopfolder = "./test_data/CALIOP/")

main(args)