from application_properties import main_decorator
from crcm5.mh_domains import show_domains
from crcm5.uqam_domain.qc_domains import qc_rll_domain_huziy_thesis_04


@main_decorator
def main():
    show_domains.show_domain(qc_rll_domain_huziy_thesis_04, show_Churchil_Nelson_basins=False, img_folder="qc", domain_label="qc")

if __name__ == '__main__':
    main()