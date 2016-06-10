from datetime import datetime
from pathlib import Path
import numpy as np
from rpn import level_kinds
from rpn.rpn_multi import MultiRPN

from crcm5.nemo_vs_hostetler.nemo_hl_util import get_monthyeardate_to_paths_map


class RPNLakeIceManager(object):
    def __init__(self, samples_dir="", start_year=-np.Inf, end_year=np.Inf):
        self.samples_dir = Path(samples_dir)

        self.year_month_date_to_file_list = get_monthyeardate_to_paths_map(file_prefix="pm", start_year=start_year,
                                                                           end_year=end_year,
                                                                           samples_dir_path=self.samples_dir)

        self.varname = "LC"
        self.level = -1
        self.level_kind = level_kinds.ARBITRARY

        self.cached_data = {}

    def get_lake_fraction_for_date(self, the_date=None):

        """
        Get the lake ice cover fraction for the specified date
        :param the_date:
        :return:
        """
        if the_date not in self.cached_data:
            month_date = datetime(the_date.year, the_date.month, 1)

            self.cached_data = {}
            mr = MultiRPN(self.year_month_date_to_file_list[month_date])
            self.cached_data = mr.get_all_time_records_for_name_and_level(varname=self.varname, level=self.level,
                                                                          level_kind=self.level_kind)
            mr.close()

        # if still it is not there try looking in the previous month
        if the_date not in self.cached_data:

            month = the_date.month - 1
            year = the_date.year
            if the_date.month == 0:
                month = 12
                year -= 1


            month_date = datetime(year, month, 1)

            mr = MultiRPN(self.year_month_date_to_file_list[month_date])
            self.cached_data.update(mr.get_all_time_records_for_name_and_level(varname=self.varname, level=self.level,
                                                                               level_kind=self.level_kind))
            mr.close()

        return self.cached_data[the_date]
