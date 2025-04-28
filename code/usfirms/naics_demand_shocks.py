from utils import *


class NaicsDemandShocks:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    # Private methods
    def _construct_gdp_shocks(self):
        """
        Constructs trade shocks for naics level using gdp data
        """
        # TODO

    def _construct_exchange_rate_shocks(self):
        """
        Constructs trade shocks for naics level using exchange rate data
        """
        # TODO

    def _construct_shea_shocks(self):
        """
        Constructs shocks for naics level using data on intermediate inputs
        """
        # TODO

    def _compile_naics_data(self):
        """
        Compile naics data on prices, quantities, investment
        """
        # TODO

    # Public methods
    def initialize_data(self):
        """
        Construct full data
        """
        gdp_shocks = self._construct_gdp_shocks()
        exchange_rate_shocks = self._construct_exchange_rate_shocks()
        shea_shocks = self._construct_shea_shocks()
        naics_data = self._compile_naics_data()
        # TODO: merge all and construct df
        # self.data = df

    def run_regressions(self):
        """
        Run regressions on naics data
        """
