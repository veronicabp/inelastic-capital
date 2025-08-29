# %%
import utils
from utils import *

from dotenv import load_dotenv

load_dotenv()

# %%
reload(utils)

# %%
data_folder = os.path.join("..", "data", "inelastic-capital-data")
