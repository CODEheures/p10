from typing import TypedDict
import pandas as pd

class Dfs(TypedDict):
    images: pd.DataFrame
    boxes: pd.DataFrame