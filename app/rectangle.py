from typing import TypedDict

class Rectangle(TypedDict):
    x_min: int
    x_max: int
    y_max: int
    y_min: int
    width: int
    height: int
    center_x: int
    center_y: int
    area: int